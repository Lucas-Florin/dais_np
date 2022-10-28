import copy
import logging
import os
import types
from itertools import product
from typing import List, Optional, Tuple

import numpy as np
import torch
import yaml
from metalearning_benchmarks import (
    MetaLearningBenchmark,
    MetaLearningTask,
)
from np_util.datasets import MetaLearningDataset
from np_util.tqdm_logging_handler import TqdmLoggingHandler
from torch.utils.data import DataLoader
from tqdm import tqdm

from neural_process.aggregator import (
    BayesianAggregator,
    MeanAggregatorRtoZ,
)
from neural_process.decoder_network import DecoderNetworkSamples
from neural_process.encoder_network import (
    EncoderNetworkBA,
    EncoderNetworkMA,
)

from neural_process.dais import differentiable_annealed_importance_sampling


class NeuralProcess:
    _f_settings = "000_settings.txt"
    _f_n_tasks_seen = "000_n_tasks_seen.txt"
    _available_aggregator_types = ["BA", "MA"]
    _available_loss_types = ["VI", "MC", "DAIS"]
    _available_input_mlp_std_y = ["xz", "x", "z", "cov_z", ""]

    def __init__(
        self,
        logpath: str,
        seed: int,
        d_x: int,
        d_y: int,
        d_z: int,
        n_context: Tuple,
        aggregator_type: str = "BA",
        loss_type: str = "MC",
        input_mlp_std_y: Optional[str] = None,
        latent_prior_scale: float = 1.0,
        f_act: str = "relu",
        n_hidden_layers: int = 2,
        n_hidden_units: int = 16,
        decoder_output_scale: float = 1.0,
        decoder_output_scale_min: Optional[float] = None,
        device: str = "cpu",
        adam_lr: float = 1e-4,
        batch_size: int = 16,
        n_samples: int = 16,
        n_annealing_steps: int = 10,
        dais_step_size: float = 0.08,
    ):
        # build config
        self._config = self._build_config(
            logpath=logpath,
            seed=seed,
            d_x=d_x,
            d_y=d_y,
            d_z=d_z,
            n_context=n_context,
            aggregator_type=aggregator_type,
            loss_type=loss_type,
            input_mlp_std_y=input_mlp_std_y,
            latent_prior_scale=latent_prior_scale,
            f_act=f_act,
            n_hidden_layers=n_hidden_layers,
            n_hidden_units=n_hidden_units,
            decoder_output_scale=decoder_output_scale,
            decoder_output_scale_min=decoder_output_scale_min,
            device=device,
            adam_lr=adam_lr,
            batch_size=batch_size,
            n_samples=n_samples,
            n_annealing_steps=n_annealing_steps,
            dais_step_size=dais_step_size
        )

        # logging
        assert os.path.isdir(logpath)
        self._logpath = logpath
        self._logger = None
        self._configure_logger()

        # write config to file
        self._write_config_to_file()

        # initialize random number generator
        self._rng = np.random.RandomState()
        self._seed(self._config["seed"])

        # set n_meta_tasks_seen and write it to file
        self._n_meta_tasks_seen = 0
        self._write_n_meta_tasks_seen_to_file()

        # initialize architecture
        self._modules = []
        self._create_architecture()
        self._set_device(device)

        # initialize optimizer
        self._optimizer = None
        self._create_optimizer()

        self._logger.info("Initialized new model of type {}...".format(type(self)))

    @staticmethod
    def get_valid_model_specs() -> List[dict]:
        model_specs = []
        for (
            aggregator_type,
            loss_type,
            input_mlp_std_y,
        ) in product(
            NeuralProcess._available_aggregator_types,
            NeuralProcess._available_loss_types,
            NeuralProcess._available_input_mlp_std_y,
        ):
            model_specs.append(
                {
                    "aggregator_type": aggregator_type,
                    "loss_type": loss_type,
                    "input_mlp_std_y": input_mlp_std_y,
                }
            )
        return model_specs

    @staticmethod
    def is_valid_model_spec(model_spec: dict) -> bool:
        ms = copy.deepcopy(model_spec)
        return ms in NeuralProcess.get_valid_model_specs()

    @staticmethod
    def _build_config(
        logpath: int,
        seed: int,
        d_x: int,
        d_y: int,
        d_z: int,
        n_context: Tuple,
        aggregator_type: str,
        loss_type: str,
        input_mlp_std_y: Optional[str],
        f_act: str,
        n_hidden_layers: int,
        n_hidden_units: int,
        latent_prior_scale: float,
        decoder_output_scale: float,
        decoder_output_scale_min: float,
        device: str,
        adam_lr: float,
        batch_size: int,
        n_samples: int,
        n_annealing_steps: int,
        dais_step_size: float,
    ) -> dict:
        config = {
            "logpath": logpath,
            "seed": seed,
            "d_x": d_x,
            "d_y": d_y,
            "d_z": d_z,
            "aggregator_type": aggregator_type,
            "loss_type": loss_type,
            "input_mlp_std_y": input_mlp_std_y,
            "decoder_output_scale_min": decoder_output_scale_min,
            "f_act": f_act,
            "n_hidden_layers": n_hidden_layers,
            "n_hidden_units": n_hidden_units,
            "adam_lr": adam_lr,
            "batch_size": batch_size,
            "n_context_meta": n_context,
            "n_context_val": n_context,
            "device": device,
        }

        # check model spec
        model_spec = {
            "aggregator_type": config["aggregator_type"],
            "loss_type": config["loss_type"],
            "input_mlp_std_y": config["input_mlp_std_y"],
        }
        assert NeuralProcess.is_valid_model_spec(model_spec)

        # network architecture for encoder and decoder networks
        network_arch = config["n_hidden_layers"] * [config["n_hidden_units"]]

        # decoder kwargs
        decoder_kwargs = {
            "mlp_layers_mu_y": network_arch,
            "input_mlp_std_y": config["input_mlp_std_y"],
            "decoder_output_scale_min": config["decoder_output_scale_min"],
        }

        decoder_kwargs["arch"] = "separate_networks"
        if config["input_mlp_std_y"] != "":
            decoder_kwargs["mlp_layers_std_y"] = network_arch
        else:
            decoder_kwargs["global_std_y"] = decoder_output_scale
            decoder_kwargs["global_std_y_is_learnable"] = False

        # encoder_kwargs
        if config["aggregator_type"] == "BA":
            encoder_kwargs = {
                "arch": "separate_networks",
                "mlp_layers_r": network_arch,
                "mlp_layers_cov_r": network_arch,
            }
        else:  # MA or MAX
            encoder_kwargs = {"mlp_layers_r": network_arch}

        # aggregator_kwargs
        if config["aggregator_type"] == "MA":
            aggregator_kwargs = {
                "arch": "separate_networks",
                "mlp_layers_r_to_mu_z": 1 * [config["n_hidden_units"]],
                "mlp_layers_r_to_cov_z": 1 * [config["n_hidden_units"]],
            }
        elif config["aggregator_type"] == "BA":
            aggregator_kwargs = {
                "var_z_0": latent_prior_scale,
                "var_z_0_is_learnable": False,
            }
        else:
            aggregator_kwargs = {}

        # loss_kwargs
        if config["loss_type"] in {"MC", "DAIS"}:
            loss_kwargs = {
                "n_marg": n_samples,
                "n_steps": n_annealing_steps,
                'step_size': dais_step_size,
            }
        else:  # loss_type == "VI"
            loss_kwargs = {}

        config.update(
            {
                "encoder_kwargs": encoder_kwargs,
                "aggregator_kwargs": aggregator_kwargs,
                "decoder_kwargs": decoder_kwargs,
                "loss_kwargs": loss_kwargs,
            }
        )

        return config

    @property
    def settings(self) -> dict:
        return self._config

    @property
    def n_meta_tasks_seen(self) -> int:
        return self._n_meta_tasks_seen

    @property
    def parameters(self):
        """
        Returns an iterable of parameters that are trainable in the model.
        """
        parameters = []
        for module in self._modules:
            if isinstance(module.parameters, list):
                parameters += module.parameters
            else:
                assert isinstance(module.parameters(), types.GeneratorType)
                parameters += list(module.parameters())
        return parameters

    def _configure_logger(self):
        """
        Creates a logger and pairs it with the tqdm-progress logging handler.
        """
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)

        # define format
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # write to stderr
        # the logger might already have a TqdmLoggingHandler
        has_sh = any(
            [
                isinstance(handler, TqdmLoggingHandler)
                for handler in self._logger.handlers
            ]
        )
        if not has_sh:
            sh = TqdmLoggingHandler()
            sh.setLevel(logging.DEBUG)
            sh.setFormatter(formatter)
            self._logger.addHandler(sh)

    def _create_architecture(self) -> None:
        # create encoder
        if self._config["aggregator_type"] == "BA":
            encoder = EncoderNetworkBA
        else:  # MeanAggregator or MaxAggregator
            encoder = EncoderNetworkMA
        self.encoder = encoder(
            d_x=self._config["d_x"],
            d_y=self._config["d_y"],
            d_r=self._config["d_z"],  # we set d_r == d_z
            f_act=self._config["f_act"],
            seed=self._config["seed"],
            **self._config["encoder_kwargs"],
        )

        # create aggregator
        if self._config["aggregator_type"] == "BA":
            aggregator = BayesianAggregator
        else:  # MeanAggregator (w/ or w/o self-attention)
            aggregator = MeanAggregatorRtoZ
        self.aggregator = aggregator(
            d_r=self._config["d_z"],  # we set d_r == d_z
            d_z=self._config["d_z"],
            f_act=self._config["f_act"],
            seed=self._config["seed"],
            **self._config["aggregator_kwargs"],
        )

        # create decoder
        decoder = DecoderNetworkSamples
        self.decoder = decoder(
            d_x=self._config["d_x"],
            d_y=self._config["d_y"],
            d_z=self._config["d_z"],
            f_act=self._config["f_act"],
            seed=self._config["seed"],
            **self._config["decoder_kwargs"],
        )

        self._modules = [self.encoder, self.aggregator, self.decoder]

    def _create_optimizer(self) -> None:
        self._optimizer = torch.optim.Adam(
            params=self.parameters, lr=self._config["adam_lr"]
        )

    def _set_device(self, device):
        if device == "cuda" and not torch.cuda.is_available():
            self._logger.warning("CUDA not available! Using CPU instead!")
            self.device = "cpu"
        else:
            self.device = device

        for module in self._modules:
            module.to(self.device)

    def _write_config_to_file(self):
        with open(os.path.join(self._logpath, self._f_settings), "w") as f:
            yaml.safe_dump(self._config, f)

    def _load_config_from_file(self, load_path: str = None):
        with open(os.path.join(self._logpath if load_path is None else load_path, self._f_settings), "r") as f:
            config = yaml.safe_load(f)

        return config

    def _write_n_meta_tasks_seen_to_file(self):
        with open(os.path.join(self._logpath, self._f_n_tasks_seen), "w") as f:
            yaml.safe_dump(self._n_meta_tasks_seen, f)

    def _load_n_meta_tasks_seen_from_file(self, load_path: str = None):
        with open(os.path.join(self._logpath if load_path is None else load_path, self._f_n_tasks_seen), "r") as f:
            epoch = yaml.safe_load(f)

        return epoch

    def _load_weights_from_file(self, load_path: str = None):
        for module in self._modules:
            module.load_weights(self._logpath, self._n_meta_tasks_seen)

    def _seed(self, seed: int) -> None:
        self._rng.seed(seed=seed)

    def _collate_batch(
        self,
        task_list: List[MetaLearningTask],
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # collect sizes
        n_tsk = len(task_list)
        n_points_per_task = task_list[0].n_points
        d_x = task_list[0].d_x
        d_y = task_list[0].d_y

        # collect all tasks
        x = torch.zeros((n_tsk, n_points_per_task, d_x))
        y = torch.zeros((n_tsk, n_points_per_task, d_y))
        for i, task in enumerate(task_list):
            x[i] = torch.tensor(task.x)
            y[i] = torch.tensor(task.y)

        # send to device
        x, y = x.to(self.device), y.to(self.device)

        return x, y

    def _compute_loss(
        self, x: torch.Tensor, y: torch.Tensor, mode: str
    ) -> torch.Tensor:
        # determine loss_type, loss_kwargs and ctx_steps
        if mode == "meta":
            # use the loss_type and loss_kwargs given in settings
            loss_type = self._config["loss_type"]
            loss_kwargs = self._config["loss_kwargs"]

            # sample a random context set size
            ctx_steps = [
                self._rng.randint(
                    low=self._config["n_context_meta"][0],
                    high=self._config["n_context_meta"][1] + 1,
                    size=(1,),
                ).item()
            ]
            n_ctx = max(ctx_steps)
        elif mode == "val":
            # TODO: is it correct to evaluate MC also for IWMC-models?
            loss_type = "MC"
            loss_kwargs = {
                "n_marg": 500,  # TODO: how many samples to use?
            }

            # average over some context set sizes
            ctx_steps = list(
                np.linspace(
                    self.settings["n_context_val"][0],
                    self.settings["n_context_val"][1],
                    num=min(
                        5,
                        (
                            self.settings["n_context_val"][1]
                            - self.settings["n_context_val"][0]
                        )
                        + 1,
                    ),  # TODO: how many context steps to use?
                    dtype=np.int,
                )
            )
            n_ctx = max(ctx_steps)
        else:
            raise ValueError("Unknown value of argument 'mode'!")

        # create context and test sets
        x_ctx, y_ctx, x_tgt, y_tgt, latent_obs_all = self._create_ctx_tst_sets(
            x_all=x, y_all=y, n_ctx=n_ctx, loss_type=loss_type
        )

        # compute loss for all context set sizes
        loss = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        for i in range(len(ctx_steps)):
            # reset aggregator
            self.aggregator.reset(x_ctx.shape[0])

            # encode current context set
            # (due to self-attention, we cannot encode the whole context set at once)
            cur_latent_obs = self.encoder.encode(
                x=x_ctx[:, : ctx_steps[i], :], y=y_ctx[:, : ctx_steps[i], :]
            )
            # due to self-attention, we cannot use incremental updates
            self.aggregator.update(latent_obs=cur_latent_obs, incremental=False)

            # compute loss
            ls_ctx = self.aggregator.last_latent_state
            agg_state_ctx = self.aggregator.last_agg_state

            # TODO: remove ls-dimension altogether for backwards compatibility
            # add dummy ls-dimension
            ls_ctx_dummy = []
            for i in range(len(ls_ctx)):
                if ls_ctx[i] is not None:
                    ls_ctx_dummy.append(ls_ctx[i][:, None, :])
                else:
                    ls_ctx_dummy.append(None)
            ls_ctx = tuple(ls_ctx_dummy)
            agg_state_ctx = tuple(
                agg_state_ctx[i][:, None, :] for i in range(len(agg_state_ctx))
            )

            # decode latent state
            mu_z_ctx = ls_ctx[0]
            cov_z_ctx = ls_ctx[1]
            assert mu_z_ctx.ndim == 3
            if cov_z_ctx is not None:
                assert cov_z_ctx.ndim == 3
                assert mu_z_ctx.shape[1] == cov_z_ctx.shape[1] == 1


            if loss_type == "MC":
                loss = loss - self._log_marg_lhd_np_mc(
                    x_tgt=x_tgt,
                    y_tgt=y_tgt,
                    mu_z_ctx=mu_z_ctx,
                    cov_z_ctx=cov_z_ctx,
                    n_marg=loss_kwargs["n_marg"],
                )
            elif loss_type == "DAIS":
                loss = loss - self._log_marg_lhd_np_dais(
                    x_tgt=x_tgt,
                    y_tgt=y_tgt,
                    mu_z_ctx=mu_z_ctx,
                    cov_z_ctx=cov_z_ctx,
                    n_marg=loss_kwargs["n_marg"],
                    n_steps=loss_kwargs['n_steps'],
                    step_size=loss_kwargs['step_size'],
                )
            elif loss_type == "VI":
                loss = loss - self._elbo_np(
                    x_tgt=x_tgt,
                    y_tgt=y_tgt,
                    mu_z_ctx=mu_z_ctx,
                    cov_z_ctx=cov_z_ctx,
                    agg_state_ctx=agg_state_ctx,
                    latent_obs_all=latent_obs_all,
                )
            else:
                raise ValueError("Unknown loss_type '{}'!".format(loss_type))

        # average loss over number of context sets evaluated
        loss = loss / len(ctx_steps)

        return loss

    def save_model(self):
        self._logger.info("Saving model...")
        self._logger.info(self._logpath)
        for module in self._modules:
            module.save_weights(self._logpath, self._n_meta_tasks_seen)
        self._write_n_meta_tasks_seen_to_file()

    def load_model(self, load_n_meta_tasks_seen: int = -1, load_path: str = None) -> None:
        assert isinstance(load_n_meta_tasks_seen, int)

        # load settings
        config = self._load_config_from_file(load_path)
        config_a = copy.deepcopy(config)
        config_b = copy.deepcopy(self._config)
        del config_b['logpath']
        del config_a['logpath']
        assert config_a == config_b

        # load n_meta_tasks_seen
        if load_n_meta_tasks_seen == -1:  # load latest checkpoint
            self._n_meta_tasks_seen = self._load_n_meta_tasks_seen_from_file()
        else:
            self._n_meta_tasks_seen = load_n_meta_tasks_seen

        self._logger.info(
            "Loaded model at n_tasks_seen={:d}!".format(self._n_meta_tasks_seen)
        )

        # load architecture
        self._create_architecture()
        self._load_weights_from_file(load_path)
        self._set_device(self.device)

        # initialize random number generator
        self._rng = np.random.RandomState()
        self._seed(self._config["seed"])

        # initialize optimizer
        self._create_optimizer()

        self._is_initialized = True

    def _check_data_shapes(self, x, y=None):
        if len(x.shape) < 2 or x.shape[-1] != self.settings["d_x"]:
            raise NotImplementedError("x has wrong shape!")

        if y is not None:
            if len(y.shape) < 2 or y.shape[-1] != self.settings["d_y"]:
                raise NotImplementedError("y has wrong shape!")

    def _prepare_data_for_testing(self, data):
        data = torch.tensor(data, device=self.device).float()
        assert 2 <= data.ndim <= 3
        if data.ndim == 2:
            data = data[None, :, :]  # add task dimension
        data.to(self.device)

        return data

    def _create_ctx_tst_sets(self, x_all, y_all, n_ctx, loss_type):
        n_all = x_all.shape[1]

        # determine context points
        idx_pts = self._rng.permutation(x_all.shape[1])
        x_ctx = x_all[:, idx_pts[:n_ctx], :]
        y_ctx = y_all[:, idx_pts[:n_ctx], :]

        # determine target points
        if not (loss_type == "MCIW" or loss_type == "IWMCIW" or loss_type == "VI"):
            # use all points as test points
            x_tgt = x_all[:, idx_pts, :]
            y_tgt = y_all[:, idx_pts, :]
            latent_obs_all = None  # not necessary
        else:  # loss_type == "VI"
            # sample a test set size between [n_ctx + 1, n_all]
            assert n_ctx < n_all, "Context set must not comprise all data!"
            low = n_ctx + 1
            high = n_all + 1  # exclusive
            n_tst = self._rng.randint(low=low, high=high, size=(1,)).squeeze()
            x_tgt = x_all[:, idx_pts[:n_tst], :]
            y_tgt = y_all[:, idx_pts[:n_tst], :]
            latent_obs_all = self.encoder.encode(x_all, y_all)

        return x_ctx, y_ctx, x_tgt, y_tgt, latent_obs_all

    def _conditional_ll(self, x_tgt, y_tgt, mu_z, cov_z):
        assert x_tgt.ndim == y_tgt.ndim == 3  # (n_tsk, n_tst, d_x/d_y)
        assert x_tgt.nelement() != 0
        assert y_tgt.nelement() != 0

        # obtain predictions
        mu_y, std_y = self._predict(x=x_tgt, mu_z=mu_z, cov_z=cov_z, n_marg=1)
        # mu_y, std_y shape = (n_tsk, n_ls, 1, n_tst, d_y)
        mu_y, std_y = mu_y.squeeze(2), std_y.squeeze(2)
        assert mu_y.ndim == 4 and std_y.ndim == 4

        # add latent state dimension to y-values
        n_ls = mu_y.shape[1]
        y_tgt = y_tgt[:, None, :, :].expand(-1, n_ls, -1, -1)

        # compute mean log-likelihood
        gaussian = torch.distributions.Normal(mu_y, std_y)
        ll = gaussian.log_prob(y_tgt)

        # take sum of lls over output dimension
        ll = torch.sum(ll, axis=-1)

        # take mean over all datapoints
        ll = torch.mean(ll)

        return ll

    def _log_marg_lhd_np_mc(self, x_tgt, y_tgt, mu_z_ctx, cov_z_ctx, n_marg):
        assert x_tgt.ndim == y_tgt.ndim == 3  # (n_tsk, n_tst, d_x/d_y)
        assert x_tgt.nelement() != 0
        assert y_tgt.nelement() != 0

        # obtain predictions
        mu_y, std_y = self._predict(x_tgt, mu_z_ctx, cov_z_ctx, n_marg=n_marg)
        # mu_y, std_y shape = (n_tsk, n_ls, n_marg, n_tst, d_y)

        # add latent state and marginalization dimension to y-values
        n_ls = mu_y.shape[1]
        n_tsk = x_tgt.shape[0]
        n_tgt = x_tgt.shape[1]
        assert n_marg > 0
        y_tgt = y_tgt[:, None, None, :, :].expand(-1, n_ls, n_marg, -1, -1)

        # compute log-likelihood for all datapoints
        gaussian = torch.distributions.Normal(mu_y, std_y)
        ll = gaussian.log_prob(y_tgt)

        # sum log-likelihood over output and datapoint dimension
        ll = torch.sum(ll, dim=(-2, -1))

        # compute MC-average
        ll = torch.logsumexp(ll, dim=2)

        # add -log(n_marg)
        ll = -np.log(n_marg) + ll

        # sum task- and ls-dimensions
        ll = torch.sum(ll, dim=(0, 1))
        assert ll.ndim == 0

        # compute average log-likelihood over all datapoints
        ll = ll / (n_tsk * n_ls * n_tgt)

        return ll
    
    
    def _log_marg_lhd_np_dais(self, x_tgt, y_tgt, mu_z_ctx, cov_z_ctx, n_marg, n_steps=10, step_size=0.08):
        assert x_tgt.ndim == y_tgt.ndim == 3  # (n_tsk, n_tst, d_x/d_y)
        assert x_tgt.nelement() != 0
        assert y_tgt.nelement() != 0

        # obtain predictions
        # mu_y, std_y = self._predict(x_tgt, mu_z_ctx, cov_z_ctx, n_marg=n_marg)
        # mu_y, std_y shape = (n_tsk, n_ls, n_marg, n_tst, d_y)

        # add latent state and marginalization dimension to y-values
        n_ls = mu_z_ctx.shape[1]
        n_tsk = x_tgt.shape[0]
        n_tgt = x_tgt.shape[1]
        assert n_marg > 0
        y_tgt = y_tgt[:, None, None, :, :].expand(-1, n_ls, n_marg, -1, -1)

        cov_z_ctx = torch.sqrt(cov_z_ctx)

        # expand mu_z, std_z w.r.t. n_marg
        mu_z_ctx = mu_z_ctx[:, :, None, :]
        mu_z_ctx = mu_z_ctx.expand(n_tsk, n_ls, n_marg, -1)
        cov_z_ctx = cov_z_ctx[:, :, None, :]
        cov_z_ctx = cov_z_ctx.expand(n_tsk, n_ls, n_marg, -1)

        eps = self._rng.randn(*mu_z_ctx.shape)
        eps = torch.tensor(eps, dtype=torch.float32).to(self.device)
        initial_z = mu_z_ctx + eps * cov_z_ctx
        
        def log_likelihood(z):
            # obtain predictions
            mu_y, std_y = self.decoder.decode(x_tgt, z)
            # mu_y, std_y shape = (n_tsk, n_ls, n_marg, n_tst, d_y)

            # add latent state and marginalization dimension to y-values

            gaussian = torch.distributions.Normal(mu_y, std_y)
            ll = gaussian.log_prob(y_tgt)
            # sum log-likelihood over output and datapoint dimension
            ll = torch.sum(ll, dim=(-2, -1))
            return ll
        
        def log_prior(z):
            dist = torch.distributions.Normal(mu_z_ctx, cov_z_ctx.sqrt())
            lp = dist.log_prob(z)
            lp = torch.sum(lp, dim=(-1))
            return lp
        
        def log_posterior(z):
            return log_prior(z) + log_likelihood(z)
    
        
        # compute log-likelihood for all datapoints
        
        
        ll, _ = differentiable_annealed_importance_sampling(
            initial_z,
            log_posterior,
            log_prior,
            n_steps=n_steps,
            step_size=step_size,
        )

        # sum log-likelihood over output and datapoint dimension
        # ll = torch.sum(ll, dim=(-2, -1))

        # compute MC-average
        ll = torch.logsumexp(ll, dim=2)

        # add -log(n_marg)
        ll = -np.log(n_marg) + ll

        # sum task- and ls-dimensions
        ll = torch.sum(ll, dim=(0, 1))
        assert ll.ndim == 0

        # compute average log-likelihood over all datapoints
        ll = ll / (n_tsk * n_ls * n_tgt)

        return ll

    def _elbo_np(
        self,
        x_tgt,
        y_tgt,
        mu_z_ctx,
        cov_z_ctx,
        agg_state_ctx,
        latent_obs_all,
    ):
        # computes the vi-inspired loss
        #  latent_obs_all are the latent observations w.r.t. context + target
        #  mu_z, cov_z, agg_state are the latent/agg states w.r.t. the *context set*

        # obtain shapes
        n_ls = mu_z_ctx.shape[1]
        n_tgt = x_tgt.shape[1]

        # compute posterior latent states w.r.t. the test sets
        mu_z_all = torch.zeros(mu_z_ctx.shape, device=self.device)
        cov_z_all = torch.zeros(cov_z_ctx.shape, device=self.device)
        for j in range(n_ls):
            cur_agg_state_old = tuple(entry[:, j, :] for entry in agg_state_ctx)
            cur_agg_state_new = self.aggregator.step(
                agg_state_old=cur_agg_state_old,
                latent_obs=latent_obs_all,
            )
            (cur_mu_z_all, cur_cov_z_all) = self.aggregator.agg2latent(
                cur_agg_state_new
            )
            mu_z_all[:, j, :] = cur_mu_z_all
            cov_z_all[:, j, :] = cur_cov_z_all

        # compute log likelihood using posterior latent states
        ll = self._conditional_ll(
            x_tgt=x_tgt, y_tgt=y_tgt, mu_z=mu_z_all, cov_z=cov_z_all
        )

        # compute kls between posteriors and corresponding priors
        std_z_ctx = torch.sqrt(cov_z_ctx)
        std_z_all = torch.sqrt(cov_z_all)
        gaussian_z_ctx = torch.distributions.Normal(loc=mu_z_ctx, scale=std_z_ctx)
        gaussian_z_tgt = torch.distributions.Normal(loc=mu_z_all, scale=std_z_all)
        kl = torch.distributions.kl.kl_divergence(gaussian_z_tgt, gaussian_z_ctx)
        # sum over latent dimension (diagonal Gaussians)
        kl = torch.sum(kl, axis=-1)
        # take mean over task and ls dimensions
        kl = torch.mean(kl, dim=[0, 1]).squeeze()

        # compute loss
        elbo = ll - kl / n_tgt

        return elbo

    def _sample_z(self, mu_z, cov_z, n_samples):
        # read out sizes
        n_tsk = mu_z.shape[0]
        n_ls = mu_z.shape[1]

        # expand mu_z, std_z w.r.t. n_marg
        mu_z = mu_z[:, :, None, :]
        mu_z = mu_z.expand(n_tsk, n_ls, n_samples, self.settings["d_z"])
        std_z = torch.sqrt(cov_z)
        std_z = std_z[:, :, None, :]
        std_z = std_z.expand(n_tsk, n_ls, n_samples, self.settings["d_z"])

        # sample z
        eps = self._rng.randn(*mu_z.shape)
        eps = torch.tensor(eps, dtype=torch.float32).to(self.device)
        z = mu_z + eps * std_z

        # check output
        assert z.shape == (n_tsk, n_ls, n_samples, self.settings["d_z"])
        return z

    def _predict(self, x, mu_z, cov_z, n_marg, return_latent_samples=False):
        assert x.ndim == 3  # (n_tsk, n_tst, d_x)
        assert mu_z.ndim == 3  # (n_tsk, n_ls, d_z)
        if cov_z is not None:
            assert mu_z.shape == cov_z.shape

        # collect shapes
        n_tsk = mu_z.shape[0]
        n_ls = mu_z.shape[1]
        d_z = mu_z.shape[2]

        z = self._sample_z(mu_z=mu_z, cov_z=cov_z, n_samples=n_marg)
        mu_y, std_y = self.decoder.decode(
            x=x, z=z, mu_z=mu_z, cov_z=cov_z
        )  # (n_tsk, n_ls, n_marg, n_tst, d_y)

        assert mu_y.ndim == 5 and std_y.ndim == 5

        if not return_latent_samples:
            # mu_y, std_y = (n_tsk, n_ls, n_marg, n_tst, d_y)
            return mu_y, std_y
        else:
            # mu_y, std_y = (n_tsk, n_ls, n_marg, n_tst, d_y)
            # z = (n_tsk, n_ls, n_marg, d_z)
            return mu_y, std_y, z

    def meta_train(
        self,
        benchmark_meta: MetaLearningBenchmark,
        n_tasks_train: int,
        validation_interval: Optional[int] = None,
        benchmark_val: Optional[MetaLearningBenchmark] = None,
        callback=None,
    ) -> float:
        def validate_now() -> bool:
            if validation_interval is None:
                return False

            # at beginning
            if self._n_meta_tasks_seen == 0:
                return True

            # if we jumped into the next validation_interval with the last batch
            if (
                (self._n_meta_tasks_seen - self._config["batch_size"])
                // validation_interval
                < self._n_meta_tasks_seen // validation_interval
            ):
                return True

            return False

        def validation_loss() -> float:
            if benchmark_val is None:
                return None

            with torch.no_grad():
                x_val, y_val = next(iter(dataloader_val))

                # compute validation loss
                loss = self._compute_loss(x=x_val, y=y_val, mode="val")
                loss = loss.cpu().numpy().item()

            return loss

        def optimizer_step() -> float:
            # perform optimizer step on batch of metadata
            x_meta, y_meta = next(iter(dataloader_meta))

            # drop last elements if batch is too large
            n_tasks_in_batch = x_meta.shape[0]
            if self._n_meta_tasks_seen + n_tasks_in_batch > n_tasks_train:
                n_tasks_remaining = n_tasks_train - self._n_meta_tasks_seen
                x_meta, y_meta = x_meta[:n_tasks_remaining], y_meta[:n_tasks_remaining]

            # reset gradient
            self._optimizer.zero_grad()

            # compute loss
            loss_meta = self._compute_loss(x=x_meta, y=y_meta, mode="meta")

            # perform gradient step on minibatch
            loss_meta.backward()
            self._optimizer.step()

            # update self.n_meta_tasks_seen
            n_tasks_in_batch = x_meta.shape[0]
            self._n_meta_tasks_seen += n_tasks_in_batch
            pbar.update(n_tasks_in_batch)

            return loss_meta.detach().cpu().numpy()

        # log
        self._logger.info(
            "Training model on {:d} tasks ({:d} remaining)...".format(
                n_tasks_train, max(n_tasks_train - self._n_meta_tasks_seen, 0)
            )
        )

        # set device for training
        self._set_device(self.device)

        # create dataloader
        dataloader_meta = DataLoader(
            dataset=MetaLearningDataset(benchmark_meta),
            batch_size=self._config["batch_size"],
            collate_fn=lambda task_list: self._collate_batch(task_list),
        )
        if benchmark_val is not None:
            dataloader_val = DataLoader(
                dataset=MetaLearningDataset(benchmark_val),
                batch_size=self._config["batch_size"],
                collate_fn=lambda task_list: self._collate_batch(task_list),
            )
        else:
            dataloader_val = None

        # training loop
        loss_meta, loss_val = None, None
        with tqdm(
            total=n_tasks_train, leave=False, desc="meta-fit", mininterval=1
        ) as pbar:
            pbar.update(self._n_meta_tasks_seen)
            while self._n_meta_tasks_seen < n_tasks_train:
                if validate_now():
                    loss_val = validation_loss()
                loss_meta = optimizer_step()
                if callback is not None:
                    callback_data = {"loss_meta": loss_meta, 'loss_val': loss_val} if loss_meta is not None else None
                    callback(
                        n_meta_tasks_seen=self._n_meta_tasks_seen,
                        np_model=self,
                        metrics=callback_data,
                    )
                pbar.set_postfix(
                    {"loss_meta": loss_meta, "loss_val": loss_val}, refresh=False
                )

            # compute loss_val once again at the end
            loss_val = validation_loss()

        self._set_device(self.device)
        self._logger.info("Training finished successfully!")

        return loss_val

    @torch.no_grad()
    def predict(
        self, x: np.ndarray, n_samples: int = 1
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        # check input data
        self._check_data_shapes(x=x)

        # prepare x
        has_tsk_dim = x.ndim == 3
        x = self._prepare_data_for_testing(x)

        # read out last latent state
        ls = self.aggregator.last_latent_state
        mu_z = ls[0][:, None, :]
        cov_z = ls[1][:, None, :] if ls[1] is not None else None

        # obtain predictions
        mu_y, std_y = self._predict(x, mu_z, cov_z, n_marg=n_samples)

        # check that target and context data are consistent
        if has_tsk_dim and mu_y.shape[0] != x.shape[0]:
            raise NotImplementedError(
                "Target and context data have different numbers of tasks!"
            )

        # squeeze latent state dimension (this is always singleton here)
        mu_y, std_y = mu_y.squeeze(1), std_y.squeeze(1)  # squeeze n_ls dimension

        # # squeeze marginalization dimension (this is always singleton here)
        # if n_samples == 1:
        #     mu_y, std_y = mu_y.squeeze(1), std_y.squeeze(1)  # squeeze n_marg dimension

        # squeeze task dimension (if singleton)
        if not has_tsk_dim:
            mu_y = mu_y.squeeze(0)
            std_y = std_y.squeeze(0)
        mu_y, std_y = mu_y.cpu().numpy(), std_y.cpu().numpy()

        return mu_y, std_y**2  # ([n_tsk,], [n_samples], n_pts, d_y)

    @torch.no_grad()
    def adapt(self, x: np.ndarray, y: np.ndarray) -> None:
        self._check_data_shapes(x=x, y=y)

        # prepare x and y
        x = self._prepare_data_for_testing(x)
        y = self._prepare_data_for_testing(y)

        # accumulate data in aggregator
        self.aggregator.reset(n_tsk=x.shape[0])
        if x.shape[1] > 0:
            latent_obs = self.encoder.encode(x=x, y=y)
            self.aggregator.update(latent_obs)

    @torch.no_grad()
    def sample_z(self, n_samples):
        # check input
        assert self.settings["loss_type"] != "PB"

        # read out last latent state
        ls = self.aggregator.last_latent_state
        mu_z = ls[0][:, None, :]
        cov_z = ls[1][:, None, :] if ls[1] is not None else None

        # read out sizes
        n_tsk = mu_z.shape[0]
        n_ls = mu_z.shape[1]
        assert n_ls == 1

        # sample z
        z = self._sample_z(mu_z=mu_z, cov_z=cov_z, n_samples=n_samples)

        # squeeze n_ls dimension
        z = z.squeeze(1)

        # check output
        z = z.numpy()
        assert z.shape == (n_tsk, n_samples, self.settings["d_z"])
        return z

    @torch.no_grad()
    def predict_at_z(self, x, z):
        # check input
        assert self.settings["loss_type"] != "PB"
        n_tsk = x.shape[0]
        n_pts = x.shape[1]
        n_samples = z.shape[1]
        assert x.shape == (n_tsk, n_pts, self.settings["d_x"])
        assert z.shape == (n_tsk, n_samples, self.settings["d_z"])

        # check input data
        self._check_data_shapes(x=x)

        # prepare x and z
        x = self._prepare_data_for_testing(x)
        z = torch.Tensor(z, device=self.device)
        z = z[:, None, :, :]  # add dummy ls dimension

        # predict
        mu_y, std_y = self.decoder.decode(x=x, z=z)
        assert mu_y.shape == (n_tsk, 1, n_samples, n_pts, self.settings["d_y"])
        assert std_y.shape == (n_tsk, 1, n_samples, n_pts, self.settings["d_y"])

        # squeeze latent state dimension (this is always singleton here)
        mu_y, std_y = mu_y.squeeze(1), std_y.squeeze(1)  # squeeze n_ls dimension

        # check output
        mu_y, std_y = mu_y.numpy(), std_y.numpy()
        assert mu_y.shape == (n_tsk, n_samples, n_pts, self.settings["d_y"])
        assert std_y.shape == (n_tsk, n_samples, n_pts, self.settings["d_y"])
        return mu_y, std_y**2
