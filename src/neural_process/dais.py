import math
import torch
import numpy as np
from torch import autograd, optim
from torch.distributions.multivariate_normal import MultivariateNormal
import scipy.optimize
from tqdm import tqdm


def get_schedule(num, rad=4):
    if num == 1:
        return np.array([0.0, 1.0])
    t = np.linspace(-rad, rad, num)
    s = 1.0 / (1.0 + np.exp(-t))
    return (s - np.min(s)) / (np.max(s) - np.min(s))


def differentiable_annealed_importance_sampling(s:torch.Tensor, log_likelihood, log_q, n_steps,
        step_size, partial=False, gamma=0.9, mass_matrix=None, clip_grad=None,
        lrates=None, betas=None, block_grad=False, is_train=True, rng:np.random.RandomState=None):
    """
    s: partcle state: n_particles x d
    """
    n_particles = s.shape[:-1]
    dim = s.shape[-1]

    if n_steps == 0:
        return - log_q(s) + log_likelihood(s), s
    
    # if lrates is None:
    #     lrates = step_size * torch.ones(n_steps+1, device=s.device)
    if (type(step_size) is torch.Tensor or type(step_size) is np.ndarray) and len(step_size.shape) > 0:
        assert step_size.shape == s.shape[:-1]
        step_size = step_size[..., None]
    
    if betas is None:
        # betas = torch.linspace(1.0/n_steps, 1.0, n_steps, device=s.device)
        betas = get_schedule(n_steps+1)

    if mass_matrix is None:
        mass_matrix = torch.eye(dim, device=s.device)
    pi_mean = torch.zeros(dim, device=s.device)
    pi = MultivariateNormal(pi_mean, mass_matrix)
    inverse_mass_matrix = torch.inverse(mass_matrix)

    # s.requires_grad = True

    def log_annealed_prob(beta, s: torch.Tensor):
        return (1 - beta) * log_q(s) + beta * log_likelihood(s)

    def grad_log_annealed_prob(beta, s):
        ''' it's important to set create_graph=True '''
        with torch.enable_grad():
            s.requires_grad_()
            grad = autograd.grad(log_annealed_prob(beta, s).sum(), s, create_graph=is_train)[0]
            if clip_grad is not None:
                max_ = torch.prod(torch.tensor(s.shape)) * clip_grad # last dimension of mu_z
                grad = torch.clamp(grad, -max_, max_)
        return grad

    # sample initial momentum
    def pi_sample(n_particles):
        if rng is None:
            eps = pi.sample(n_particles)
        else:
            eps = torch.tensor(rng.multivariate_normal(pi_mean, mass_matrix, n_particles), 
                               dtype=s.dtype, device=s.device)
        return eps
    
    v = pi_sample(n_particles)

    with torch.set_grad_enabled(is_train):

        elbo = - log_q(s)
        for k in range(1, n_steps+1):
            assert not torch.any(s.isnan()), "Current state has nan values"
            elbo = elbo - pi.log_prob(v)

            # leapfrog
            s = s + step_size / 2 * v @ inverse_mass_matrix
            v = v + step_size * grad_log_annealed_prob(betas[k], s)
            s = s + step_size / 2 * v @ inverse_mass_matrix

            elbo = elbo + pi.log_prob(v)

            if partial:
                # partial_refreshment
                v = gamma * v + math.sqrt(1 - math.pow(gamma, 2)) * pi_sample(n_particles)
            else:
                v = pi_sample(n_particles)

        elbo = elbo + log_likelihood(s)

    return elbo, s


def leapfrogs_and_bounds_optlr(s, log_likelihood, log_q, n_steps,
        step_size, partial=False, gamma=0.9, mass_matrix=None):
    
    init_log_lrates = math.log(step_size) * np.ones(n_steps)
    bounds = scipy.optimize.Bounds(-np.infty, -1.0)
    # log_lrates = torch.tensor(llrates_np, dtype=s.dtype, device=s.device, requires_grad=True)

    def func_fn(log_lrates):
        log_lrates = torch.tensor(log_lrates, dtype=s.dtype, device=s.device, requires_grad=True)
        elbo, _ = differentiable_annealed_importance_sampling(s, log_likelihood, log_q, n_steps, 
            step_size, partial, gamma, mass_matrix, lrates=torch.exp(log_lrates))
        return -elbo.sum().data.cpu().numpy()

    def grad_fn(log_lrates):
        log_lrates = torch.tensor(log_lrates, dtype=s.dtype, device=s.device, requires_grad=True)
        elbo, _ = differentiable_annealed_importance_sampling(s, log_likelihood, log_q, n_steps, 
            step_size, partial, gamma, mass_matrix, lrates=torch.exp(log_lrates))
        loss = -elbo.sum()
        return autograd.grad(loss, log_lrates)[0].data.cpu().numpy().astype(np.float64)

    res = scipy.optimize.minimize(func_fn, init_log_lrates, jac=grad_fn, bounds=bounds)
    log_lrates = torch.tensor(res.x, dtype=s.dtype, device=s.device)

    return differentiable_annealed_importance_sampling(s, log_likelihood, log_q, n_steps, 
        step_size, partial, gamma, mass_matrix, lrates=torch.exp(log_lrates))





