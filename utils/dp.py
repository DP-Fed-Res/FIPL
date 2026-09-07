import math
import torch
from opacus.accountants.utils import get_noise_multiplier as _opacus_get_noise_multiplier


def compute_noise_multiplier(eps, delta, sample_rate, steps):
    if eps <= 0:
        return 0.0
    return _opacus_get_noise_multiplier(
        target_epsilon=eps,
        target_delta=delta,
        sample_rate=sample_rate,
        steps=steps,
        accountant='prv',
    )


def get_gaussian_noise_std(eps, delta):
    if eps <= 0:
        return 0.0
    return math.sqrt(2.0 * math.log(1.25 / delta)) / eps


def add_noise_to_prototype(proto, noise_std, sensitivity):
    if noise_std <= 0:
        return proto
    noise = torch.randn_like(proto) * noise_std * sensitivity
    return proto + noise


def compute_multi_round_gaussian_std(eps, delta, num_rounds):
    if eps <= 0 or num_rounds <= 0:
        return 0.0
    eps_per_round = eps / num_rounds
    return get_gaussian_noise_std(eps_per_round, delta)


def make_private(net, data_loader, noise_multiplier, max_grad_norm, lr=0.01, device='cuda', privacy_engine=None):
    import torch.optim as optim
    from opacus import PrivacyEngine

    net = net.to(device)

    trainable_params = [p for n, p in net.named_parameters()
                        if not n.startswith('encoder.')
                        and not n.startswith('_module.encoder.')]
    optimizer = optim.SGD(trainable_params, lr=lr, momentum=0.9, weight_decay=1e-5)

    if privacy_engine is None:
        privacy_engine = PrivacyEngine(accountant='prv')
    net, optimizer, data_loader = privacy_engine.make_private(
        module=net,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        poisson_sampling=False,  # client data is already shuffled in FL
    )
    net.to(device)
    return net, optimizer, data_loader, privacy_engine


def find_unified_sigma(target_eps, delta, history_template, epsilon_tolerance=0.01):
    if target_eps <= 0:
        return 0.0

    from opacus.accountants.prv import PRVAccountant

    accountant = PRVAccountant()

    sigma_high = 0.5
    while True:
        sigma_high = 2.0 * sigma_high
        accountant.history = [(sigma_high, q, s) for q, s in history_template]
        eps_high = accountant.get_epsilon(delta)
        if eps_high <= target_eps:
            break
        if sigma_high > 1000.0:
            raise ValueError(
                f'Could not find a noise multiplier satisfying eps={target_eps} (sigma exceeded 1000). '
                f'Check the history_template parameter or increase target eps.'
            )

    sigma_low = 0.0
    while target_eps - eps_high > epsilon_tolerance:
        sigma = (sigma_low + sigma_high) / 2.0
        accountant.history = [(sigma, q, s) for q, s in history_template]
        eps = accountant.get_epsilon(delta)

        if eps < target_eps:
            sigma_high = sigma
            eps_high = eps
        else:
            sigma_low = sigma

    return sigma_high


def get_privacy_spent(privacy_engine_or_optimizer, delta):
    if hasattr(privacy_engine_or_optimizer, 'accountant') and privacy_engine_or_optimizer.accountant is not None:
        try:
            result = privacy_engine_or_optimizer.accountant.get_epsilon(delta)
            return result
        except Exception:
            return 0.0
    return 0.0
