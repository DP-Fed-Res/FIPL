best_args = {
    'fl_digits': {
        'fipl_da': {
            'local_lr': 0.01,
            'local_batch_size': 64,
            'mu': 100,
            'mu_decay': 0.9,
            'proto_margin': -0.1,
        },
    },

    'fl_pacs': {
        'fipl_da': {
            'local_lr': 0.01,
            'local_batch_size': 32,
            'mu': 100,
            'mu_decay': 0.9,
            'proto_margin': -0.1,
        },
    },
}

dataset_config = {
    'fl_digits': {
        'domain_dict': {
            10: {'mnist': 2, 'usps': 2, 'svhn': 2, 'syn': 2, 'mnist-m': 2},
            20: {'mnist': 4, 'usps': 2, 'svhn': 5, 'syn': 6, 'mnist-m': 3},
            30: {'mnist': 6, 'usps': 2, 'svhn': 8, 'syn': 10, 'mnist-m': 4},
        },
        'percent_dict': {'mnist': 1/32, 'usps': 1/4, 'svhn': 1/44, 'syn': 1/48, 'mnist-m': 1/10},
    },

    'fl_pacs': {
        'domain_dict': {
            10: {'photo': 2, 'art_painting': 2, 'cartoon': 2, 'sketch': 4},
        },
        'percent_dict': {'photo': 0.5, 'art_painting': 0.5, 'cartoon': 0.5, 'sketch': 0.25},
    },
}


def _scale_domain_dict(base_dd, target_n):
    domains = list(base_dd.keys())
    base_total = sum(base_dd.values())
    scaled = {}
    remaining = target_n
    for i, d in enumerate(domains):
        if i == len(domains) - 1:
            scaled[d] = remaining
        else:
            v = max(1, int(round(base_dd[d] / base_total * target_n)))
            v = min(v, remaining - (len(domains) - i - 1))
            scaled[d] = v
            remaining -= v
    return scaled


def get_domain_dict(dataset_name, num_clients=0):
    config = dataset_config.get(dataset_name, {})
    dd_all = config.get('domain_dict', {})
    if not dd_all:
        return {}

    if num_clients <= 0:
        num_clients = max(dd_all.keys())

    if num_clients in dd_all:
        return dict(dd_all[num_clients])

    base_n = max(dd_all.keys())
    base_dd = dd_all[base_n]
    scaled = _scale_domain_dict(base_dd, num_clients)
    print(f'[num_clients] auto-scale: {dataset_name} {base_n}->{num_clients} -> {scaled}')
    return scaled
