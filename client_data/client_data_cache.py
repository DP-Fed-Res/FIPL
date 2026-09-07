import os
import json
import hashlib
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple

_CACHE_ROOT = os.path.dirname(os.path.abspath(__file__))


def _build_cache_key(dataset_name: str,
                     selected_domains: List[str],
                     domain_dict: Dict[str, int],
                     percent_dict: Dict[str, float],
                     seed: int) -> str:
    payload = {
        'dataset': dataset_name,
        'selected_domains': selected_domains,
        'domain_dict': domain_dict,
        'percent_dict': percent_dict,
        'seed': seed,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _cache_dir(dataset_name: str) -> str:
    d = os.path.join(_CACHE_ROOT, dataset_name)
    os.makedirs(d, exist_ok=True)
    return d


def _config_path(dataset_name: str, cache_key: str) -> str:
    return os.path.join(_cache_dir(dataset_name), f'{cache_key}.json')


def _indices_path(dataset_name: str, cache_key: str) -> str:
    return os.path.join(_cache_dir(dataset_name), f'{cache_key}.pkl')


def load_cached_indices(dataset_name: str,
                        selected_domains: List[str],
                        domain_dict: Dict[str, int],
                        percent_dict: Dict[str, float],
                        seed: int = 0) -> Optional[List[np.ndarray]]:
    cache_key = _build_cache_key(dataset_name, selected_domains, domain_dict, percent_dict, seed)
    ipath = _indices_path(dataset_name, cache_key)
    if os.path.exists(ipath):
        with open(ipath, 'rb') as f:
            return pickle.load(f)
    return None


def save_cached_indices(dataset_name: str,
                        selected_domains: List[str],
                        domain_dict: Dict[str, int],
                        percent_dict: Dict[str, float],
                        seed: int,
                        indices: List[np.ndarray],
                        extra_meta: Optional[Dict] = None) -> str:
    cache_key = _build_cache_key(dataset_name, selected_domains, domain_dict, percent_dict, seed)

    with open(_indices_path(dataset_name, cache_key), 'wb') as f:
        pickle.dump(indices, f)

    meta = {
        'cache_key': cache_key,
        'dataset': dataset_name,
        'selected_domains': selected_domains,
        'domain_dict': domain_dict,
        'percent_dict': percent_dict,
        'seed': seed,
        'num_clients': len(indices),
        'samples_per_client': [int(len(idx)) for idx in indices],
        'total_samples': int(sum(len(idx) for idx in indices)),
    }
    if extra_meta:
        meta['extra'] = extra_meta

    with open(_config_path(dataset_name, cache_key), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return cache_key


def list_cached_configs(dataset_name: str = None) -> List[Dict]:
    configs = []
    datasets = [dataset_name] if dataset_name else os.listdir(_CACHE_ROOT)
    for ds in datasets:
        ds_dir = os.path.join(_CACHE_ROOT, ds)
        if not os.path.isdir(ds_dir):
            continue
        for fname in os.listdir(ds_dir):
            if fname.endswith('.json'):
                with open(os.path.join(ds_dir, fname), 'r', encoding='utf-8') as f:
                    configs.append(json.load(f))
    return configs
