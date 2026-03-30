import json
import os

import numpy as np


def resolve_phase_d_interface_dir(root_path, interface_dir):
    if not interface_dir:
        return ''
    if os.path.isabs(interface_dir):
        return interface_dir
    return os.path.join(root_path, interface_dir)


def _load_json(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def load_phase_d_static_bundle(root_path, interface_dir, expected_nodes=None):
    resolved_dir = resolve_phase_d_interface_dir(root_path, interface_dir)
    if not resolved_dir:
        return None

    a_base_path = os.path.join(resolved_dir, 'phaseD_a_base_agg.npy')
    support_path = os.path.join(resolved_dir, 'phaseD_support.npy')
    manifest_path = os.path.join(resolved_dir, 'phaseD_interface_manifest.json')

    a_base = np.load(a_base_path).astype(np.float32)
    support = np.load(support_path).astype(np.float32)
    manifest = _load_json(manifest_path)

    if a_base.ndim != 2 or a_base.shape[0] != a_base.shape[1]:
        raise ValueError(f'Phase D static graph must be square 2D, got {a_base.shape}')
    if support.shape != a_base.shape:
        raise ValueError(f'Phase D support shape {support.shape} does not match static graph {a_base.shape}')
    if expected_nodes is not None and a_base.shape[0] != int(expected_nodes):
        raise ValueError(f'Phase D static graph node count {a_base.shape[0]} does not match expected {expected_nodes}')

    return {
        'interface_dir': resolved_dir,
        'a_base': a_base,
        'support': support,
        'manifest': manifest,
        'manifest_path': manifest_path,
    }


def load_phase_d_train_bundle(root_path, interface_dir, window_starts, expected_nodes=None,
                              shuffle_lambda=False, shuffle_seed=0):
    resolved_dir = resolve_phase_d_interface_dir(root_path, interface_dir)
    if not resolved_dir:
        return None

    lambda_path = os.path.join(resolved_dir, 'phaseD_lambda_train.npy')
    delta_path = os.path.join(resolved_dir, 'phaseD_deltaA_train.npy')
    index_path = os.path.join(resolved_dir, 'phaseD_window_index_train.json')
    manifest_path = os.path.join(resolved_dir, 'phaseD_interface_manifest.json')

    lambda_train = np.load(lambda_path).astype(np.float32).reshape(-1)
    delta_train = np.load(delta_path).astype(np.float32)
    window_index = _load_json(index_path)
    manifest = _load_json(manifest_path)

    expected_count = len(window_starts)
    if len(lambda_train) != expected_count:
        raise ValueError(f'Phase D lambda train length {len(lambda_train)} does not match dataset windows {expected_count}')
    if delta_train.shape[0] != expected_count:
        raise ValueError(f'Phase D delta train length {delta_train.shape[0]} does not match dataset windows {expected_count}')
    if len(window_index) != expected_count:
        raise ValueError(f'Phase D window index length {len(window_index)} does not match dataset windows {expected_count}')
    if delta_train.ndim != 3 or delta_train.shape[1] != delta_train.shape[2]:
        raise ValueError(f'Phase D delta train must be (B, N, N), got {delta_train.shape}')
    if expected_nodes is not None and delta_train.shape[1] != int(expected_nodes):
        raise ValueError(f'Phase D delta node count {delta_train.shape[1]} does not match expected {expected_nodes}')

    starts_expected = np.asarray(window_starts, dtype=np.int64)
    starts_found = np.asarray([int(item['window_start']) for item in window_index], dtype=np.int64)
    sample_ids = np.asarray([int(item['sample_id']) for item in window_index], dtype=np.int64)
    if not np.array_equal(sample_ids, np.arange(expected_count, dtype=np.int64)):
        raise ValueError('Phase D window index sample_id ordering does not match dataset ordering')
    if not np.array_equal(starts_found, starts_expected):
        raise ValueError('Phase D window starts do not match dataset window_starts')

    if shuffle_lambda:
        rng = np.random.RandomState(int(shuffle_seed))
        lambda_train = lambda_train[rng.permutation(expected_count)]

    return {
        'interface_dir': resolved_dir,
        'lambda_train': lambda_train,
        'delta_train': delta_train,
        'window_index': window_index,
        'manifest': manifest,
        'manifest_path': manifest_path,
    }
