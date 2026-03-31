import json
import os

import numpy as np


LEGACY_STATIC_CANDIDATES = {
    'a_base': ['a_base_agg.npy', 'phaseD_a_base_agg.npy'],
    'support': ['support.npy', 'phaseD_support.npy'],
    'manifest': ['interface_manifest.json', 'phaseD_interface_manifest.json'],
}

LEGACY_TRAIN_CANDIDATES = {
    'lambda_train': ['lambda_train.npy', 'phaseD_lambda_train.npy'],
    'delta_train': ['deltaA_train.npy', 'phaseD_deltaA_train.npy'],
    'window_index': ['window_index_train.json', 'phaseD_window_index_train.json'],
    'manifest': ['interface_manifest.json', 'phaseD_interface_manifest.json'],
}


def resolve_graph_interface_dir(root_path, interface_dir):
    if not interface_dir:
        return ''
    if os.path.isabs(interface_dir):
        return interface_dir
    return os.path.join(root_path, interface_dir)


def _load_json(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def _resolve_existing_file(resolved_dir, candidates, label):
    for name in candidates:
        path = os.path.join(resolved_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f'Could not find {label} in {resolved_dir}. Tried: {candidates}'
    )


def load_graph_static_bundle(root_path, interface_dir, expected_nodes=None):
    resolved_dir = resolve_graph_interface_dir(root_path, interface_dir)
    if not resolved_dir:
        return None

    a_base_path = _resolve_existing_file(resolved_dir, LEGACY_STATIC_CANDIDATES['a_base'], 'static graph file')
    support_path = _resolve_existing_file(resolved_dir, LEGACY_STATIC_CANDIDATES['support'], 'support file')
    manifest_path = _resolve_existing_file(resolved_dir, LEGACY_STATIC_CANDIDATES['manifest'], 'manifest file')

    a_base = np.load(a_base_path).astype(np.float32)
    support = np.load(support_path).astype(np.float32)
    manifest = _load_json(manifest_path)

    if a_base.ndim != 2 or a_base.shape[0] != a_base.shape[1]:
        raise ValueError(f'Graph static bundle must be square 2D, got {a_base.shape}')
    if support.shape != a_base.shape:
        raise ValueError(f'Graph support shape {support.shape} does not match static graph {a_base.shape}')
    if expected_nodes is not None and a_base.shape[0] != int(expected_nodes):
        raise ValueError(f'Graph static node count {a_base.shape[0]} does not match expected {expected_nodes}')

    return {
        'interface_dir': resolved_dir,
        'a_base': a_base,
        'support': support,
        'manifest': manifest,
        'manifest_path': manifest_path,
        'resolved_paths': {
            'a_base': a_base_path,
            'support': support_path,
            'manifest': manifest_path,
        },
    }


def load_graph_train_bundle(root_path, interface_dir, window_starts, expected_nodes=None,
                            shuffle_lambda=False, shuffle_seed=0):
    resolved_dir = resolve_graph_interface_dir(root_path, interface_dir)
    if not resolved_dir:
        return None

    lambda_path = _resolve_existing_file(resolved_dir, LEGACY_TRAIN_CANDIDATES['lambda_train'], 'lambda train file')
    delta_path = _resolve_existing_file(resolved_dir, LEGACY_TRAIN_CANDIDATES['delta_train'], 'delta train file')
    index_path = _resolve_existing_file(resolved_dir, LEGACY_TRAIN_CANDIDATES['window_index'], 'window index file')
    manifest_path = _resolve_existing_file(resolved_dir, LEGACY_TRAIN_CANDIDATES['manifest'], 'manifest file')

    lambda_train = np.load(lambda_path).astype(np.float32).reshape(-1)
    delta_train = np.load(delta_path).astype(np.float32)
    window_index = _load_json(index_path)
    manifest = _load_json(manifest_path)

    expected_count = len(window_starts)
    if len(lambda_train) != expected_count:
        raise ValueError(f'Graph lambda train length {len(lambda_train)} does not match dataset windows {expected_count}')
    if delta_train.shape[0] != expected_count:
        raise ValueError(f'Graph delta train length {delta_train.shape[0]} does not match dataset windows {expected_count}')
    if len(window_index) != expected_count:
        raise ValueError(f'Graph window index length {len(window_index)} does not match dataset windows {expected_count}')
    if delta_train.ndim != 3 or delta_train.shape[1] != delta_train.shape[2]:
        raise ValueError(f'Graph delta train must be (B, N, N), got {delta_train.shape}')
    if expected_nodes is not None and delta_train.shape[1] != int(expected_nodes):
        raise ValueError(f'Graph delta node count {delta_train.shape[1]} does not match expected {expected_nodes}')

    starts_expected = np.asarray(window_starts, dtype=np.int64)
    starts_found = np.asarray([int(item['window_start']) for item in window_index], dtype=np.int64)
    sample_ids = np.asarray([int(item['sample_id']) for item in window_index], dtype=np.int64)
    if not np.array_equal(sample_ids, np.arange(expected_count, dtype=np.int64)):
        raise ValueError('Graph window index sample_id ordering does not match dataset ordering')
    if not np.array_equal(starts_found, starts_expected):
        raise ValueError('Graph window starts do not match dataset window_starts')

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
        'resolved_paths': {
            'lambda_train': lambda_path,
            'delta_train': delta_path,
            'window_index': index_path,
            'manifest': manifest_path,
        },
    }
