from utils.graph_interface import (
    load_graph_static_bundle,
    load_graph_train_bundle,
    resolve_graph_interface_dir,
)


def resolve_phase_d_interface_dir(root_path, interface_dir):
    return resolve_graph_interface_dir(root_path, interface_dir)


def load_phase_d_static_bundle(root_path, interface_dir, expected_nodes=None):
    return load_graph_static_bundle(root_path, interface_dir, expected_nodes=expected_nodes)


def load_phase_d_train_bundle(root_path, interface_dir, window_starts, expected_nodes=None,
                              shuffle_lambda=False, shuffle_seed=0):
    return load_graph_train_bundle(
        root_path,
        interface_dir,
        window_starts,
        expected_nodes=expected_nodes,
        shuffle_lambda=shuffle_lambda,
        shuffle_seed=shuffle_seed,
    )
