"""Dataset loading utilities for the federated data-report pipeline.

Functions here read a dataset directory on disk and return the
``data_splits`` list expected by the FLAME ``StarModelTester``.  Each
split represents one federated node and contains a mapping from filename
to raw file bytes so that the analyzer can handle multiple file types
without the loader needing to understand them.
"""
from pathlib import Path


def load_dataset(dataset_path: Path):
    """Load a single dataset directory and return per-node byte splits.

    Each immediate subdirectory of ``dataset_path`` is treated as one
    federated node.  All files inside that subdirectory are read as raw
    bytes and collected into a ``{filename: bytes}`` mapping, which is
    wrapped in a one-element list to match the ``StarModelTester``
    ``data_splits`` format.

    Args:

        dataset_path (Path): Path to the dataset directory.  Its immediate
            subdirectories must each represent one node.

    Returns:

        list: A list of ``[{filename: bytes}]`` elements, one per node,
            sorted by node directory name.
    """
    data_splits = []

    for node_dir in sorted(dataset_path.iterdir()):
        # if the directory doesn't have node data then continue
        if not node_dir.is_dir():
            continue
        node_data = {}
        for file in node_dir.iterdir():
            if file.is_file():
                # read the file as raw bytes
                # file type check comes later
                node_data[file.name] = file.read_bytes()

        data_splits.append([node_data])
    return data_splits

def load_all_datasets(data_path: Path):
    """Load every dataset directory under ``data_path`` and return a mapping.

    Args:

        data_path (Path): Root directory whose immediate subdirectories are
            individual dataset directories (each consumed by
            ``load_dataset``).

    Returns:

        dict: Mapping of dataset name (directory name) to the ``data_splits``
            list returned by ``load_dataset``.
    """
    # make a dict for the datasets like {dataset_name: data_splits}
    datasets = {}
    for dataset_path in data_path.iterdir():
        if dataset_path.is_dir():
            datasets[dataset_path.name] = load_dataset(dataset_path)
    return datasets
