"""Utils for manipulating sc data."""

import os
import warnings

from scipy.sparse import issparse, csr_matrix

_warn_skips = (os.path.dirname(__file__),)


def restore_X(adata):
    """Restore adata.X from a copy of adata.raw.X in-place.

    If adata.raw.var_names != adata.var_names, a warning is raised and the copy
    is re-aligned to adata.var_names.

    Args:
        adata (AnnData): Annotated data matrix.

    """
    try:
        adata.X = adata.raw.X.copy()
    except ValueError:
        if not adata.var_names.equals(adata.raw.var_names):
            warnings.warn(
                "adata.raw.var_names != adata.var_names; re-aligning to restore adata.X",
                category=UserWarning,
                skip_file_prefixes=_warn_skips,
            )
            adata.X = adata.raw[:, adata.var_names].X.copy()
        else:
            raise

    # if issparse(adata.X):
    #     adata.X = adata.X.toarray()


def to_sparse(adata):
    """Convert adata.X and adata.layers to sparse in-place. Ignores adata.raw.

    Args:
        adata (AnnData): Annotated data matrix.

    """
    if not issparse(adata.X):
        adata.X = csr_matrix(adata.X)

    for layer in adata.layers:
        if not issparse(adata.layers[layer]):
            adata.layers[layer] = csr_matrix(adata.layers[layer])


def to_dense(adata):
    """Convert adata.X and adata.layers to dense in-place. Ignores adata.raw.

    Args:
        adata (AnnData): Annotated data matrix.

    """
    if issparse(adata.X):
        adata.X = adata.X.toarray()

    for layer in adata.layers:
        if issparse(adata.layers[layer]):
            adata.layers[layer] = adata.layers[layer].toarray()
