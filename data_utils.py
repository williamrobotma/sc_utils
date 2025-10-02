"""Utils for manipulating sc data."""

import os
import subprocess
import warnings

import anndata as ad
import gffutils
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse

_warn_skips = (os.path.dirname(__file__),)

ENSEMBL_82_URL = (
    "ftp://ftp.ensembl.org/pub/grch37/release-84/gtf/homo_sapiens/" "Homo_sapiens.GRCh37.82.gtf.gz"
)


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


def toarray_if_sparse(a):
    if issparse(a):
        return a.toarray()

    return a


def get_h5ad(dset_path, log1p=False, raw_counts=False):
    """Load AnnData from a .h5ad and sets X to the appropriate layer for log1p.

    Args:
        dset_path (str): Path to the h5ad file.
        log1p (bool, optional): If True, use 'loglsn_counts' layer as reference,
            otherwise use 'lsn_counts'. Ignored if `raw_counts=True`. Defaults
            to False.
        raw_counts (bool, optional): If True, use raw counts. Defaults to False.

    Returns:
        AnnData: Loaded AnnData object with X matrix set to reference layer.
    """

    adata = ad.read_h5ad(dset_path)

    data_issparse = issparse(adata.X)
    x_layer = "loglsn_counts" if log1p else "lsn_counts"

    if raw_counts:
        if log1p:
            warnings.warn("`log1p=True` is ignored when `raw_counts=True`", UserWarning)
        x_layer = "counts"

    try:
        if data_issparse:
            assert np.array_equal(adata.X.data, adata.layers[x_layer].data)
            assert np.array_equal(adata.X.indices, adata.layers[x_layer].indices)
            assert np.array_equal(adata.X.indptr, adata.layers[x_layer].indptr)
        else:
            assert np.array_equal(adata.X, adata.layers[x_layer])
    except AssertionError:
        adata.X = adata.layers[x_layer].copy()

    return adata


def safe_stratify(stratify):
    """Makes stratify arg for sklearn splits safe when there is only one class.

    Args:
        stratify (array-like): Array to stratify.

    Returns:
        `stratify` if there is more than one unique value, else None.
    """
    if len(np.unique(stratify)) > 1:
        return stratify

    return None


def download_gtf(dir, url):
    """Downloads and extracts a GTF file from a URL.

    Args:
        dir (str): Directory to download the file to.
        url (str): URL to download the GTF file from.

    Returns:
        str: Path to the extracted GTF file.
    """
    os.makedirs(dir, exist_ok=True)

    gtf_gz = os.path.basename(url)
    gtf_gz_path = os.path.join(dir, gtf_gz)
    gtf_path = os.path.splitext(gtf_gz_path)[0]

    if not os.path.exists(gtf_path):
        if not os.path.exists(gtf_gz_path):
            print(f"Downloading {url} to {gtf_gz_path}")
            subprocess.run(
                [
                    "curl",
                    "--create-dirs",
                    "-o",
                    gtf_gz_path,
                    url,
                ]
            )
        print(f"Unzipping {gtf_gz_path}")
        subprocess.run(
            [
                "gunzip",
                gtf_gz_path,
            ]
        )

    return gtf_path


def get_reference_genome_db(
    dir,
    gtf_fname="Homo_sapiens.GRCh38.84.gtf",
    cache_fname=None,
    use_cache=True,
):
    """Creates or loads a gffutils database from a GTF file.

    Adapted from Ryan Dale's (https://www.biostars.org/u/528/) comment at
    https://www.biostars.org/p/152517/.

    Args:
        dir (str): Directory containing the GTF file.
        gtf_fname (str): Filename of the GTF file. Defaults to
            "Homo_sapiens.GRCh38.84.gtf".
        cache_fname (str, optional): Filename for the database cache. If None,
            will use the GTF filename with a .db extension. Defaults to None.
        use_cache (bool): Whether to use an existing cache file. Defaults to
            True.

    Returns:
        gffutils.FeatureDB: The database of genomic features.
    """
    if cache_fname is None:
        cache_fname = os.path.splitext(gtf_fname)[0] + ".db"

    gtf_path = os.path.join(dir, gtf_fname)
    cache_path = os.path.join(dir, cache_fname)
    id_spec = {
        "exon": "exon_id",
        "gene": "gene_id",
        "transcript": "transcript_id",
        # # [1] These aren't needed for speed, but they do give nicer IDs.
        # 'CDS': [subfeature_handler],
        # 'stop_codon': [subfeature_handler],
        # 'start_codon': [subfeature_handler],
        # 'UTR':  [subfeature_handler],
    }
    if os.path.exists(cache_path) and use_cache:
        db = gffutils.FeatureDB(cache_path)
    else:
        db = gffutils.create_db(
            gtf_path,
            cache_path,
            # Since Ensembl GTF files now come with genes and transcripts already in
            # the file, we don't want to spend the time to infer them (which we would
            # need to do in an on-spec GTF file)
            disable_infer_genes=True,
            disable_infer_transcripts=True,
            # Here's where we provide our custom id spec
            id_spec=id_spec,
            # "create_unique" runs a lot faster than "merge"
            # See https://pythonhosted.org/gffutils/database-ids.html#merge-strategy
            # for details.
            merge_strategy="create_unique",
            verbose=True,
            force=True,
        )

        for f in db.featuretypes():
            if f == "gene":
                continue

            db.delete(db.features_of_type(f), make_backup=False)

    return db


def populate_vars_from_ref(adata, db):
    """Populates the var attribute of an AnnData object with information from a
    reference genome.

    Args:
        adata (:obj: AnnData): AnnData object to populate.
        db (:obj: gffutils.FeatureDB): Reference genome database.

    Returns:
        None: The input AnnData object is modified in-place.
    """
    if adata.var_names.name == "gene_ids":
        gene_ids = adata.var_names.to_series()
    else:
        gene_ids = adata.var["gene_ids"]

    attrs = gene_ids.map(lambda x: dict(db[x].attributes))
    attrs = pd.DataFrame.from_records(attrs, index=attrs.index).agg(lambda x: x.str[0])

    if "gene_ids" in attrs.columns:
        attrs = attrs.drop(columns=["gene_ids"])
    if "gene_id" in attrs.columns:
        attrs = attrs.drop(columns=["gene_id"])

    adata.var = adata.var.join(attrs, how="left", validate="one_to_one")


def drop_by_obs(adata, drop_obs):
    """Drop observations from AnnData object based on values in obs dataframe.

    Args:
        adata (:obj: AnnData): AnnData object to filter.
        drop_obs (dict): Dictionary where keys are column names in adata.obs and
            values are lists of values to drop from those columns.

    Returns:
        AnnData: Filtered AnnData object with specified observations removed.
    """
    obs = adata.obs.copy()
    obs_indexer = get_bool_mask_to_drop_rows_by_column_values(obs, drop_obs)

    return adata[obs_indexer]


def get_bool_mask_to_drop_rows_by_column_values(df, column_vals_dict):
    """Creates a boolean mask to filter dataframe rows based on column values.

    Args:
        df (:obj: pandas.DataFrame): The dataframe to filter.
        column_vals_dict (dict): Dictionary where keys are column names and
            values are lists of values to exclude from those columns.

    Returns:
        pandas.Series: Boolean mask where True indicates rows to keep.
    """
    bool_indexer = pd.Series(data=True, index=df.index, dtype=bool)
    for k, v in column_vals_dict.items():
        if k not in df.columns:
            warnings.warn(f"'{k}' not found in columns", UserWarning)
            continue

        for x in v:
            if x in df[k].unique():
                bool_indexer = bool_indexer & (df[k] != x)
            else:
                warnings.warn(f"'{x}' not found in '{k}'", UserWarning)

    return bool_indexer
