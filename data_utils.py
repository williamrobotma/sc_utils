"""Utils for manipulating sc data."""

import contextlib
import logging
import os
import pickle
import subprocess
import warnings
from pathlib import Path
from typing import Literal, Optional, Sequence

import anndata as ad
import gffutils
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix, issparse
from sklearn import model_selection
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)
_warn_skips = (os.path.dirname(__file__),)

ENSEMBL_82_URL = "ftp://ftp.ensembl.org/pub/grch37/release-84/gtf/homo_sapiens/Homo_sapiens.GRCh37.82.gtf.gz"
ENSEMBL_109_URL = "https://ftp.ensembl.org/pub/release-109/gtf/homo_sapiens/Homo_sapiens.GRCh38.109.gtf.gz"


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
    """Return ``a.toarray()`` when ``a`` is sparse."""
    if issparse(a):
        return a.toarray()

    return a


def _coerce_sparse_indices_to_int32(x):
    """Return sparse matrix ``x`` with ``indices``/``indptr`` as ``int32``."""
    if not issparse(x):
        return x

    indices = getattr(x, "indices", None)
    indptr = getattr(x, "indptr", None)
    if indices is None or indptr is None:
        return x

    if indices.dtype == np.int32 and indptr.dtype == np.int32:
        return x

    out = x.copy()
    out.indices = indices.astype(np.int32, copy=False)
    out.indptr = indptr.astype(np.int32, copy=False)
    return out


HVGUMAPVariant = Literal["loglsn", "lograw"]


def infer_hvg_umap_variant(
    *,
    raw_counts: bool,
    log1p_raw_counts: Optional[bool] = None,
) -> HVGUMAPVariant:
    """Infer which saved HVG UMAP variant matches the current input contract.

    Args:
        raw_counts: Whether the active data path starts from raw counts.
        log1p_raw_counts: Optional override for whether raw counts are log1p
            transformed before projection. When omitted, this follows
            ``raw_counts`` for backward compatibility with the repository's
            saved UMAP assets.

    Returns:
        HVGUMAPVariant: ``"lograw"`` when the projector should expect
        ``log1p(counts)``, otherwise ``"loglsn"`` for ``loglsn_counts``.

    Notes:
        Current repository conventions support ``loglsn`` for UMAPs fit on
        ``loglsn_counts`` and ``lograw`` for UMAPs fit on ``log1p(counts)``.
    """
    if log1p_raw_counts is None:
        log1p_raw_counts = raw_counts

    return "lograw" if log1p_raw_counts else "loglsn"


def get_hvg_umap_key(*, hvg_umap_variant: HVGUMAPVariant) -> str:
    """Return the saved ``obsm`` key for an HVG UMAP variant."""
    if hvg_umap_variant == "lograw":
        return "X_umap_HVGs_lograw"
    if hvg_umap_variant == "loglsn":
        return "X_umap_HVGs"
    raise ValueError(f"Unsupported HVGUMAPVariant: {hvg_umap_variant!r}.")


def get_hvg_umap_model_path(
    processed_dset_dir: Path | str,
    *,
    pc_only: bool,
    hvg_umap_variant: HVGUMAPVariant,
) -> Path:
    """Return the saved HVG UMAP pickle path."""
    processed_dset_dir = Path(processed_dset_dir)
    if hvg_umap_variant == "lograw":
        filename = "umap_pconly_hvg_lograw.pkl" if pc_only else "umap_hvg_lograw.pkl"
    elif hvg_umap_variant == "loglsn":
        filename = "umap_pconly_hvg.pkl" if pc_only else "umap_hvg.pkl"
    else:
        raise ValueError(f"Unsupported HVGUMAPVariant: {hvg_umap_variant!r}.")

    return processed_dset_dir / filename


def get_preencoded_dset_rel_path(
    *,
    model_name: str,
    model_version: str,
    torch_seed: int | str,
) -> Path:
    """Return the embeddings subdirectory used by ``c-vae.py``."""
    return Path(model_name) / model_version / str(torch_seed)


def build_preencoded_dset_path(
    dset_path: Path | str,
    *,
    model_name: str,
    model_version: str,
    torch_seed: int | str,
) -> Path:
    """Return the embeddings path that ``c-vae.py`` writes for a dataset."""
    dset_path = Path(dset_path)
    return (
        dset_path.parent
        / get_preencoded_dset_rel_path(
            model_name=model_name,
            model_version=model_version,
            torch_seed=torch_seed,
        )
        / f"{dset_path.stem}_embeddings.zarr"
    )


def infer_non_preencoded_dset_path(
    preencoded_dset_path: Path | str,
    *,
    model_name: str,
    model_version: str,
    torch_seed: int | str,
) -> Path:
    """Recover the source dataset path from a ``*_embeddings.zarr`` path."""
    preencoded_dset_path = Path(preencoded_dset_path)
    expected_rel_path = get_preencoded_dset_rel_path(
        model_name=model_name,
        model_version=model_version,
        torch_seed=torch_seed,
    )

    if preencoded_dset_path.suffix != ".zarr" or not preencoded_dset_path.stem.endswith(
        "_embeddings"
    ):
        raise ValueError(f"Not a c-vae embeddings path: {preencoded_dset_path}")

    if (
        preencoded_dset_path.parent.parts[-len(expected_rel_path.parts) :]
        != expected_rel_path.parts
    ):
        raise ValueError(
            "Embeddings path does not match c-vae output layout for "
            f"{expected_rel_path}: {preencoded_dset_path}"
        )

    dataset_parent = preencoded_dset_path.parents[len(expected_rel_path.parts)]
    dataset_stem = preencoded_dset_path.stem.removesuffix("_embeddings")
    candidates = [
        dataset_parent / f"{dataset_stem}.h5ad",
        dataset_parent / f"{dataset_stem}.zarr",
    ]
    existing_candidates = [candidate for candidate in candidates if candidate.exists()]

    if len(existing_candidates) == 1:
        return existing_candidates[0]
    if len(existing_candidates) > 1:
        raise ValueError(
            "Ambiguous source dataset path for "
            f"{preencoded_dset_path}: {existing_candidates}"
        )

    raise FileNotFoundError(
        f"Could not infer source dataset for {preencoded_dset_path}. Tried: {candidates}"
    )


class HVGUMAPProjector:
    """Load and apply a saved HVG UMAP projector."""

    def __init__(
        self,
        *,
        processed_dset_dir: Path | str,
        pc_only: bool,
        hvg_umap_variant: HVGUMAPVariant,
        hvg_mask=None,
        input_is_prelogged: bool,
        transform_batch_size: Optional[int] = None,
    ) -> None:
        self.processed_dset_dir = Path(processed_dset_dir)
        self.pc_only = pc_only
        self.hvg_umap_variant = hvg_umap_variant
        self.hvg_mask = None if hvg_mask is None else np.asarray(hvg_mask, dtype=bool)
        self.input_is_prelogged = input_is_prelogged
        if transform_batch_size is not None and transform_batch_size <= 0:
            raise ValueError(
                "transform_batch_size must be a positive integer when provided."
            )
        self.transform_batch_size = transform_batch_size

        self.key = get_hvg_umap_key(hvg_umap_variant=self.hvg_umap_variant)
        self.model_path = get_hvg_umap_model_path(
            self.processed_dset_dir,
            pc_only=self.pc_only,
            hvg_umap_variant=self.hvg_umap_variant,
        )
        self._umapper = None

    def get_umapper(self):
        """Load and cache the saved UMAP model."""
        if self._umapper is None:
            logger.debug("Loading HVG UMAP projector from %s", self.model_path)
            with open(self.model_path, "rb") as file:
                self._umapper = pickle.load(file)

            search_index = getattr(self._umapper, "_knn_search_index", None)
            if search_index is not None:
                for attr in ("_search_graph", "_neighbor_graph"):
                    graph = getattr(search_index, attr, None)
                    if graph is not None:
                        setattr(
                            search_index,
                            attr,
                            _coerce_sparse_indices_to_int32(graph),
                        )
        return self._umapper

    def _prepare_matrix(self, X):
        # Full-gene matrices need the HVG subset; HVG-only matrices pass through unchanged.
        X = toarray_if_sparse(X)
        if self.hvg_mask is not None and X.shape[1] == self.hvg_mask.shape[0]:
            X = X[:, self.hvg_mask]
        if not self.input_is_prelogged:
            X = np.log1p(X)
        return X

    def _transform_in_chunks(self, X, *, batch_size: int) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer when provided.")

        umapper = self.get_umapper()
        transformed_l = []
        total = X.shape[0]
        # Use a single overall progress bar and suppress per-chunk stdout/stderr
        # emitted by the underlying umapper.transform to keep logs concise.
        with tqdm(total=total, desc="UMAP transform", unit="rows") as pbar:
            for start in range(0, total, batch_size):
                stop = min(start + batch_size, total)
                with (
                    open(os.devnull, "w") as devnull,
                    contextlib.redirect_stdout(devnull),
                    contextlib.redirect_stderr(devnull),
                ):
                    chunk_trans = umapper.transform(self._prepare_matrix(X[start:stop]))
                transformed_l.append(chunk_trans)
                pbar.update(stop - start)

        if not transformed_l:
            return np.empty((0, umapper.embedding_.shape[1]))

        return np.concatenate(transformed_l, axis=0)

    def _adata_matrix(self, adata: ad.AnnData):
        if self.hvg_mask is not None and adata.n_vars == self.hvg_mask.shape[0]:
            adata = adata[:, self.hvg_mask]

        if self.hvg_umap_variant == "lograw":
            try:
                return np.log1p(toarray_if_sparse(adata.layers["counts"]))
            except KeyError:
                message = (
                    f"{self.key}: `counts` layer missing; falling back to `adata.X` "
                    "for lograw UMAP projection."
                )
                logger.warning(message)
                warnings.warn(message, UserWarning, stacklevel=2)
                return self._prepare_matrix(adata.X)

        try:
            return toarray_if_sparse(adata.layers["loglsn_counts"])
        except KeyError:
            message = (
                f"{self.key}: `loglsn_counts` layer missing; falling back to `adata.X` "
                "for loglsn UMAP projection."
            )
            logger.warning(message)
            warnings.warn(message, UserWarning, stacklevel=2)
            return self._prepare_matrix(adata.X)

    def transform(self, X, *, batch_size: Optional[int] = None):
        """Project an expression matrix into the saved HVG UMAP space."""
        if batch_size is None:
            batch_size = self.transform_batch_size

        if batch_size is None:
            return self.get_umapper().transform(self._prepare_matrix(X))

        if batch_size >= X.shape[0]:
            return self.get_umapper().transform(self._prepare_matrix(X))

        return self._transform_in_chunks(X, batch_size=batch_size)

    def transform_adata(
        self,
        adata: ad.AnnData,
        *,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Project an ``AnnData`` object into the saved HVG UMAP space."""
        if batch_size is None:
            batch_size = self.transform_batch_size

        if batch_size is None:
            return self.get_umapper().transform(self._adata_matrix(adata))

        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer when provided.")
        if batch_size >= adata.n_obs:
            return self.get_umapper().transform(self._adata_matrix(adata))

        transformed_l = []
        umapper = self.get_umapper()
        total = adata.n_obs
        with tqdm(total=total, desc="UMAP transform (adata)", unit="rows") as pbar:
            for start in range(0, total, batch_size):
                stop = min(start + batch_size, total)
                with (
                    open(os.devnull, "w") as devnull,
                    contextlib.redirect_stdout(devnull),
                    contextlib.redirect_stderr(devnull),
                ):
                    chunk_trans = umapper.transform(
                        self._adata_matrix(adata[start:stop])
                    )
                transformed_l.append(chunk_trans)
                pbar.update(stop - start)

        if not transformed_l:
            return np.empty((0, umapper.embedding_.shape[1]))

        return np.concatenate(transformed_l, axis=0)

    def transform_splits(
        self,
        X_d,
        *,
        source_split: str = "train",
        source_index=None,
        copy_splits: Sequence[str] = ("val", "test"),
    ):
        """Project one split and reuse that projection for sibling split keys.

        Args:
            X_d: Mapping from split name to expression matrix.
            source_split: Split whose matrix is actually projected.
            source_index: Optional row index applied before projection, used by
                the evaluation scripts to reuse only the reselected train rows.
            copy_splits: Additional split names that should receive the same
                projected output object.

        Returns:
            dict: Mapping where ``source_split`` and every name in
            ``copy_splits`` point at the projected source matrix.

        Warnings:
            This intentionally reuses the same transformed array for every split
            in ``copy_splits``. Callers must not assume those outputs are
            independently projected from their own source matrices.

        Notes:
            A debug log is emitted when reuse is applied so that this behavior
            remains visible during evaluation debugging.
        """
        X_source = X_d[source_split]
        if source_index is not None:
            X_source = X_source[source_index]

        transformed = self.transform(X_source)
        out = {source_split: transformed}
        if copy_splits:
            logger.debug(
                "Reusing projected split '%s' for %s in HVGUMAPProjector",
                source_split,
                tuple(copy_splits),
            )
        for split in copy_splits:
            out[split] = transformed
        return out

    def populate_adata_if_not_present(
        self, adata: ad.AnnData, *, source_adata: Optional[ad.AnnData] = None
    ):
        """Populate ``adata.obsm[self.key]`` if needed and return the embedding.

        Args:
            adata: Target `AnnData` whose ``obsm`` entry may be filled in place.
            source_adata: Optional `AnnData` whose expression matrix should be
                projected instead of ``adata`` when populating a missing
                embedding. This is used when embeddings were generated from a
                preencoded dataset but the visualization should follow the source
                dataset's counts/log-count layers.

        Returns:
            np.ndarray: Embedding stored at ``adata.obsm[self.key]``.

        Warnings:
            Warns and logs when ``source_adata`` observation names do not match
            ``adata.obs_names``, because the populated embedding may then be
            misaligned with ``adata`` rows.

        Notes:
            This mutates ``adata.obsm`` only when the key is absent; otherwise
            the existing embedding is returned unchanged.
        """
        if self.key not in adata.obsm:
            reference_adata = adata if source_adata is None else source_adata
            if source_adata is not None and not adata.obs_names.equals(
                source_adata.obs_names
            ):
                message = (
                    f"{self.key}: source_adata.obs_names does not match adata.obs_names; "
                    "populated embedding may be misaligned."
                )
                logger.warning(message)
                warnings.warn(message, UserWarning, stacklevel=2)
            logger.debug("Populating missing HVG UMAP embedding '%s'", self.key)
            adata.obsm[self.key] = self.transform_adata(reference_adata)
        return adata.obsm[self.key]


def get_h5ad(dset_path, log1p=False, raw_counts=False, force_raw_log1p=False):
    """Load AnnData from a .h5ad and sets X to the appropriate layer for log1p.

    Args:
        dset_path (str): Path to the h5ad file.
        log1p (bool, optional): If True, use 'loglsn_counts' layer as reference,
            otherwise use 'lsn_counts'. Ignored if `raw_counts=True` unless
            `force_raw_log1p=True`. Defaults to False.
        raw_counts (bool, optional): If True, use raw counts. Defaults to False.
        force_raw_log1p (bool, optional): If True and `raw_counts=True`, use
            log1p transform on raw counts. Defaults to False.

    Returns:
        AnnData: Loaded AnnData object with X matrix set to reference layer.
    """

    adata = ad.read_h5ad(dset_path)

    data_issparse = issparse(adata.X)
    x_layer = "loglsn_counts" if log1p else "lsn_counts"

    if raw_counts:
        if log1p and not force_raw_log1p:
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

    if force_raw_log1p:
        if log1p and raw_counts:
            sc.pp.log1p(adata)
        else:
            warnings.warn(
                f"`force_raw_log1p=True` but `{log1p=}`, `{raw_counts=}`; no log1p applied",
                UserWarning,
            )

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


def sample_eval_train_indices(
    adata: ad.AnnData,
    *,
    split_key: str,
    split_on: str = "celltype",
    seed: int = 2895,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the train-set evaluation subsets used by VAE and diffusion plots.

    This mirrors the repository's existing evaluation convention: draw two
    disjoint train subsets, each with the same size as the validation split and
    stratified by ``split_on``.

    Args:
        adata: AnnData object containing train and validation split labels.
        split_key: Observation column identifying dataset splits.
        split_on: Observation column used for stratified train sampling.
        seed: Seed for the local NumPy random generator.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(selected, reselected)`` train-row
        indices relative to the train split only.

    Notes:
        Both returned index arrays are expressed in train-split coordinates, not
        full-adata row coordinates.
    """
    rng = np.random.default_rng(seed=seed)
    train_mask = adata.obs[split_key] == "train"
    val_size = int((adata.obs[split_key] == "val").sum())
    train_labels = adata.obs.loc[train_mask, split_on]

    remainder, selected_samples = model_selection.train_test_split(
        train_labels,
        test_size=val_size,
        random_state=rng.integers(2**32),
        stratify=train_labels,
    )

    _, reselected_samples = model_selection.train_test_split(
        remainder,
        test_size=val_size,
        random_state=rng.integers(2**32),
        stratify=remainder,
    )

    train_index = adata.obs.index[train_mask]
    selected = train_index.get_indexer(selected_samples.index)
    reselected = train_index.get_indexer(reselected_samples.index)
    return selected, reselected


def download_gtf(out_dir, url) -> Path:
    """Downloads and extracts a GTF file from a URL.

    Args:
        out_dir (str): Directory to download the file to. Will be created if
            it does not already exist.
        url (str): URL to download the GTF file from.

    Returns:
        pathlib.Path: Path to the extracted GTF file.

    Raises:
        subprocess.CalledProcessError: If the download or unzip command exits
            with a non-zero status.
    """
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    gtf_gz = os.path.basename(url)
    gtf_gz_path = out_dir_path / gtf_gz
    gtf_path = gtf_gz_path.with_suffix("")

    if not gtf_path.exists():
        if not gtf_gz_path.exists():
            print(f"Downloading {url} to {gtf_gz_path}")
            subprocess.run(
                ["curl", "--create-dirs", "-o", str(gtf_gz_path), url],
                check=True,
            )
        print(f"Unzipping {gtf_gz_path}")
        subprocess.run(["gunzip", str(gtf_gz_path)], check=True)

    return gtf_path


def get_reference_genome_db(
    out_dir,
    gtf_fname="Homo_sapiens.GRCh38.84.gtf",
    cache_fname=None,
    use_cache=True,
):
    """Creates or loads a gffutils database from a GTF file.

    Adapted from Ryan Dale's (https://www.biostars.org/u/528/) comment at
    https://www.biostars.org/p/152517/.

    Args:
        out_dir (str): Directory containing the GTF file (and where the
            database cache will be created/loaded).
        gtf_fname (str): Filename of the GTF file. Defaults to
            "Homo_sapiens.GRCh38.84.gtf".
        cache_fname (str, optional): Filename for the database cache. If None,
            will use the GTF filename with a .db extension. Defaults to None.
        use_cache (bool): Whether to use an existing cache file. Defaults to
            True.

    Raises:
        FileNotFoundError: If the specified GTF file does not exist under
            ``out_dir``.
        Exception: Propagates exceptions raised by ``gffutils`` when creating
            the database (these indicate problems parsing the GTF or writing
            the cache).

    Returns:
        gffutils.FeatureDB: The database of genomic features.
    """
    if cache_fname is None:
        cache_fname = os.path.splitext(gtf_fname)[0] + ".db"

    out_dir_path = Path(out_dir)
    gtf_path = out_dir_path / gtf_fname
    cache_path = out_dir_path / cache_fname
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
    if cache_path.exists() and use_cache:
        db = gffutils.FeatureDB(str(cache_path))
    else:
        db = gffutils.create_db(
            str(gtf_path),
            str(cache_path),
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


def save_adata_embeddings(
    adata: ad.AnnData,
    out_path: Path,
    layer_name: str = None,
):
    """Saves embeddings from an AnnData object to a .npz file.

    Args:
        adata (ad.AnnData): The AnnData object containing the embeddings, where
            adata.obs_names are the keys, while rows of adata.X or
            adata.layers[layer_name] are the values to be saved.
        out_path (Path): The path to save the .npz file to.
        layer_name (str, optional): The name of the layer containing the
            embeddings. If None, uses adata.X. Defaults to None.
    """
    embs_d = {}
    layer = adata.layers[layer_name] if layer_name is not None else adata.X

    for i in range(adata.n_obs):
        embs_d[adata.obs_names[i]] = layer[i]

    if not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **embs_d)
