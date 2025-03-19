import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


def multipletests_masknan(pvals, **kwargs):
    mask = np.isfinite(pvals)

    pval_corrected = np.full(pvals.shape, np.nan)
    if np.sum(mask) > 0:
        pval_corrected[mask] = multipletests(pvals[mask], **kwargs)[1]

    return pval_corrected


def fdr_cpr(df, correct_for="all", **kwargs):
    # check if an array or series
    if df.ndim < 2:
        return pd.Series(
            multipletests_masknan(df.values, **kwargs),
            index=df.index,
            name=df.name,
        )
    if correct_for == "all":
        return pd.DataFrame(
            multipletests_masknan(df.values.flatten(), **kwargs).reshape(df.shape),
            columns=df.columns,
            index=df.index,
        )
    if correct_for in {"vars_only", "across_rows"}:
        return df.apply(lambda x: multipletests_masknan(x, **kwargs), axis=0)
    if correct_for in {"rows_only", "across_cols"}:
        return df.apply(lambda x: multipletests_masknan(x, **kwargs), axis=1, raw=True)

    raise ValueError(
        f"correct_for must be one of 'all', 'vars_only', 'rows_only', 'across_rows', "
        f"or 'across_cols'; got {correct_for} instead."
    )
