"""Synthetic conftest checks for the single-cell worker."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse


@pytest.fixture
def payload(tmp_path):
    obs = pd.DataFrame(
        [(t, c, "d") for t in ("held", "A", "B") for c in ("c", "s") for _ in range(6)],
        columns=["type", "condition", "donor"],
        index=[f"cell{i}" for i in range(36)],
    )
    x = np.tile([1.0, 2.0, 4.0, 7.0], (36, 1)).astype("float32")
    x[obs.condition == "s"] += [3, 1, 2, 4]
    a = ad.AnnData(sparse.csr_matrix(x), obs=obs)
    a.layers["counts"] = a.X.copy()
    path = tmp_path / "data.h5ad"
    a.write_h5ad(path)
    return dict(
        action="sample",
        data_path=str(path),
        work_dir=str(tmp_path / "out"),
        fields=dict(cell_type="type", condition="condition", donor="donor"),
        conditions=dict(control="c", perturbed="s"),
        holdout_cell_type="held",
        budget=12,
        candidates=[dict(id="r1", mixture={"A": 0.5, "B": 0.5}, seed=11)],
    )
