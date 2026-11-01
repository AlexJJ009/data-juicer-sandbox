"""Check full-reference pairing, holdout exclusion and input immutability on synthetic cells."""

from typing import Any

import anndata as ad
import pytest

from data_juicer_sandbox.specific_hooks.single_cell.config import Request
from data_juicer_sandbox.specific_hooks.single_cell.reference import (
    arithmetic_population,
)


def test_full_reference_excludes_unpaired_types_and_holdout_stim(payload: dict[str, Any]) -> None:
    """Keep only paired non-holdout types and holdout controls without mutating input."""
    request = Request.from_dict(payload)
    data = ad.read_h5ad(request.data_path)
    data = data[~((data.obs.type == "B") & (data.obs.condition == "s"))].copy()
    names = data.obs_names.tolist()
    result = arithmetic_population(data[:1], request, data)
    assert set(result.obs.type) == {"held", "A"}
    assert result.n_obs == 18
    assert not ((result.obs.type == "held") & (result.obs.condition == "s")).any()
    assert data.obs_names.tolist() == names
    assert "_scgen_setup" not in data.uns
    with pytest.raises(ValueError, match="paired non-holdout"):
        arithmetic_population(data[:1], request, data[data.obs.type != "A"])
