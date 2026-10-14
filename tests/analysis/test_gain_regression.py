from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from wikitrace.analysis.gain_regression import compute, read_rows

FIXTURES = Path(__file__).parent / "fixtures"
PAPER_METHODS = ["00I-inject", "RCI-traverse", "awm", "expel", "reasoningbank"]


@pytest.fixture(scope="module")
def tables():
    return compute(read_rows(FIXTURES / "paper_per_sample.csv"))


def close(mine: pd.DataFrame, frozen: pd.DataFrame, keys: list[str]) -> None:
    m = mine.merge(frozen, on=keys, suffixes=("", "_paper"))
    assert len(m) == len(frozen)
    for col in ("estimate", "ci_lo", "ci_hi"):
        assert (m[col] - m[f"{col}_paper"]).abs().max() < 1e-4 + 1e-9


def test_suite_average(tables):
    frozen = pd.read_csv(FIXTURES / "stats_suite_average.csv").rename(columns={"arm": "method"})
    frozen = frozen[frozen.method.isin(PAPER_METHODS + ["floor"])][["metric", "method", "estimate", "ci_lo", "ci_hi"]]
    close(tables["average"], frozen, ["metric", "method"])


def test_per_suite(tables):
    frozen = pd.read_csv(FIXTURES / "stats_suite_gain_regression.csv").rename(columns={"arm": "method"})
    frozen = frozen[(frozen.variant == "majority") & frozen.method.isin(PAPER_METHODS)]
    close(tables["suite"], frozen[["metric", "suite", "method", "eligible_tasks", "estimate", "ci_lo", "ci_hi"]],
          ["metric", "suite", "method", "eligible_tasks"])


def test_noise_floor(tables):
    frozen = pd.read_csv(FIXTURES / "stats_suite_noise_floor.csv").rename(columns={"arm": "method"})
    frozen = frozen[frozen.floor == "rerun"].assign(method="floor")[["metric", "suite", "method", "estimate", "ci_lo",
                                                                  "ci_hi"]]
    close(tables["suite"], frozen, ["metric", "suite", "method"])


def test_row_order_does_not_matter(tables):
    rows = read_rows(FIXTURES / "paper_per_sample.csv").sample(frac=1, random_state=0)
    shuffled = compute(rows)
    for name, t in tables.items():
        keys = [c for c in ("metric", "benchmark", "suite", "method") if c in t]
        pd.testing.assert_frame_equal(t.sort_values(keys).reset_index(drop=True),
                                      shuffled[name].sort_values(keys).reset_index(drop=True))


def test_a_dropped_row_is_refused():
    rows = read_rows(FIXTURES / "paper_per_sample.csv")
    drop = rows[(rows.suite == "valid-seen") & (rows.method == "RCI-traverse") & (rows.trial == 0)].index[0]
    sample_id = rows.loc[drop, "sample_id"]
    with pytest.raises(ValueError, match=rf"^\(valid-seen, RCI-traverse, {sample_id}\) has trials \[1, 2\]"):
        compute(rows.drop(index=drop))


def test_a_missing_baseline_trial_is_refused():
    rows = read_rows(FIXTURES / "paper_per_sample.csv")
    nm = rows[(rows.suite == "retail") & (rows.method == "no-memory")]
    rows = rows.drop(index=nm[nm.trial.isin([1, 2])].index)
    with pytest.raises(ValueError, match=r"^\(retail, no-memory, .+\) has trials \[0, 3, 4, 5\], expected \[0, 1"):
        compute(rows)
