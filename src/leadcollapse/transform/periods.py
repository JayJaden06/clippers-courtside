"""Map play-by-play periods onto the box-score split names (FULL, H1, H2, Q1-Q4, OT)."""

from __future__ import annotations

import pandas as pd


def splits_for_period(period: int) -> list[str]:
    """Every split a period belongs to: 3 -> [FULL, H2, Q3]; 5 -> [FULL, OT]."""
    if period <= 2:
        return ["FULL", "H1", f"Q{period}"]
    if period <= 4:
        return ["FULL", "H2", f"Q{period}"]
    return ["FULL", "OT"]


def explode_to_splits(df: pd.DataFrame, period_col: str = "period") -> pd.DataFrame:
    """Repeat each row once per split its period belongs to, in a ``split`` column."""
    out = df.assign(split=df[period_col].map(splits_for_period))
    return out.explode("split", ignore_index=True)


def count_by_split(df: pd.DataFrame, keys: list[str], flags: dict[str, pd.Series]) -> pd.DataFrame:
    """Sum boolean/int ``flags`` (name -> Series aligned with ``df``) by ``keys`` + split."""
    base = df[keys + ["period"]].assign(**flags)
    exploded = explode_to_splits(base)
    return exploded.groupby(keys + ["split"], as_index=False)[list(flags)].sum()
