from __future__ import annotations

from pathlib import Path

import pandas as pd

from desim.domain.metrics import ModelResult


def results_to_dataframe(results: list[ModelResult]) -> pd.DataFrame:
    # Converts model results into a pandas DataFrame so they can be filtered,
    # plotted, or exported without additional manual transformation.
    return pd.DataFrame([result.as_dict() for result in results])


def export_dataframe(dataframe: pd.DataFrame, target: str | Path) -> Path:
    # CSV keeps the export dependency-light and works well with Excel as well.
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)
    return path
