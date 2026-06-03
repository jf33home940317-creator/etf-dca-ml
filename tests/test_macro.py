import numpy as np
import pandas as pd
from features.macro import build_macro_frame


def test_build_macro_frame_columns():
    idx = pd.bdate_range("2024-01-01", periods=250, name="date")
    raw = {
        "VIX": pd.DataFrame({"close": np.linspace(15, 25, 250)}, index=idx),
        "TNX": pd.DataFrame({"close": np.linspace(3.5, 4.5, 250)}, index=idx),
        "DXY": pd.DataFrame({"close": np.linspace(100, 105, 250)}, index=idx),
        "SMH": pd.DataFrame({"close": np.linspace(200, 220, 250)}, index=idx),
    }
    out = build_macro_frame(raw)
    for col in ["vix", "vix_ma200_ratio", "vix_chg5",
                "tnx", "tnx_slope", "dxy", "dxy_chg5", "smh_roc1"]:
        assert col in out.columns
    assert out.index.name == "date"
