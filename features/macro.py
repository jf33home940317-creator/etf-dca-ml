import pandas as pd


def build_macro_frame(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    vix = raw["VIX"]["close"]
    tnx = raw["TNX"]["close"]
    dxy = raw["DXY"]["close"]
    smh = raw["SMH"]["close"]

    out = pd.DataFrame(index=vix.index)
    out.index.name = "date"

    out["vix"] = vix
    out["vix_ma200_ratio"] = vix / vix.rolling(200).mean()
    out["vix_chg5"] = vix.pct_change(5)

    out["tnx"] = tnx
    out["tnx_slope"] = tnx.diff(20)

    out["dxy"] = dxy
    out["dxy_chg5"] = dxy.pct_change(5)

    out["smh_roc1"] = smh.pct_change(1)

    return out
