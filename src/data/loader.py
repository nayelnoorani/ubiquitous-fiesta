import json
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def load_pool_metadata() -> list[dict]:
    path = RAW_DIR / "pools_metadata.json"
    with open(path) as f:
        return json.load(f)


def load_chart(pool: dict) -> pd.DataFrame:
    pool_id_prefix = pool["pool_id"][:8]
    label = f"{pool['project']}_{pool['symbol']}_{pool['chain']}".lower().replace("-", "_")
    filename = f"chart_{label}_{pool_id_prefix}.json"

    with open(RAW_DIR / filename) as f:
        records = json.load(f)

    df = pd.DataFrame(records)
    df = df.drop(columns=["il7d", "apyBase7d"], errors="ignore")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["project"] = pool["project"]
    df["symbol"] = pool["symbol"]
    df["chain"] = pool["chain"]

    return df


def load_all() -> pd.DataFrame:
    pools = load_pool_metadata()
    frames = [load_chart(pool) for pool in pools]
    return pd.concat(frames, ignore_index=True)


def print_report(df: pd.DataFrame):
    print("=" * 55)
    print("SHAPE")
    print("=" * 55)
    print(f"  Rows: {df.shape[0]:,}  Columns: {df.shape[1]}")

    print("\n" + "=" * 55)
    print("COLUMNS & DTYPES")
    print("=" * 55)
    for col, dtype in df.dtypes.items():
        print(f"  {col:<20} {dtype}")

    print("\n" + "=" * 55)
    print("SUMMARY STATISTICS")
    print("=" * 55)
    print(df.describe().to_string())

    print("\n" + "=" * 55)
    print("MISSING VALUES")
    print("=" * 55)
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(1)
    report = pd.DataFrame({"missing": missing, "pct": pct})
    report = report[report["missing"] > 0]
    if report.empty:
        print("  No missing values.")
    else:
        print(report.to_string())

    print("\n" + "=" * 55)
    print("DATE RANGE")
    print("=" * 55)
    for project, grp in df.groupby("project"):
        print(f"  {project}: {grp['timestamp'].min().date()} → {grp['timestamp'].max().date()}  ({len(grp):,} records)")

    print("=" * 55)


if __name__ == "__main__":
    df = load_all()
    print_report(df)
