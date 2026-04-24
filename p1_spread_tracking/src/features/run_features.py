import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_all
from src.features.engineering import create_features, select_features

FEATURES_PATH = ROOT / "data" / "features.csv"


def main():
    t0 = time.time()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    print("Loading data from data/raw/ ...")
    df = load_all()
    print(f"  Input shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ------------------------------------------------------------------
    # Create features
    # ------------------------------------------------------------------
    print("\nEngineering features ...")
    wide = create_features(df)
    print(f"  After create_features: {wide.shape[0]:,} rows × {wide.shape[1]} columns")

    # ------------------------------------------------------------------
    # Select features
    # ------------------------------------------------------------------
    print("\nSelecting features ...")
    selected_cols, reduced_df = select_features(wide)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    print(f"\nSaving to {FEATURES_PATH.relative_to(ROOT)} ...")
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    reduced_df.to_csv(FEATURES_PATH)
    print(f"  Saved: {reduced_df.shape[0]:,} rows × {reduced_df.shape[1]} columns")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Input rows:        {df.shape[0]:,}")
    print(f"  Output rows:       {reduced_df.shape[0]:,}")
    print(f"  Features before:   {wide.shape[1]}")
    print(f"  Features after:    {len(selected_cols)}")
    print(f"  Dropped:           {wide.shape[1] - len(selected_cols)}")
    print("\n  Kept features:")
    for col in selected_cols:
        print(f"    {col}")
    print(f"\n  Elapsed: {elapsed:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
