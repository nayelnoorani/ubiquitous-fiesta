import pandas as pd

# Required columns and their expected pandas dtype kinds:
# 'M' = datetime, 'f' = float, 'O' = object (string)
REQUIRED_SCHEMA = {
    "timestamp": "M",
    "apyBase":   "f",
    "tvlUsd":    "i",
    "project":   "O",
    "symbol":    "O",
    "chain":     "O",
}

# Inclusive [min, max] bounds for numeric columns
VALUE_BOUNDS = {
    "apyBase":    (0.0,  500.0),
    "apy":        (0.0,  500.0),
    "apyReward":  (0.0,  500.0),
    "tvlUsd":     (0.0,  None),   # None = no upper bound
}

# Column whose distribution is treated as the "class" label for check 5
GROUP_COL = "project"

NULL_CRITICAL_THRESHOLD = 0.50
NULL_WARN_THRESHOLD     = 0.20
MIN_ROWS_CRITICAL       = 100
MIN_ROWS_WARN           = 1000
CLASS_MIN_SHARE         = 0.05   # warn if any class < 5 % of data


def check_data_quality(df: pd.DataFrame) -> dict:
    failures   = []
    warnings   = []
    statistics = {"total_rows": len(df), "total_nulls_by_column": {}}

    # ------------------------------------------------------------------
    # Check 1 — Schema validation
    # ------------------------------------------------------------------
    for col, kind in REQUIRED_SCHEMA.items():
        if col not in df.columns:
            failures.append(f"[schema] Required column missing: '{col}'")
        elif df[col].dtype.kind != kind:
            failures.append(
                f"[schema] Column '{col}' has dtype '{df[col].dtype}' "
                f"(expected kind '{kind}')"
            )

    # ------------------------------------------------------------------
    # Check 2 — Row count
    # ------------------------------------------------------------------
    n = len(df)
    if n < MIN_ROWS_CRITICAL:
        failures.append(f"[row_count] Only {n:,} rows — minimum required is {MIN_ROWS_CRITICAL:,}")
    elif n < MIN_ROWS_WARN:
        warnings.append(f"[row_count] {n:,} rows — results may be unreliable below {MIN_ROWS_WARN:,}")

    # ------------------------------------------------------------------
    # Check 3 — Null rates
    # ------------------------------------------------------------------
    null_counts = df.isnull().sum()
    statistics["total_nulls_by_column"] = null_counts.to_dict()

    for col, count in null_counts.items():
        rate = count / n if n > 0 else 0
        if rate > NULL_CRITICAL_THRESHOLD:
            failures.append(
                f"[nulls] '{col}' is {rate:.1%} null ({count:,} / {n:,} rows)"
            )
        elif rate > NULL_WARN_THRESHOLD:
            warnings.append(
                f"[nulls] '{col}' is {rate:.1%} null ({count:,} / {n:,} rows)"
            )

    # ------------------------------------------------------------------
    # Check 4 — Value ranges
    # ------------------------------------------------------------------
    for col, (lo, hi) in VALUE_BOUNDS.items():
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if lo is not None and (series < lo).any():
            n_bad = (series < lo).sum()
            failures.append(
                f"[range] '{col}' has {n_bad:,} value(s) below minimum {lo}"
            )
        if hi is not None and (series > hi).any():
            n_bad = (series > hi).sum()
            failures.append(
                f"[range] '{col}' has {n_bad:,} value(s) above maximum {hi}"
            )

    # ------------------------------------------------------------------
    # Check 5 — Group / class distribution
    # ------------------------------------------------------------------
    if GROUP_COL in df.columns:
        counts = df[GROUP_COL].value_counts(normalize=True)
        statistics["group_distribution"] = df[GROUP_COL].value_counts().to_dict()

        if len(counts) < 2:
            failures.append(
                f"[distribution] '{GROUP_COL}' has only {len(counts)} unique value(s) — "
                "need at least 2 for cross-protocol comparison"
            )
        else:
            for label, share in counts.items():
                if share < CLASS_MIN_SHARE:
                    warnings.append(
                        f"[distribution] '{GROUP_COL}={label}' represents only "
                        f"{share:.1%} of data — severely imbalanced"
                    )
    else:
        warnings.append(f"[distribution] Group column '{GROUP_COL}' not found — skipping check 5")

    return {
        "success":    len(failures) == 0,
        "failures":   failures,
        "warnings":   warnings,
        "statistics": statistics,
    }


def print_quality_report(result: dict):
    status = "PASSED" if result["success"] else "FAILED"
    print("=" * 55)
    print(f"DATA QUALITY GATE: {status}")
    print("=" * 55)

    if result["failures"]:
        print(f"\nCRITICAL FAILURES ({len(result['failures'])}):")
        for msg in result["failures"]:
            print(f"  [FAIL] {msg}")

    if result["warnings"]:
        print(f"\nWARNINGS ({len(result['warnings'])}):")
        for msg in result["warnings"]:
            print(f"  [WARN] {msg}")

    if not result["failures"] and not result["warnings"]:
        print("\n  All checks passed with no warnings.")

    stats = result["statistics"]
    print("\nSTATISTICS:")
    print(f"  Total rows: {stats['total_rows']:,}")

    if "group_distribution" in stats:
        print("  Group distribution:")
        for label, count in stats["group_distribution"].items():
            print(f"    {label}: {count:,} rows")

    nulls = {k: v for k, v in stats["total_nulls_by_column"].items() if v > 0}
    if nulls:
        print("  Columns with nulls:")
        for col, count in nulls.items():
            pct = count / stats["total_rows"] * 100
            print(f"    {col}: {count:,} ({pct:.1f}%)")

    print("=" * 55)


if __name__ == "__main__":
    from loader import load_all

    df = load_all()
    result = check_data_quality(df)
    print_quality_report(result)
