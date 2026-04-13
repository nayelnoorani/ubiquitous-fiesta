import json
import time
import requests
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

POOLS_URL = "https://yields.llama.fi/pools"
CHART_URL = "https://yields.llama.fi/chart/{pool_id}"

TARGET_POOLS = {
    ("aave-v3", "USDC", "Ethereum"),
    ("compound-v3", "USDC", "Ethereum"),
}


def fetch_pools() -> list[dict]:
    response = requests.get(POOLS_URL, timeout=30)
    response.raise_for_status()
    return response.json()["data"]


def filter_target_pools(pools: list[dict]) -> list[dict]:
    matched = []
    for pool in pools:
        key = (pool.get("project"), pool.get("symbol"), pool.get("chain"))
        if key in TARGET_POOLS and pool.get("poolMeta") is None:
            matched.append({
                "pool_id": pool["pool"],
                "project": pool["project"],
                "symbol": pool["symbol"],
                "chain": pool["chain"],
                "pool_meta": pool.get("poolMeta"),
                "tvl_usd": pool.get("tvlUsd"),
            })
    return matched


def fetch_chart(pool_id: str) -> dict:
    url = CHART_URL.format(pool_id=pool_id)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()["data"]


def save_raw(filename: str, data: object) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f)
    return path


def run():
    print("Fetching pool list from DefiLlama...")
    all_pools = fetch_pools()
    targets = filter_target_pools(all_pools)

    if not targets:
        raise RuntimeError("No matching pools found. Check project/symbol/chain filters.")

    print(f"Found {len(targets)} matching pool(s):")
    for p in targets:
        print(f"  {p['project']} | {p['symbol']} | {p['chain']} | {p['pool_id']}")

    save_raw("pools_metadata.json", targets)

    for pool in targets:
        pool_id = pool["pool_id"]
        label = f"{pool['project']}_{pool['symbol']}_{pool['chain']}".lower().replace("-", "_")
        print(f"Fetching chart data for {label}...")

        chart_data = fetch_chart(pool_id)
        filename = f"chart_{label}_{pool_id[:8]}.json"
        path = save_raw(filename, chart_data)
        print(f"  Saved {len(chart_data)} records → {path.name}")

        time.sleep(0.5)

    print("Ingestion complete.")


if __name__ == "__main__":
    run()
