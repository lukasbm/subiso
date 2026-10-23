"""Aggregate the large-host rows: agreement, per-engine regret, timeouts, crashes.

Missing rows are jobs killed at the driver's wall-clock cap; they are reported
as ``timeout`` and can optionally be censored into the regret (``timeout_cap``)
so that an engine which hangs on the hard cells is not rewarded for it.
"""

from __future__ import annotations

import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import fire


def main(
    rows_dir: str = "/scratch/lukas/tmp/opencode/large_hosts/rows",
    jobs_file: str = "/scratch/lukas/tmp/opencode/large_hosts/jobs.txt",
    timeout_cap: float = 600.0,
    output: str = "",
) -> None:
    records: list[dict[str, str]] = []
    for path in sorted(Path(rows_dir).glob("*.csv")):
        with path.open() as handle:
            records.extend(csv.DictReader(handle))

    expected = []
    if jobs_file and Path(jobs_file).exists():
        for line in Path(jobs_file).read_text().splitlines():
            if line.strip():
                cell, engine, protocol = line.split()
                expected.append((cell, engine, protocol))
    seen = {(record["cell"], record["engine"], record["protocol"]) for record in records}
    for cell, engine, protocol in expected:
        if (cell, engine, protocol) not in seen:
            records.append(
                {
                    "cell": cell,
                    "engine": engine,
                    "protocol": protocol,
                    "operation": "contains",
                    "seconds": "",
                    "result": "",
                    "prepare_seconds": "",
                    "status": "timeout",
                    "detail": "killed at the wall-clock cap",
                }
            )
    if not records:
        raise SystemExit(f"no rows under {rows_dir}")

    status = Counter(record["status"] for record in records)
    print(f"rows {len(records)}  status {dict(status)}")
    for engine, count in Counter(r["engine"] for r in records if r["status"] == "timeout").most_common():
        print(f"  timeout {engine}: {count}")
    for engine, count in Counter(r["engine"] for r in records if r["status"] == "error").most_common():
        print(f"  crash   {engine}: {count}")

    by_cell: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for record in records:
        by_cell[(record["cell"], record["protocol"])][record["engine"]] = record

    regrets: dict[tuple[str, str], list[float]] = defaultdict(list)
    censored: dict[tuple[str, str], list[float]] = defaultdict(list)
    timeouts: Counter = Counter()
    oracle_wins: Counter = Counter()
    mismatches = 0
    per_cell = []
    for (cell, protocol), engines in sorted(by_cell.items()):
        ok = {
            name: float(record["seconds"])
            for name, record in engines.items()
            if record["status"] == "ok" and name != "portfolio"
        }
        if not ok:
            continue
        oracle_engine = min(ok, key=ok.get)
        oracle_seconds = ok[oracle_engine]
        oracle_wins[(protocol, oracle_engine)] += 1
        answers = {record["result"] for record in engines.values() if record["status"] == "ok"}
        if len(answers) > 1:
            mismatches += 1
            print(
                f"MISMATCH {cell} [{protocol}] { {n: r['result'] for n, r in engines.items() if r['status'] == 'ok'} }"
            )
        for name, record in engines.items():
            if record["status"] == "ok":
                value = float(record["seconds"]) / oracle_seconds
                regrets[(protocol, name)].append(value)
                censored[(protocol, name)].append(value)
            elif record["status"] == "timeout":
                timeouts[(protocol, name)] += 1
                censored[(protocol, name)].append(timeout_cap / oracle_seconds)
        portfolio = engines.get("portfolio")
        portfolio_seconds = float(portfolio["seconds"]) if portfolio and portfolio["status"] == "ok" else None
        per_cell.append(
            {
                "cell": cell,
                "protocol": protocol,
                "oracle_engine": oracle_engine,
                "oracle_seconds": oracle_seconds,
                "portfolio_seconds": portfolio_seconds,
                "portfolio_regret": portfolio_seconds / oracle_seconds if portfolio_seconds is not None else "",
                "timeouts": ",".join(name for name, record in engines.items() if record["status"] == "timeout"),
                "crashes": ",".join(name for name, record in engines.items() if record["status"] == "error"),
            }
        )

    print(f"agreement mismatches: {mismatches}")
    for protocol in ("one-shot", "amortized"):
        print(f"\n== {protocol}: engine/oracle regret ==")
        print(f"{'engine':<10} {'ok':>4} {'TO':>4} {'med_ok':>7} {'mean_ok':>8} {'med_cens':>9} {'mean_cens':>10}")
        keys = {engine for (prot, engine) in censored if prot == protocol}
        for engine in sorted(keys, key=lambda name: statistics.median(censored[(protocol, name)])):
            ok_values = regrets[(protocol, engine)]
            cens_values = censored[(protocol, engine)]
            print(
                f"{engine:<10} {len(ok_values):>4} {timeouts[(protocol, engine)]:>4} "
                f"{statistics.median(ok_values) if ok_values else float('nan'):>7.2f} "
                f"{statistics.mean(ok_values) if ok_values else float('nan'):>8.2f} "
                f"{statistics.median(cens_values):>9.2f} {statistics.mean(cens_values):>10.2f}"
            )
        wins = ", ".join(f"{engine} {count}" for (prot, engine), count in oracle_wins.most_common() if prot == protocol)
        print(f"oracle wins: {wins}")

    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_cell[0]))
            writer.writeheader()
            writer.writerows(per_cell)
        print(f"\nwrote {path}")


if __name__ == "__main__":
    fire.Fire(main)
