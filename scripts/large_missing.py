"""Print jobs that produced no row (timed-out or killed), optionally excluding engines."""

from __future__ import annotations

from pathlib import Path

import fire


def main(jobs_file: str, rows_dir: str, exclude: str | tuple = "") -> None:
    if isinstance(exclude, str):
        excluded = {name for name in exclude.split(",") if name}
    else:
        excluded = {name for name in exclude if name}
    done = {path.name for path in Path(rows_dir).glob("*.csv")}
    for line in Path(jobs_file).read_text().splitlines():
        if not line.strip():
            continue
        cell, engine, protocol = line.split()
        if engine in excluded:
            continue
        if f"{cell}__{engine}__{protocol}.csv" not in done:
            print(line)


if __name__ == "__main__":
    fire.Fire(main)
