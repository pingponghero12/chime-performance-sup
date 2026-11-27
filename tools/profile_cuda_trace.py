"""Summarize first upload and propagation from a CHIME Nsight SQLite export."""

import argparse
import json
import sqlite3
from pathlib import Path


def first_range(db, name):
    row = db.execute(
        """
        SELECT n.start, n.end
        FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id = n.textId
        WHERE COALESCE(n.text, s.value) = ? AND n.end IS NOT NULL
        ORDER BY n.start LIMIT 1
        """,
        (name,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Missing NVTX range: {name}")
    return row


def first_upload(db):
    start, end = first_range(db, "transfer_api/image_to_device")
    netcdf = db.execute(
        """
        SELECT SUM(n.end - n.start)
        FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id = n.textId
        WHERE n.start >= ? AND n.end <= ? AND COALESCE(n.text, s.value) = 'netcdf/read'
        """,
        (start, end),
    ).fetchone()[0]
    allocations = dict(db.execute(
        """
        SELECT substr(s.value, 1, instr(s.value, '_v') - 1),
               SUM(r.end - r.start) / 1e6
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id = r.nameId
        WHERE r.start >= ? AND r.start < ?
              AND (s.value GLOB 'cudaHostAlloc_v*' OR s.value GLOB 'cudaMalloc_v*')
        GROUP BY 1
        """,
        (start, end),
    ))

    # Match by API call so an asynchronous copy is counted even if it finishes
    # after the upload wrapper returns.
    size, copied = db.execute(
        """
        SELECT SUM(m.bytes), SUM(m.end - m.start)
        FROM CUPTI_ACTIVITY_KIND_MEMCPY m
        WHERE m.copyKind = 1 AND EXISTS (
            SELECT 1 FROM CUPTI_ACTIVITY_KIND_RUNTIME r
            WHERE r.correlationId = m.correlationId AND r.start >= ? AND r.start < ?
        )
        """,
        (start, end),
    ).fetchone()
    return {
        "wall_ms": (end - start) / 1e6,
        "netcdf_read_ms": (netcdf or 0) / 1e6,
        "pinned_host_allocation_api_ms": allocations.get("cudaHostAlloc", 0.0),
        "device_allocation_api_ms": allocations.get("cudaMalloc", 0.0),
        "host_to_device_ms": (copied or 0) / 1e6,
        "host_to_device_bytes": size or 0,
    }


def propagation(db):
    start, end = first_range(db, "operation/propagate")
    kernels = [
        {"name": name, "ms": duration}
        for name, duration in db.execute(
            """
            SELECT s.value, (k.end - k.start) / 1e6
            FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.shortName
            WHERE k.start >= ? AND k.start < ? ORDER BY k.start
            """,
            (start, end),
        )
    ]
    return {
        "wall_ms": (end - start) / 1e6,
        "kernel_ms": sum(kernel["ms"] for kernel in kernels),
        "kernels": kernels,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    with sqlite3.connect(f"{args.trace.resolve().as_uri()}?mode=ro", uri=True) as db:
        report = {"first_upload": first_upload(db), "propagation": propagation(db)}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
