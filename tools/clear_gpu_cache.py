#!/usr/bin/env python3
"""
Utility to clear GPU memory caches for common Python frameworks.

- PyTorch: Releases cached blocks via torch.cuda.empty_cache() and triggers IPC garbage collection.
- CuPy (optional): Frees default device and pinned memory pools if CuPy is installed.
- TensorFlow (optional): Clears Keras backend session to help release GPU allocations.

This does not forcibly reset the GPU. It is safe to run within a Python process
that previously used GPU memory and wants to free as much as possible.

Usage examples:
  python tools/clear_gpu_cache.py                  # clear all devices using available frameworks
  python tools/clear_gpu_cache.py --devices 0,1    # target specific device indices
  python tools/clear_gpu_cache.py --no-cupy --no-tf
  python tools/clear_gpu_cache.py --report-json cache_report.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from typing import Dict, List, Optional


def _import_torch():
    try:
        import torch  # type: ignore

        return torch
    except Exception:
        return None


def _import_cupy():
    try:
        import cupy  # type: ignore

        return cupy
    except Exception:
        return None


def _import_tensorflow():
    try:
        import tensorflow as tf  # type: ignore

        return tf
    except Exception:
        return None


def bytes_to_mib(num_bytes: int) -> float:
    return float(num_bytes) / (1024.0 * 1024.0)


def get_torch_device_stats(torch_mod, device_index: int) -> Dict[str, int]:
    try:
        with torch_mod.cuda.device(device_index):
            reserved = torch_mod.cuda.memory_reserved()
            allocated = torch_mod.cuda.memory_allocated()
            inactive = 0
            try:
                # Newer PyTorch exposes inactive split by small/large
                inactive = torch_mod.cuda.memory_stats().get(
                    "inactive_split", 0
                )
            except Exception:
                pass
        return {
            "reserved_bytes": int(reserved),
            "allocated_bytes": int(allocated),
            "inactive_split_bytes": int(inactive),
        }
    except Exception:
        return {
            "reserved_bytes": 0,
            "allocated_bytes": 0,
            "inactive_split_bytes": 0,
        }


def clear_with_torch(torch_mod, device_index: int, synchronize: bool = True) -> None:
    if torch_mod is None or not torch_mod.cuda.is_available():
        return
    try:
        torch_mod.cuda.set_device(device_index)
        if synchronize:
            torch_mod.cuda.synchronize()
        torch_mod.cuda.empty_cache()
        # Give the driver a moment to reclaim pages
        if synchronize:
            torch_mod.cuda.synchronize()
        # Collect interprocess handles
        try:
            torch_mod.cuda.ipc_collect()
        except Exception:
            pass
    except Exception:
        # Best-effort; do not raise
        pass


def clear_with_cupy(cupy_mod, device_index: int) -> None:
    if cupy_mod is None:
        return
    try:
        with cupy_mod.cuda.Device(device_index):
            try:
                cupy_mod.cuda.runtime.deviceSynchronize()
            except Exception:
                pass
            try:
                pool = cupy_mod.get_default_memory_pool()
                pool.free_all_blocks()
            except Exception:
                pass
            try:
                pinned = cupy_mod.get_default_pinned_memory_pool()
                pinned.free_all_blocks()
            except Exception:
                pass
    except Exception:
        pass


def clear_with_tensorflow(tf_mod) -> None:
    if tf_mod is None:
        return
    try:
        # Clears the current TF graph and frees resources held by the session.
        tf_mod.keras.backend.clear_session()
        # Force Python GC to accelerate release of underlying buffers
        gc.collect()
    except Exception:
        pass


def parse_devices_arg(devices_arg: Optional[str], default_device_count: int) -> List[int]:
    if devices_arg is None or devices_arg.strip().lower() in ("all", "*"):
        return list(range(max(default_device_count, 0)))
    indices: List[int] = []
    for chunk in devices_arg.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            indices.append(int(chunk))
        except ValueError:
            raise SystemExit(f"Invalid device index: {chunk}")
    return indices


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Clear GPU caches for PyTorch, CuPy, and TensorFlow (best-effort).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="all",
        help="Comma-separated CUDA device indices to target, or 'all'",
    )
    parser.add_argument(
        "--no-torch",
        action="store_true",
        help="Disable PyTorch cache clearing",
    )
    parser.add_argument(
        "--no-cupy",
        action="store_true",
        help="Disable CuPy cache clearing",
    )
    parser.add_argument(
        "--no-tf",
        action="store_true",
        help="Disable TensorFlow cache clearing",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.1,
        help="Delay in seconds between framework clears per device",
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default=None,
        help="Optional path to write a JSON report of before/after stats",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-step logs",
    )
    args = parser.parse_args(argv)

    torch_mod = None if args.no_torch else _import_torch()
    cupy_mod = None if args.no_cupy else _import_cupy()
    tf_mod = None if args.no_tf else _import_tensorflow()

    device_count = 0
    if torch_mod is not None:
        try:
            device_count = torch_mod.cuda.device_count()
        except Exception:
            device_count = 0
    elif cupy_mod is not None:
        try:
            device_count = cupy_mod.cuda.runtime.getDeviceCount()
        except Exception:
            device_count = 0
    else:
        device_count = 0

    target_devices = parse_devices_arg(args.devices, device_count)

    report: Dict[str, Dict[str, Dict[str, int]]] = {}

    if len(target_devices) == 0:
        print("No CUDA devices detected or specified. Nothing to do.")
        # Still try to clear TF session to release host resources
        clear_with_tensorflow(tf_mod)
        return 0

    for device_index in target_devices:
        device_key = f"cuda:{device_index}"
        report[device_key] = {"before": {}, "after": {}}

        before_stats = (
            get_torch_device_stats(torch_mod, device_index) if torch_mod is not None else {}
        )
        report[device_key]["before"] = before_stats

        if args.verbose:
            if before_stats:
                print(
                    f"[{device_key}] Before - reserved: {bytes_to_mib(before_stats.get('reserved_bytes', 0)):.1f} MiB, "
                    f"allocated: {bytes_to_mib(before_stats.get('allocated_bytes', 0)):.1f} MiB"
                )
            else:
                print(f"[{device_key}] Before - no PyTorch stats available")

        if torch_mod is not None:
            clear_with_torch(torch_mod, device_index)
            time.sleep(args.sleep)

        if cupy_mod is not None:
            clear_with_cupy(cupy_mod, device_index)
            time.sleep(args.sleep)

        # TensorFlow is process-wide; call once per loop to encourage GC
        clear_with_tensorflow(tf_mod)
        time.sleep(args.sleep)

        after_stats = (
            get_torch_device_stats(torch_mod, device_index) if torch_mod is not None else {}
        )
        report[device_key]["after"] = after_stats

        if args.verbose:
            if after_stats:
                print(
                    f"[{device_key}] After  - reserved: {bytes_to_mib(after_stats.get('reserved_bytes', 0)):.1f} MiB, "
                    f"allocated: {bytes_to_mib(after_stats.get('allocated_bytes', 0)):.1f} MiB"
                )
            else:
                print(f"[{device_key}] After  - no PyTorch stats available")

    if args.report_json:
        try:
            with open(args.report_json, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        except Exception as exc:
            print(f"Failed to write report to {args.report_json}: {exc}")

    # Human-readable summary if not verbose
    if not args.verbose and torch_mod is not None:
        for device_index in target_devices:
            device_key = f"cuda:{device_index}"
            b = report[device_key]["before"]
            a = report[device_key]["after"]
            if b and a:
                print(
                    f"{device_key}: reserved {bytes_to_mib(b.get('reserved_bytes', 0)):.1f} → {bytes_to_mib(a.get('reserved_bytes', 0)):.1f} MiB, "
                    f"allocated {bytes_to_mib(b.get('allocated_bytes', 0)):.1f} → {bytes_to_mib(a.get('allocated_bytes', 0)):.1f} MiB"
                )
            else:
                print(f"{device_key}: cleared (no PyTorch stats available)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

