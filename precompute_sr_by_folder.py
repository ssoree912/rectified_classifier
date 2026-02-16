import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SR cache precompute by generator/class folders (parallel)."
    )
    parser.add_argument("--input_root", type=str, required=True, help="Root containing generator folders")
    parser.add_argument("--output_root", type=str, required=True, help="Root to store SR cache")
    parser.add_argument("--phase", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--classes", nargs="+", default=["ai", "nature"])
    parser.add_argument(
        "--generators",
        nargs="+",
        default=None,
        help="Optional generator folder names. If omitted, auto-scan under input_root.",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="cuda",
        help='Comma-separated devices, e.g. "cuda:0,cuda:1" or "cpu"',
    )
    parser.add_argument("--max_parallel", type=int, default=None, help="Max concurrent jobs")
    parser.add_argument("--sr_model_name", type=str, default="RealESRGAN_x4plus")
    parser.add_argument("--sr_scale", type=int, default=4)
    parser.add_argument("--sr_tile", type=int, default=512)
    parser.add_argument("--save_ext", type=str, default="keep", choices=["keep", "png", "jpg"])
    parser.add_argument("--jpg_quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def discover_generators(input_root: Path, phase: str):
    out = []
    for d in sorted(input_root.iterdir()):
        if not d.is_dir():
            continue
        if (d / phase).is_dir():
            out.append(d.name)
    return out


def build_jobs(args):
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    generators = args.generators or discover_generators(input_root, args.phase)
    jobs = []
    for gen in generators:
        for cls in args.classes:
            in_dir = input_root / gen / args.phase / cls
            if not in_dir.is_dir():
                continue
            out_dir = output_root / gen / args.phase / cls
            jobs.append((gen, cls, in_dir, out_dir))
    return jobs


def run_jobs(args, jobs):
    script = Path(__file__).resolve().parent / "precompute_sr_cache.py"
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    if not devices:
        devices = ["cuda"]
    max_parallel = args.max_parallel or len(devices)
    max_parallel = max(1, max_parallel)

    print(f"[Launcher] jobs={len(jobs)} devices={devices} max_parallel={max_parallel}")
    for i, (gen, cls, in_dir, out_dir) in enumerate(jobs, start=1):
        print(f"[{i}] {gen}/{args.phase}/{cls} -> {out_dir}")

    if args.dry_run or not jobs:
        return

    running = []
    next_idx = 0
    job_idx = 0

    while job_idx < len(jobs) or running:
        while job_idx < len(jobs) and len(running) < max_parallel:
            gen, cls, in_dir, out_dir = jobs[job_idx]
            dev = devices[next_idx % len(devices)]
            next_idx += 1

            cmd = [
                sys.executable,
                str(script),
                "--input_root",
                str(in_dir),
                "--output_root",
                str(out_dir),
                "--device",
                dev,
                "--sr_model_name",
                args.sr_model_name,
                "--sr_scale",
                str(args.sr_scale),
                "--sr_tile",
                str(args.sr_tile),
                "--save_ext",
                args.save_ext,
                "--jpg_quality",
                str(args.jpg_quality),
                "--log_every",
                str(args.log_every),
            ]
            if args.overwrite:
                cmd.append("--overwrite")
            if args.max_items is not None:
                cmd.extend(["--max_items", str(args.max_items)])

            print(f"[Start] ({job_idx + 1}/{len(jobs)}) device={dev} target={gen}/{args.phase}/{cls}")
            proc = subprocess.Popen(cmd)
            running.append((job_idx + 1, gen, cls, dev, proc))
            job_idx += 1

        still_running = []
        for idx, gen, cls, dev, proc in running:
            ret = proc.poll()
            if ret is None:
                still_running.append((idx, gen, cls, dev, proc))
            else:
                status = "OK" if ret == 0 else f"FAIL({ret})"
                print(f"[Done] ({idx}/{len(jobs)}) {gen}/{args.phase}/{cls} on {dev}: {status}")
        running = still_running
        time.sleep(1)


def main():
    args = parse_args()
    jobs = build_jobs(args)
    run_jobs(args, jobs)


if __name__ == "__main__":
    main()
