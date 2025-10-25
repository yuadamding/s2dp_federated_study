# python generate_setting2.py -n 20000 --out setting2_raw --shard-size 100 --num-cpus-per-task 1
import os, math
from pathlib import Path
import argparse
import numpy as np
import neurokit2 as nk
import ray

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

def _sim_with_intlen(sim_fn, *, duration_i, sr, length_i=None, **kwargs):
    """
    Try duration as an INT first (fast path). If the simulator still complains,
    fall back to passing an explicit integer length.
    """
    try:
        return sim_fn(duration=duration_i, sampling_rate=sr, **kwargs)
    except Exception:
        Li = int(length_i if length_i is not None else duration_i * sr)
        return sim_fn(length=Li, sampling_rate=sr, **kwargs)

def _generate_one_inner(i, duration, seed_base, out_dir, max_sr, floor_sr,
                        base_ti, base_ai, base_bi):
    rng = np.random.default_rng(seed_base + i)

    # sampling rate ~ LogNormal(ln 100, 1), floored like your script
    sr = max(int(rng.lognormal(np.log(100.0), 1.0)), floor_sr)
    if max_sr is not None:
        sr = min(sr, max_sr)

    # spiky morphology with noise
    ti = rng.normal(base_ti, np.abs(base_ti / 5.0))
    ai = rng.normal(base_ai, np.abs(base_ai / 5.0))
    bi = rng.normal(base_bi, np.abs(base_bi / 5.0))

    # ensure integer duration and precompute int length
    duration_i = int(round(duration))               # <- key fix
    length_i   = int(max(1, sr * duration_i))      # <- fallback length

    prefix = Path(out_dir) / f"output_{i}_"

    # 12-lead ECG (DataFrame) + single-lead signals (1D arrays)
    ecg12 = _sim_with_intlen(
        nk.ecg_simulate,
        duration_i=duration_i, sr=sr, length_i=length_i,
        method="multileads", ti=ti, ai=ai, bi=bi
    )
    ecg12.to_csv(str(prefix) + "ecg12.csv", index=False)

    ecg = _sim_with_intlen(nk.ecg_simulate, duration_i=duration_i, sr=sr, length_i=length_i,
                           heart_rate=70)
    np.savetxt(str(prefix) + "ecg.csv", ecg, delimiter=",")

    ppg = _sim_with_intlen(nk.ppg_simulate, duration_i=duration_i, sr=sr, length_i=length_i,
                           heart_rate=70)
    np.savetxt(str(prefix) + "ppg.csv", ppg, delimiter=",")

    rsp = _sim_with_intlen(nk.rsp_simulate, duration_i=duration_i, sr=sr, length_i=length_i,
                           respiratory_rate=15)
    np.savetxt(str(prefix) + "rsp.csv", rsp, delimiter=",")

    eda = _sim_with_intlen(nk.eda_simulate, duration_i=duration_i, sr=sr, length_i=length_i,
                           scr_number=3)
    np.savetxt(str(prefix) + "eda.csv", eda, delimiter=",")

    emg = _sim_with_intlen(nk.emg_simulate, duration_i=duration_i, sr=sr, length_i=length_i,
                           burst_number=2)
    np.savetxt(str(prefix) + "emg.csv", emg, delimiter=",")

    return i, sr

@ray.remote(max_retries=1)
def generate_one(i, duration, seed_base, out_dir, max_sr, floor_sr,
                 base_ti, base_ai, base_bi):
    return _generate_one_inner(i, duration, seed_base, out_dir, max_sr, floor_sr,
                               base_ti, base_ai, base_bi)

@ray.remote(max_retries=1)
def generate_shard(start, stop, duration, seed_base, out_dir, max_sr, floor_sr,
                   base_ti, base_ai, base_bi):
    out = []
    for i in range(start, stop):
        out.append(_generate_one_inner(i, duration, seed_base, out_dir, max_sr, floor_sr,
                                       base_ti, base_ai, base_bi))
    return out

def main():
    ap = argparse.ArgumentParser(description="Ray-based generator for Setting 2 (NeuroKit2).")
    ap.add_argument("-n", "--n-samples", type=int, default=20000)
    ap.add_argument("--out", type=str, default="setting2_raw")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--address", type=str, default=None, help='Ray address (e.g., "auto" for cluster).')
    ap.add_argument("--local-mode", action="store_true", help="Ray local_mode for easier debugging.")
    ap.add_argument("--max-sr", type=int, default=None, help="Optional cap on sampling rate (e.g., 2000).")
    ap.add_argument("--shard-size", type=int, default=100,
                    help=">1 to batch indices per task for lower overhead (e.g., 50 or 100).")
    ap.add_argument("--num-cpus-per-task", type=float, default=1.0, help="Ray CPUs reserved per task.")
    args = ap.parse_args()

    ray.init(address=args.address or None, local_mode=args.local_mode,
             runtime_env=None, ignore_reinit_error=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # same floor rule as original
    floor_sr = int(math.ceil(100.0 / args.duration))

    # base morphology (same as your code)
    base_ti = np.array([-70, -15,   0,  15, 100], dtype=float)
    base_ai = np.array([ 1.2,  -5,  30, -7.5, 0.75], dtype=float)
    base_bi = np.array([0.25, 0.10, 0.10, 0.10, 0.40], dtype=float)

    # submit tasks
    refs = []
    if args.shard_size > 1:
        for start in range(0, args.n_samples, args.shard_size):
            stop = min(args.n_samples, start + args.shard_size)
            refs.append(
                generate_shard.options(num_cpus=args.num_cpus_per_task).remote(
                    start, stop, args.duration, args.seed, str(out_dir),
                    args.max_sr, floor_sr, base_ti, base_ai, base_bi
                )
            )
        remaining = set(refs)
        pbar = tqdm(total=args.n_samples, desc="Generating") if tqdm else None
        while remaining:
            done, remaining = ray.wait(list(remaining), num_returns=1, timeout=None)
            res_list = ray.get(done[0])  # list of (i, sr)
            if pbar: pbar.update(len(res_list))
        if pbar: pbar.close()
    else:
        for i in range(args.n_samples):
            refs.append(
                generate_one.options(num_cpus=args.num_cpus_per_task).remote(
                    i, args.duration, args.seed, str(out_dir),
                    args.max_sr, floor_sr, base_ti, base_ai, base_bi
                )
            )
        remaining = set(refs)
        pbar = tqdm(total=args.n_samples, desc="Generating") if tqdm else None
        while remaining:
            done, remaining = ray.wait(list(remaining), num_returns=min(512, len(remaining)))
            _ = ray.get(done)
            if pbar: pbar.update(len(done))
        if pbar: pbar.close()

    print("Done.")

if __name__ == "__main__":
    main()
