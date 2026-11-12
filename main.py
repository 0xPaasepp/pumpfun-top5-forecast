#!/usr/bin/env python3
"""pumpfun-top5-forecast-01: Monte Carlo log-cap forecaster.

Reads /input/task.json + /input/history.ndjson (10-minute marks of pump.fun
coins launched since t0, ranked by market cap) and writes
/output/forecast.json: for each of the 5 places, a probability for each coin
named being at that place, the guessed top 5 in order, and for each place a
probability for each log10(SOL) market-cap bin.

Model
-----
Each coin's log10 market cap (in SOL) is modelled as a Brownian motion with
drift (per hour) and volatility (per sqrt-hour), estimated from its own recent
trajectory and shrunk toward global priors (young / noisy coins shrink more).
Graduated (pool, complete) coins are steadier than bonding-curve coins.  A
fade hazard lets weak curve coins collapse toward a low cap, which is how
most pump.fun coins actually die.  We Monte-Carlo the trajectories (fixed
seed, so two runs on the same history agree) forward to each answer read,
rank the simulated caps, and read off rank probabilities per place and the
size-bin distribution per place.  A slice of each place is left unnamed for
newcomers: coins born after the cut that no history can name.

Python 3.13, standard library only. Deterministic. No network.
"""

import json
import math
import os
import random
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# --- tunables (fixed in advance, chosen from the dev rehearsal's shape) ---
SEED = 20261112
N_PATHS = 4000
MAX_CANDIDATES = 120          # coins simulated (top of last mark + recent risers)
MAX_ROWS_PER_PLACE = 20       # named coins per place in the output
MIN_P = 0.004                 # smallest named probability (below this, leave unnamed)
MAX_P = 0.55                  # cap on any single named probability
DRIFT_SHRINK_K = 3.0          # pseudo-observations pulling a coin's drift to the prior
VOL_SHRINK_K = 4.0
VOL_FLOOR = 0.10              # log10 per sqrt-hour
VOL_CAP = 0.80
GRAD_VOL_MULT = 0.75          # graduated coins are steadier
GRAD_DRIFT_PULL = 0.5         # graduated drift pulled halfway to 0
SIZE_WIDEN = 1.55             # widen simulated size sd for model error
FADE_FLOOR_LOGCAP = 1.0       # a faded coin collapses toward log10 cap ~1.0
DRIFT_TAU_HOURS = 3.0         # a hot streak mean-reverts over ~3h (drift decays)
SIZE_GROWTH_PER_SQRT_H = 0.06  # place caps inflate as newcomers arrive


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def logcap_sol(market_cap_lamports):
    """log10 of a market cap in SOL: log10(lamports) - 9."""
    return math.log10(int(market_cap_lamports)) - 9.0


def load_history(input_dir):
    with open(os.path.join(input_dir, "task.json")) as f:
        task = json.load(f)
    marks = []
    with open(os.path.join(input_dir, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") is True and isinstance(m.get("rows"), list):
                marks.append(m)
    marks.sort(key=lambda m: m["at"])
    return task, marks


def build_trajectories(marks, t0):
    """mint -> list of (hours_since_t0, logcap, rank, venue, complete)."""
    traj = {}
    for m in marks:
        t = (parse_time(m["at"]) - t0).total_seconds() / 3600.0
        for r in m["rows"]:
            try:
                lc = logcap_sol(r["market_cap_lamports"])
            except (KeyError, TypeError, ValueError):
                continue
            traj.setdefault(r["mint"], []).append(
                (t, lc, r.get("rank", 999), r.get("venue", ""),
                 bool(r.get("complete", False)), r.get("created_at", ""))
            )
    for v in traj.values():
        v.sort(key=lambda x: x[0])
    return traj


def estimate_params(traj, last_time):
    """Per-coin (mu per hour, sigma per sqrt-hour, fade_per_hour, last_logcap).

    mu/sigma from the coin's own log returns, shrunk to global priors.
    fade_per_hour: hazard that a weak (curve, falling) coin collapses.
    """
    # global priors from all consecutive 10-minute log returns
    all_rets = []
    for v in traj.values():
        for a, b in zip(v, v[1:]):
            dt = b[0] - a[0]
            if 0.05 < dt < 0.5:
                all_rets.append((b[1] - a[1]) / dt)  # per hour
    if all_rets:
        mu0 = sum(all_rets) / len(all_rets)
        var0 = sum((x - mu0) ** 2 for x in all_rets) / max(1, len(all_rets) - 1)
        sig0 = math.sqrt(max(var0, 0.01))
    else:
        mu0, sig0 = -0.05, 0.30
    # returns are per-hour drifts; sigma per sqrt-hour ~ sd of hourly returns
    # (returns over ~1/6h scaled): use sd of per-hour returns as sigma directly.
    params = {}
    for mint, v in traj.items():
        # recent window: last 6 points at most, weight recent more
        pts = v[-6:]
        rets = []
        for a, b in zip(pts, pts[1:]):
            dt = b[0] - a[0]
            if 0.05 < dt < 0.5:
                rets.append((b[1] - a[1]) / dt)
        n = len(rets)
        if n:
            mu_obs = sum(rets) / n
            var_obs = sum((x - mu_obs) ** 2 for x in rets) / n if n > 1 else sig0 * sig0
        else:
            mu_obs, var_obs = mu0, sig0 * sig0
        mu = (n * mu_obs + DRIFT_SHRINK_K * mu0) / (n + DRIFT_SHRINK_K)
        var = (n * var_obs + VOL_SHRINK_K * sig0 * sig0) / (n + VOL_SHRINK_K)
        sigma = math.sqrt(max(var, VOL_FLOOR * VOL_FLOOR))
        sigma = min(sigma, VOL_CAP)
        last = v[-1]
        last_lc, venue, complete = last[1], last[3], last[4]
        # graduated coins: steadier, drift pulled toward 0 (they persist)
        if complete or venue == "pool":
            sigma *= GRAD_VOL_MULT
            mu = mu * (1.0 - GRAD_DRIFT_PULL) + 0.0 * GRAD_DRIFT_PULL
        # fade hazard: curve coins with negative recent drift die fast;
        # graduated coins rarely collapse outright.
        if complete or venue == "pool":
            fade = 0.02
        else:
            # per hour: base 0.10, higher when the coin is falling
            fade = 0.10 + max(0.0, -mu) * 0.6
            fade = min(fade, 0.60)
        # cap drift: a coin cannot be expected to compound forever; a very
        # hot recent run mean-reverts.  Shrink extreme positive drift.
        mu = max(mu, -1.5)
        mu = min(mu, 0.9)
        params[mint] = (mu, sigma, fade, last_lc, complete)
    return params, mu0, sig0


def newcomer_share(horizon_hours):
    """Probability a place is taken by a coin no history can name.

    Grows with horizon: in the rehearsal ~1/5 of top-5 slots at a 10-20 min
    horizon were newcomers, more over longer horizons while the market is
    young.  Capped so the named coins still carry most of the mass.
    """
    # 0.30 at ~0.3h, rising ~0.02 per hour, capped at 0.50.  The practice
    # shows ~1/3 of top-5 slots going to coins born after the cut.
    return min(0.50, 0.30 + 0.02 * max(0.0, horizon_hours - 0.3))


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def forecast(task, marks):
    t0 = parse_time(task["t0"])
    places = int(task.get("places", 5))
    edges = task["size"]["edges"]
    n_bins = len(edges) + 1  # 82
    reads = [parse_time(r) for r in task["reads"]]
    last_mark_time = parse_time(marks[-1]["at"])
    # horizon per read in hours from the last mark
    horizons = [(r - last_mark_time).total_seconds() / 3600.0 for r in reads]
    mean_horizon = sum(horizons) / len(horizons)

    traj = build_trajectories(marks, t0)
    params, mu0, sig0 = estimate_params(traj, last_mark_time)

    # candidate set: top of last mark + best recent risers
    last_rows = sorted(marks[-1]["rows"], key=lambda r: r.get("rank", 999))
    cand = []
    seen = set()
    for r in last_rows[:MAX_CANDIDATES]:
        cand.append(r["mint"])
        seen.add(r["mint"])
    # risers: coins whose logcap rose most over the history and are near top
    risers = []
    for mint, v in traj.items():
        if mint in seen or len(v) < 2:
            continue
        gain = v[-1][1] - v[0][1]
        risers.append((gain, mint))
    risers.sort(reverse=True)
    for _, mint in risers[:20]:
        if len(cand) >= MAX_CANDIDATES:
            break
        cand.append(mint)
        seen.add(mint)
    cand = [m for m in cand if m in params]
    if len(cand) < places:
        raise ValueError("too few candidate coins")

    # Monte Carlo
    rng = random.Random(SEED)
    # rank_counts[place][mint] and size_hist[place][bin]
    rank_counts = [dict() for _ in range(places)]
    size_hist = [[0.0] * n_bins for _ in range(places)]
    total_draws = 0
    # pre-extract per-coin arrays
    mus = [params[m][0] for m in cand]
    sigs = [params[m][1] for m in cand]
    fades = [params[m][2] for m in cand]
    last_lc = [params[m][3] for m in cand]
    n_cand = len(cand)

    for _ in range(N_PATHS):
        # simulate each coin's logcap at each read (incremental Brownian steps)
        # cur[j] = current logcap of coin j; t_prev = hours since last mark
        cur = list(last_lc)
        alive = [True] * n_cand
        t_prev = 0.0
        for h in horizons:
            dt = h - t_prev
            t_prev = h
            if dt <= 0:
                dt = 0.0
            sq = math.sqrt(dt) if dt > 0 else 0.0
            for j in range(n_cand):
                if not alive[j]:
                    continue
                if dt > 0:
                    # fade: collapse toward the floor with hazard fade*dt
                    if rng.random() < fades[j] * dt:
                        alive[j] = False
                        cur[j] = FADE_FLOOR_LOGCAP
                        continue
                    # mean-reverting drift: mu decays as exp(-t/tau); the
                    # drift over [t_prev, h] is mu*tau*(exp(-t_prev/tau)
                    # - exp(-h/tau)), so a hot streak saturates instead of
                    # compounding forever over a long horizon.
                    drift = mus[j] * DRIFT_TAU_HOURS * (
                        math.exp(-t_prev / DRIFT_TAU_HOURS)
                        - math.exp(-h / DRIFT_TAU_HOURS)
                    )
                    cur[j] = cur[j] + drift + sigs[j] * sq * rng.gauss(0.0, 1.0)
            # rank this read
            order = sorted(range(n_cand), key=lambda j: cur[j], reverse=True)
            for k in range(places):
                j = order[k]
                mint = cand[j]
                rank_counts[k][mint] = rank_counts[k].get(mint, 0) + 1
                lc = cur[j]
                # size bin of the cap at place k
                b = bin_of(lc, edges)
                size_hist[k][b] += 1.0
            total_draws += 1

    # newcomer mass per place (shared, from the mean horizon)
    newcomer = newcomer_share(mean_horizon)

    probabilities = []
    for k in range(places):
        counts = rank_counts[k]
        # top coins by simulated frequency
        ranked = sorted(counts.items(), key=lambda kv: -kv[1])
        named = []
        for mint, c in ranked[:MAX_ROWS_PER_PLACE]:
            p = c / total_draws
            if p < MIN_P:
                continue
            named.append((mint, p))
        # scale named so they sum to (1 - newcomer); cap any single p
        s = sum(p for _, p in named)
        if s <= 0:
            continue
        scale = (1.0 - newcomer) / s
        for mint, p in named:
            p2 = min(p * scale, MAX_P)
            probabilities.append({"mint": mint, "place": k + 1, "p": round(p2, 6)})
    # top5: the most likely coin at each place (by simulated frequency)
    top5 = []
    used = set()
    for k in range(places):
        counts = rank_counts[k]
        best = None
        for mint, c in sorted(counts.items(), key=lambda kv: -kv[1]):
            if mint not in used:
                best = mint
                break
        if best is None and counts:
            best = max(counts, key=counts.get)
        top5.append(best)
        used.add(best)
    # sizes: widen the simulated histogram a little (model error), then
    # convert counts to probabilities; leave the rest unnamed.
    sizes = []
    for k in range(places):
        hist = size_hist[k]
        tot = sum(hist)
        if tot <= 0:
            continue
        # widen: mix the histogram with a discrete normal around its mean
        mean0 = sum(i * hist[i] for i in range(n_bins)) / tot
        var = sum((i - mean0) ** 2 * hist[i] for i in range(n_bins)) / tot
        # place caps inflate as newcomers arrive; shift the mean up with
        # horizon (the Monte Carlo only simulates coins already known).
        mean = mean0 + (SIZE_GROWTH_PER_SQRT_H * math.sqrt(max(mean_horizon, 0.0))) / 0.05
        sd = math.sqrt(max(var, 0.25)) * SIZE_WIDEN
        probs = []
        for b in range(n_bins):
            lo = -1e9 if b == 0 else edges[b - 1]
            hi = 1e9 if b == n_bins - 1 else edges[b]
            # bin centre in log10 SOL
            centre = (lo + hi) / 2.0 if abs(lo) < 1e8 and abs(hi) < 1e8 else (lo if abs(hi) >= 1e8 else hi)
            # histogram share
            h = hist[b] / tot
            # normal share around the simulated mean (in bin units)
            z_lo = ((lo if abs(lo) < 1e8 else (mean - 6 * sd)) - mean) / sd if sd > 0 else 0.0
            z_hi = ((hi if abs(hi) < 1e8 else (mean + 6 * sd)) - mean) / sd if sd > 0 else 0.0
            # use bin index units: convert edges to bin coordinates
            # simpler: normal in bin-index space, mean=mean(bin idx), sd=sd
            blo = b - 0.5
            bhi = b + 0.5
            nrm = norm_cdf((bhi - mean) / sd) - norm_cdf((blo - mean) / sd) if sd > 0 else 0.0
            p = 0.6 * h + 0.4 * nrm
            probs.append(p)
        s = sum(probs)
        if s <= 0:
            continue
        for b, p in enumerate(probs):
            p2 = p / s
            # keep the total named <= 0.98 so a little stays unnamed
            p2 *= 0.98
            if p2 >= 0.0005:
                sizes.append({"place": k + 1, "bin": b, "p": round(p2, 6)})
    return {"probabilities": probabilities, "top5": top5, "sizes": sizes}


def bin_of(lc, edges):
    """Bin index (0..len(edges)) of a log10 cap: bin 0 < edges[0], bin i in
    [edges[i-1], edges[i]), last bin >= edges[-1]."""
    if lc < edges[0]:
        return 0
    lo, hi = 0, len(edges)
    while lo < hi:
        mid = (lo + hi) // 2
        if lc < edges[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


def main():
    task, marks = load_history(INPUT_DIR)
    if not marks:
        raise SystemExit("the history holds no ok mark")
    out = forecast(task, marks)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, task.get("output", "forecast.json"))
    with open(path, "w") as f:
        f.write(json.dumps(out) + "\n")
    print(f"wrote {path}: {len(out['probabilities'])} prob rows, "
          f"top5={out['top5']}, {len(out['sizes'])} size rows")


if __name__ == "__main__":
    main()
