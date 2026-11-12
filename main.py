#!/usr/bin/env python3
"""pumpfun-top5-forecast-01: Monte Carlo log-cap forecaster (v2).

Reads /input/task.json + /input/history.ndjson (10-minute marks of pump.fun
coins launched since t0, ranked by market cap) and writes
/output/forecast.json: for each of the 5 places, a probability for each coin
named being at that place, the guessed top 5 in order, and for each place a
probability for each log10(SOL) market-cap bin.

Model (v2 changes vs v1)
------------------------
- Churn-adaptive newcomer mass: the share left unnamed per place starts at
  the market's own churn -- the fraction of the top-5 that turned over in
  the MOST RECENT 10-minute transition only (churn can collapse in one
  mark as a market settles, so averaging over stale transitions badly
  overstates it) -- and grows with the answer horizon, so a fast-churning
  or long-horizon forecast honestly leaves more mass for coins no history
  can name.  v1 left a fixed ~0.30, far too little at long horizons where
  most top-5 slots turn over.
- Size widening grows with horizon and with thin history: the simulated
  size distribution is widened more the further out (and the fewer marks)
  the forecast runs, because model error grows with both.
- Thin-history coins fall back to the (negative) global prior drift with
  only a small rank-momentum nudge: a coin that just rocketed up usually
  cools rather than compounding its entry sprint, so the age-implied
  growth rate (tried first) overestimated such coins and was reverted.
- Volatility: higher floor and less shrinkage, so young coins are not
  assumed steadier than they are; thin-history coins get a higher vol.
- Temperature scaling (T=1.15) softens the Monte Carlo rank frequencies:
  the log score punishes overconfidence, so the raw frequencies are
  slightly flattened before they are written out.
- 8000 Monte Carlo paths (up from 4000) for lower simulation noise.

v3 changes vs v2
----------------
- Newcomer size mixture (the big fix): v2's size distribution was purely
  the simulated incumbent coins, so at young/high-churn cuts it put most
  mass on tiny caps for places 2-5 while the real answer was set by
  newcomers arriving with much larger caps (observed newcomer jumps
  +0.3 to +0.9 log10 per 10 min, biggest at the lower places).  Each
  place's size distribution is now a mixture: (1 - newcomer_share) of the
  simulated incumbent distribution plus newcomer_share of a newcomer
  distribution -- a wide normal centred above the current place-k cap
  (a newcomer that displaces the incumbent is bigger than it), the
  premium growing with the horizon and larger at lower places.  This
  directly attacks v2's worst split (cut 20:10 -> 20:20, size -15.1).

Each coin's log10 market cap (in SOL) is a Brownian motion with drift
(per hour) and volatility (per sqrt-hour), shrunk toward global priors.
Graduated (pool, complete) coins are steadier.  A fade hazard lets weak
curve coins collapse toward a low cap, which is how most pump.fun coins
die.  Drift mean-reverts over ~3h so a hot streak saturates instead of
compounding forever over a long horizon.

Python 3.13, standard library only. Deterministic. No network.
"""

import json
import math
import os
import random
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# --- tunables ---
SEED = 20261112
N_PATHS = 8000
MAX_CANDIDATES = 120          # coins simulated (top of last mark + recent risers)
MAX_ROWS_PER_PLACE = 20       # named coins per place in the output
MIN_P = 0.004                 # smallest named probability (below this, leave unnamed)
MAX_P = 0.55                  # cap on any single named probability
DRIFT_SHRINK_K = 3.0          # pseudo-observations pulling a coin's drift to the prior
VOL_SHRINK_K = 2.0            # (v2: less shrinkage, a coin's own vol matters more)
VOL_FLOOR = 0.15              # log10 per sqrt-hour (v2: higher floor)
VOL_CAP = 0.90
VOL_THIN = 0.40               # vol for coins with <2 observations
GRAD_VOL_MULT = 0.75          # graduated coins are steadier
GRAD_DRIFT_PULL = 0.5         # graduated drift pulled halfway to 0
FADE_FLOOR_LOGCAP = 1.0       # a faded coin collapses toward log10 cap ~1.0
DRIFT_TAU_HOURS = 3.0         # a hot streak mean-reverts over ~3h
SIZE_GROWTH_PER_SQRT_H = 0.06  # place caps inflate as newcomers arrive
TEMP = 1.15                  # temperature softening of rank frequencies
NEWCOMER_CAP = 0.65           # max unnamed share per place
NEWCOMER_SLOPE = 0.30         # unnamed share growth per hour of horizon
NEWCOMER_PREM = 0.40          # (v3) log10 jump when a newcomer takes a place (per 10 min)
NEWCOMER_PREM_CAP = 0.90       # (v3) max newcomer premium (log10)
NEWCOMER_PREM_SD = 0.40       # (v3) sd of the newcomer-cap distribution (log10)
START_LOGCAP = -1.0           # log10 SOL a coin effectively starts from


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
    """mint -> list of (hours_since_t0, logcap, rank, venue, complete, created_at)."""
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


def top5_churn(marks):
    """Fraction of the top-5 that turns over per 10-minute mark, averaged
    over the last two transitions, clamped to [0.15, 0.60].  This is the
    market's own churn rate; a fast-churning market needs more unnamed
    mass because more top-5 slots will go to coins no history can name."""
    if len(marks) < 2:
        return 0.30
    # Use ONLY the most recent transition: churn can collapse in one mark
    # (the rehearsal goes 0.80 -> 0.20 in a single step as the market
    # settles), and averaging over stale high-churn transitions badly
    # overstates the current churn.  The last transition is the best
    # predictor of near-future turnover.
    prev = {r["mint"] for r in marks[-2]["rows"] if r.get("rank", 999) <= 5}
    cur = {r["mint"] for r in marks[-1]["rows"] if r.get("rank", 999) <= 5}
    if len(cur) == 0:
        return 0.30
    churn = len(cur - prev) / 5.0
    return max(0.15, min(0.60, churn))


def newcomer_share(churn, horizon_hours):
    """Probability a place is taken by a coin no history can name.

    Starts at the market's own churn rate and grows with the answer
    horizon (more time = more births and more turnover), capped so the
    named coins still carry real mass.
    """
    return min(NEWCOMER_CAP, churn + NEWCOMER_SLOPE * max(0.0, horizon_hours))


def age_adjusted_churn(churn, age_hours):
    """The churn of the most recent transition is a noisy, possibly stale
    signal: a market's gold-rush churn collapses as it ages (the rehearsal
    goes 0.80 -> 0.20 within ~30-40 minutes), so an old market's churn
    estimate is pulled toward a low floor.  Without this, a cut taken
    just before the market settles overstates near-future turnover and
    the newcomer mass (order and size) is far too aggressive.
    """
    w = math.exp(-max(age_hours, 0.0) / 0.5)
    return churn * w + 0.15 * (1.0 - w)


def estimate_params(traj, last_time, t0):
    """Per-coin (mu per hour, sigma per sqrt-hour, fade_per_hour, last_logcap).

    mu/sigma from the coin's own log returns, shrunk to global priors.
    Coins with <3 observations get a rank-momentum / age-implied drift
    instead of the (negative) global prior: a coin that just rocketed
    into the top ranks is hot, not fading.
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
    params = {}
    for mint, v in traj.items():
        pts = v[-6:]
        rets = []
        for a, b in zip(pts, pts[1:]):
            dt = b[0] - a[0]
            if 0.05 < dt < 0.5:
                rets.append((b[1] - a[1]) / dt)
        n = len(rets)
        last = v[-1]
        last_lc, last_rank, venue, complete = last[1], last[2], last[3], last[4]
        created_at = last[5]
        if n >= 2:
            mu_obs = sum(rets) / n
            var_obs = sum((x - mu_obs) ** 2 for x in rets) / n if n > 1 else sig0 * sig0
            mu = (n * mu_obs + DRIFT_SHRINK_K * mu0) / (n + DRIFT_SHRINK_K)
            var = (n * var_obs + VOL_SHRINK_K * sig0 * sig0) / (n + VOL_SHRINK_K)
            sigma = math.sqrt(max(var, VOL_FLOOR * VOL_FLOOR))
            sigma = min(sigma, VOL_CAP)
        else:
            # thin history (<2 returns): fall back to the global prior drift,
            # which is negative (most coins fade) -- a coin that just rocketed
            # up usually cools, it does not keep compounding its entry sprint.
            # A small rank-momentum nudge only: a coin that climbed many ranks
            # to reach the top is a little hotter than the average coin.
            rank_mom = 0.0
            if len(v) >= 2:
                r0, r1 = v[0][2], v[-1][2]
                rank_mom = max(0.0, (r0 - r1)) / 100.0
            mu = mu0 + 0.3 * rank_mom
            mu = max(-1.0, min(0.5, mu))
            sigma = max(VOL_THIN, sig0)
            sigma = min(sigma, VOL_CAP)
        # graduated coins: steadier, drift pulled toward 0 (they persist)
        if complete or venue == "pool":
            sigma *= GRAD_VOL_MULT
            mu = mu * (1.0 - GRAD_DRIFT_PULL)
        # fade hazard: curve coins with negative recent drift die fast;
        # graduated coins rarely collapse outright.
        if complete or venue == "pool":
            fade = 0.02
        else:
            fade = 0.10 + max(0.0, -mu) * 0.6
            fade = min(fade, 0.60)
        mu = max(mu, -1.5)
        mu = min(mu, 0.9)
        params[mint] = (mu, sigma, fade, last_lc, complete)
    return params, mu0, sig0


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def size_widen(horizon_hours, n_marks):
    """Extra widening of the simulated size distribution: grows with the
    answer horizon (model error compounds) and with thin history (fewer
    marks = more uncertainty about where the caps sit)."""
    w = 1.35 + 0.30 * math.sqrt(max(horizon_hours, 0.0))
    if n_marks < 4:
        w *= 1.0 + 0.30 * (4 - n_marks) / 4.0
    return min(w, 2.4)


def forecast(task, marks):
    t0 = parse_time(task["t0"])
    places = int(task.get("places", 5))
    edges = task["size"]["edges"]
    n_bins = len(edges) + 1  # 82
    reads = [parse_time(r) for r in task["reads"]]
    last_mark_time = parse_time(marks[-1]["at"])
    horizons = [(r - last_mark_time).total_seconds() / 3600.0 for r in reads]
    mean_horizon = sum(horizons) / len(horizons)
    n_marks = len(marks)

    traj = build_trajectories(marks, t0)
    params, mu0, sig0 = estimate_params(traj, last_mark_time, t0)
    age_hours = (last_mark_time - t0).total_seconds() / 3600.0
    churn = age_adjusted_churn(top5_churn(marks), age_hours)

    # candidate set: top of last mark + best recent risers
    last_rows = sorted(marks[-1]["rows"], key=lambda r: r.get("rank", 999))
    cand = []
    seen = set()
    for r in last_rows[:MAX_CANDIDATES]:
        cand.append(r["mint"])
        seen.add(r["mint"])
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
    rank_counts = [dict() for _ in range(places)]
    size_hist = [[0.0] * n_bins for _ in range(places)]
    total_draws = 0
    mus = [params[m][0] for m in cand]
    sigs = [params[m][1] for m in cand]
    fades = [params[m][2] for m in cand]
    last_lc = [params[m][3] for m in cand]
    n_cand = len(cand)

    for _ in range(N_PATHS):
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
                    if rng.random() < fades[j] * dt:
                        alive[j] = False
                        cur[j] = FADE_FLOOR_LOGCAP
                        continue
                    drift = mus[j] * DRIFT_TAU_HOURS * (
                        math.exp(-t_prev / DRIFT_TAU_HOURS)
                        - math.exp(-h / DRIFT_TAU_HOURS)
                    )
                    cur[j] = cur[j] + drift + sigs[j] * sq * rng.gauss(0.0, 1.0)
            order = sorted(range(n_cand), key=lambda j: cur[j], reverse=True)
            for k in range(places):
                j = order[k]
                mint = cand[j]
                rank_counts[k][mint] = rank_counts[k].get(mint, 0) + 1
                b = bin_of(cur[j], edges)
                size_hist[k][b] += 1.0
            total_draws += 1

    # newcomer mass per place (from churn + mean horizon)
    newcomer = newcomer_share(churn, mean_horizon)

    probabilities = []
    for k in range(places):
        counts = rank_counts[k]
        ranked = sorted(counts.items(), key=lambda kv: -kv[1])
        named = []
        for mint, c in ranked[:MAX_ROWS_PER_PLACE]:
            p = c / total_draws
            if p < MIN_P:
                continue
            named.append((mint, p))
        s = sum(p for _, p in named)
        if s <= 0:
            continue
        # temperature-soften the frequencies before scaling: flatten the
        # distribution a little (log score punishes overconfidence).
        soft = [(mint, (p / s) ** (1.0 / TEMP)) for mint, p in named]
        ss = sum(p for _, p in soft)
        for mint, p in soft:
            p2 = (p / ss) * (1.0 - newcomer)
            p2 = min(p2, MAX_P)
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
    # sizes: widen the simulated histogram (model error grows with horizon
    # and thin history), then convert counts to probabilities, then blend
    # in the newcomer component (v3): a newcomer that takes a place is
    # bigger than the incumbent it displaces, so the newcomer mass sits
    # above the current place-k cap.  Without this the size distribution
    # at young/high-churn cuts collapses onto tiny caps and the real
    # (newcomer-set) caps score near the floor.
    widen = size_widen(mean_horizon, n_marks)
    cur_caps = []
    for r in last_rows[:places]:
        try:
            cur_caps.append(logcap_sol(r["market_cap_lamports"]))
        except (KeyError, TypeError, ValueError):
            cur_caps.append(2.0)
    sizes = []
    for k in range(places):
        hist = size_hist[k]
        tot = sum(hist)
        if tot <= 0:
            continue
        mean0 = sum(i * hist[i] for i in range(n_bins)) / tot
        var = sum((i - mean0) ** 2 * hist[i] for i in range(n_bins)) / tot
        mean = mean0 + (SIZE_GROWTH_PER_SQRT_H * math.sqrt(max(mean_horizon, 0.0))) / 0.05
        sd = math.sqrt(max(var, 0.25)) * widen
        inc = []
        for b in range(n_bins):
            h = hist[b] / tot
            blo = b - 0.5
            bhi = b + 0.5
            nrm = norm_cdf((bhi - mean) / sd) - norm_cdf((blo - mean) / sd) if sd > 0 else 0.0
            inc.append(0.6 * h + 0.4 * nrm)
        s_inc = sum(inc)
        if s_inc <= 0:
            continue
        inc = [p / s_inc * 0.98 for p in inc]
        # newcomer component: wide normal centred above the current
        # place-k cap; premium grows with horizon (more turnover) and is
        # larger at lower places (smaller incumbents are easier to beat
        # by a wide margin).  It also scales with the market's churn:
        # in a hot, high-churn market newcomers rocket (+0.5 or more per
        # 10 min); in a settled market they barely exceed the incumbent
        # (+0.1), so a fixed premium misplaces mass at mature cuts.
        # The multipliers are deliberately mild: the premium is the
        # AVERAGE newcomer jump, not the largest, and the wide sd covers
        # the spread -- an over-aggressive premium wastes the newcomer
        # mass far above the real caps when churn collapses.
        churn_adj = 0.4 + churn
        prem = min(NEWCOMER_PREM_CAP,
                   NEWCOMER_PREM * churn_adj * math.sqrt(6.0 * max(mean_horizon, 0.0))
                   * (0.85 + 0.075 * (k + 1)))
        nc_mean = cur_caps[k] + prem
        nc_sd = NEWCOMER_PREM_SD
        newc = []
        for b in range(n_bins):
            lo = -1e9 if b == 0 else edges[b - 1]
            hi = 1e9 if b == n_bins - 1 else edges[b]
            newc.append(norm_cdf((hi - nc_mean) / nc_sd) - norm_cdf((lo - nc_mean) / nc_sd))
        s_new = sum(newc)
        newc = [p / s_new * 0.98 for p in newc] if s_new > 0 else [0.0] * n_bins
        w_new = newcomer  # same newcomer share that leaves order mass unnamed
        for b in range(n_bins):
            p2 = (1.0 - w_new) * inc[b] + w_new * newc[b]
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
