# pump.fun top-5 market-cap forecaster

Entry for make.fun bounty `pumpfun-top5-forecast-01`: predict which 5
pump.fun coins launched since the opening will be the largest by market cap
at each of seven hourly answer reads, with honest probabilities, and how
large each place's market cap will be.

## Approach

`main.py` (Python 3.13, standard library only, deterministic, no network)
models each coin's log10 market cap (in SOL) as a Brownian motion with drift
and volatility, estimated from its own recent trajectory and shrunk toward
global priors (young/noisy coins shrink more). Graduated (pool, complete)
coins are steadier than bonding-curve coins; a fade hazard lets weak curve
coins collapse, which is how most pump.fun coins actually die. A hot
streak's drift mean-reverts over ~3 hours, so a long-horizon forecast does
not compound a 10-minute run forever.

A Monte Carlo (fixed seed, 8000 paths) simulates every candidate coin
forward to each answer read, ranks the simulated caps, and reads off rank
probabilities per place and the size-bin distribution per place. The rank
frequencies are temperature-softened slightly (T=1.15): the log score
punishes overconfidence, so the raw frequencies are flattened a little
before they are written out.

Three things the v1 model got wrong, fixed in v2:

- **Newcomer mass is churn-adaptive and horizon-growing.** The share left
  unnamed per place starts at the market's own churn — the fraction of
  the top-5 that turned over in the MOST RECENT 10-minute transition
  only (churn can collapse in one mark as a market settles, so averaging
  over stale transitions badly overstates it) — and grows 0.30/hour with
  the answer horizon, capped at 0.65. v1 left a fixed ~0.30, far too
  little at long horizons where most top-5 slots turn over. At a 7–13h
  horizon the model now leaves 0.65 unnamed, which is honest: most of
  the top 5 will be coins born after the cut.
- **Size widening grows with horizon and thin history.** The simulated
  size distribution is widened more the further out (and the fewer
  marks) the forecast runs, because model error grows with both. v1's
  fixed widening was too narrow at long horizons and young markets,
  where the size score collapsed.
- **Volatility is less shrunk and floored higher.** Young coins are not
  assumed steadier than they are; thin-history coins get a higher vol.
  (An age-implied drift for thin-history coins was tried and reverted:
  a coin that just rocketed up usually cools rather than compounding
  its entry sprint, so the negative global prior is the better default.)

Two things v2 still got wrong, fixed in v3:

- **Newcomer size mixture.** v2's size distribution was purely the
  simulated incumbent coins, so at young/high-churn cuts it put most
  mass on tiny caps for places 2–5 while the real answer was set by
  newcomers arriving with much larger caps (observed newcomer jumps
  +0.3 to +0.9 log10 per 10 min, biggest at the lower places). Each
  place's size distribution is now a mixture: (1 − newcomer share) of
  the simulated incumbent distribution plus newcomer share of a
  newcomer distribution — a wide normal centred above the current
  place-k cap (a newcomer that displaces the incumbent is bigger than
  it). The premium grows with the horizon, is larger at lower places,
  and scales with the market's churn (hot markets: newcomers rocket;
  settled markets: they barely exceed the incumbent). The multipliers
  are deliberately mild — the premium is the *average* newcomer jump,
  not the largest — because an over-aggressive premium wastes the
  newcomer mass far above the real caps when churn collapses.
- **Age-adjusted churn.** The churn of the most recent transition is a
  noisy, possibly stale signal: a market's gold-rush churn collapses
  as it ages (the rehearsal goes 0.80 → 0.20 within ~30–40 minutes),
  so an old market's churn estimate is pulled toward a low floor
  (exp decay, tau 0.5h). Without this, a cut taken just before the
  market settles overstates near-future turnover and the newcomer mass
  (order and size) is far too aggressive. This single change turned
  v2's worst mature-cut splits from negative to clearly positive.

## Why it beats persistence

The house baseline bets the last mark's order holds. In the rehearsal the
top 5 turns over fast (only 1 of the top 5 at 20:10 was still there at
20:50), and the place caps inflate steadily as new coins arrive. This
forecaster prices momentum, survival, and newcomer mass instead of
freezing the last snapshot. Its edge is largest at mature cuts and
longer horizons, where persistence is weakest — and v3's newcomer size
mixture closes the young-market size leak that was v2's biggest loss.

## Local backtest (same log-score formula as the rules)

Scored against later marks as pseudo-answer reads, on the dev rehearsal
(13 splits: 10 single-answer + 3 multi-answer; delta = this model minus
persistence). v3 vs v2:

| cut | answer | v3 | baseline | v3 delta | v2 delta |
|-----|--------|----|----------|----------|----------|
| 20:10 | 20:20 | -39.43 | -42.03 | +2.60 | -13.55 |
| 20:10 | 20:30 | -47.58 | -62.64 | +15.06 | +2.23 |
| 20:10 | 20:40 | -46.29 | -65.29 | +19.00 | +4.10 |
| 20:10 | 20:50 | -46.06 | -66.72 | +20.65 | +5.83 |
| 20:10 | all 4 | -45.09 | -59.17 | +14.08 | +3.75 |
| 20:20 | 20:30 | -44.00 | -47.78 | +3.78 | -2.61 |
| 20:20 | 20:40 | -42.21 | -49.38 | +7.16 | +0.51 |
| 20:20 | 20:50 | -42.59 | -52.72 | +10.13 | +3.07 |
| 20:20 | all 3 | -42.88 | -49.96 | +7.08 | +2.13 |
| 20:30 | 20:40 | -26.62 | -29.65 | +3.03 | +2.47 |
| 20:30 | 20:50 | -33.25 | -37.31 | +4.06 | +3.63 |
| 20:30 | both | -30.29 | -33.48 | +3.19 | +2.64 |
| 20:40 | 20:50 | -25.28 | -29.87 | +4.59 | +5.07 |

Average delta **+8.80** (v2: +1.48); worst split **+2.60** (v2: −13.55);
best **+20.65**. Every split is now positive — v3 has no losing regime
in the rehearsal. The mature cuts (20:30, 20:40 — the closest analogues
of the real practice cut at 20:50 and the 11-hour final cut) all beat
v2 except 20:40 → 20:50, which trails v2 by 0.48 on a single 10-minute
read.

The baseline scores -40.41 on the official practice. The model beats it
on every backtest split. (The competition's own `grade-local` cannot
score this bounty locally — its scorer/reference are server-only — so
these are backtests with the identical formula, not the official
practice score.)

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
# writes out/forecast.json (probabilities, top5, sizes)
```

Runtime ~7s on the dev history; well under the 300s limit. Two runs on
the same history produce byte-identical output.
