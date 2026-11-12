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

## v4 (current): why v3 failed in reality, and the fix

**v3's real practice score was place 55, far worse than v2's place 11,
despite a much better backtest (+8.80 vs +1.48 avg delta, all splits
positive).** Diagnosis from the kit data:

- Top-5 churn collapsed 0.8 -> 0.2 at 20:40 and stayed there: the market
  settled before the practice cut (21:00). At the latest backtest cut
  (cut 4, settled) v2 already beats v3.
- Measured newcomer cap jumps: high churn (0.8) -> +0.14..+0.94, avg
  +0.53; low churn (0.2) -> +0.08..+0.21, avg +0.15. Jump ~ 0.65 *
  churn per 10 min.
- v3's newcomer size mixture used a premium with a floor
  (0.40 * (0.4 + churn) * ...), so even in a settled market it shifted
  ~36% of each place's size mass 0.4-0.55 log10 above the current cap --
  ~2x the real low-churn jump. In a settled market the real caps stay
  near the incumbents, so that mass scored near the floor.

**v4 = v2's proven order model + a churn-scaled newcomer size mixture:**
premium = min(0.90, 0.65 * churn * sqrt(6 * horizon) *
(0.85 + 0.075 * (place+1))) -- no floor, so it matches v3's fix at high
churn (fixes v2's worst split, cut 1 size -15.1) and nearly vanishes at
low churn (behaves like v2 at settled cuts). Plus v3's age-adjusted
churn (tau 0.5h) for the churn estimate, so a cut taken just before the
market settles (cut 3: churn 0.8 but next transition 0.2) does not
overstate turnover.

**Backtest (13 splits): v4 avg delta +8.29, worst +2.43, best +19.27 --
all splits positive.** v2: +1.48 / -13.55 / +5.83. v3: +8.80 / +2.60 /
+20.65. On the practice-relevant proxies (cut 3 both-reads, cut 4):
v4 +3.37 / +4.62, v3 +3.19 / +4.59, v2 +2.64 / +5.07. On the full
practice input, v4's forecast sits closest to v2's (the proven model),
unlike v3 which shifts size mass +1.6..+3.3 bins above v2 at places 3-5.

**Honest caveat:** the backtest is in-sample (answers inside the history
window, 10-40 min horizons) while the real practice is out-of-sample
(answers 21:00-22:00, settled market) and the final grading is a
different window again (7-13h horizons, 17h-old market). v3 proved a
great backtest can still place 55. v4 is built to be robust across
regimes rather than to maximise the backtest, but its real practice
score is only known after submission.
