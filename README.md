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

## Why it beats persistence

The house baseline bets the last mark's order holds. In the rehearsal the
top 5 turns over fast (only 1 of the top 5 at 20:10 was still there at
20:50), and the place caps inflate steadily as new coins arrive. This
forecaster prices momentum, survival, and newcomer mass instead of
freezing the last snapshot. Its edge is largest at mature cuts and
longer horizons, where persistence is weakest.

## Local backtest (same log-score formula as the rules)

Scored against later marks as pseudo-answer reads, on the dev rehearsal
(all 10 single-answer splits; delta = this model minus persistence):

| cut | answer | this model | baseline | delta |
|-----|--------|-----------|----------|-------|
| 20:10 | 20:20 | -55.58 | -42.03 | -13.55 |
| 20:10 | 20:30 | -60.41 | -62.64 | +2.23 |
| 20:10 | 20:40 | -61.19 | -65.29 | +4.10 |
| 20:10 | 20:50 | -60.88 | -66.72 | +5.83 |
| 20:20 | 20:30 | -50.39 | -47.78 | -2.61 |
| 20:20 | 20:40 | -48.87 | -49.38 | +0.51 |
| 20:20 | 20:50 | -49.65 | -52.72 | +3.07 |
| 20:30 | 20:40 | -27.19 | -29.65 | +2.47 |
| 20:30 | 20:50 | -33.67 | -37.31 | +3.63 |
| 20:40 | 20:50 | -24.80 | -29.87 | +5.07 |

Average delta **+1.08** (v1: -1.05); worst split **-13.55** (v1: -16.37).
Multi-answer splits: cut 20:30 -> 20:40+20:50 **+2.64**; cut 20:20 ->
20:30+20:40+20:50 **+2.13**; cut 20:40 -> 20:50 **+5.07**.

Known weakness: the 20:10 -> 20:20 split (a 10-minute-old market, 4 of
5 top-5 slots taken by coins born after the cut) is still far behind
persistence, and the 20:30 splits lag v1 slightly — at 20:30 the market
is just about to calm down from 0.80 to 0.20 churn, which no history
available at the cut can foresee (a regime change). The practice and
final both run on mature markets (6 marks / 11 hours of history), where
the model is at its best.

The baseline scores -40.41 on the official practice. The model beats it
on the mature splits and is far ahead on long horizons. (The
competition's own `grade-local` cannot score this bounty locally — its
scorer/reference are server-only — so these are backtests with the
identical formula, not the official practice score.)

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
# writes out/forecast.json (probabilities, top5, sizes)
```

Runtime ~6.5s on the dev history; well under the 300s limit. Two runs on
the same history produce byte-identical output.
