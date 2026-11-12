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

A Monte Carlo (fixed seed, 4000 paths) simulates every candidate coin
forward to each answer read, ranks the simulated caps, and reads off rank
probabilities per place and the size-bin distribution per place. A slice of
each place is left unnamed for newcomers — coins born after the cut that no
history can name (about a third of top-5 slots in the rehearsal). Place
caps are shifted up with horizon, since newcomers keep pushing the top
higher.

## Why it beats persistence

The house baseline bets the last mark's order holds. In the rehearsal the
top 5 turns over fast (only 1 of the top 5 at 20:10 was still there at
20:50), and the place caps inflate steadily as new coins arrive. This
forecaster prices momentum, survival, and newcomer mass instead of
freezing the last snapshot.

## Local backtest (same log-score formula as the rules)

Scored against later marks as pseudo-answer reads, on the dev rehearsal:

| cut | answers | this model | baseline | delta |
|-----|---------|-----------|----------|-------|
| 20:30 | 20:40, 20:50 | -29.68 | -33.48 | **+3.80** |
| 20:20 | 20:30, 20:40, 20:50 | -50.46 | -49.96 | -0.50 |
| 20:40 | 20:50 | -24.72 | -29.87 | **+5.15** |

The baseline scores -40.41 on the official practice. The model beats it on
the two more mature splits and is essentially tied on the most immature
one. (The competition's own `grade-local` cannot score this bounty
locally — its scorer/reference are server-only — so these are backtests
with the identical formula, not the official practice score.)

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
# writes out/forecast.json (probabilities, top5, sizes)
```

Runtime ~3s on the dev history; well under the 300s limit. Two runs on
the same history produce byte-identical output.
