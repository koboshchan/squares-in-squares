# Experimental fp64 algorithm portfolio

`make -C cuda all ARCH=sm_80` builds standalone `algos`, without PyTorch.
`cuda/algos seed.txt epochs steps chains seed output.json [mode]` uses the
packing CLI and JSON coordinates (radians). ALGOS_MODE selects mode if the
optional argument is omitted. Zeros choose 40 epochs, 64 steps and one bounded
SM wave. Existing packing is unchanged.

1. LS-style stochastic compression, not event-driven hard-particle dynamics.
   Single-square thermal moves, analytical SAT-penalty gradient descent with
   global backtracking, gradual container shrink and jam backoff.
2. Basin hopping. Perturb an adaptive number of squares, relax and shrink,
   retain only a lower feasible basin. Kick size increases on failure and
   decreases on improvement. No Metropolis uphill basin acceptance.
3. CPU grid / rotated-lattice / tilted-strip portfolios plus enumerated
   corner-grid/45-degree diagonal bands (the main packing portfolio), followed
   by GPU SAT-penalty polishing. Candidates may overlap before polishing.

All geometry and objective calculations are double precision. One deterministic
block owns each chain, with bounded global workspaces rather than n-sized shared
arrays. Analytic gradients use the active SAT axis, deterministic reductions,
backtracking, and periodic floating feasibility/dilation checkpoints. Checkpoints
are NOT certificates. Incumbent fallback is preserved. External literal AND
roundtrip certification is mandatory before accepting any result.

`full_run.py --solver algos --mode 2 ...` and `scheduler.py --solver algos
--mode 2 ...` select the new executable. Separate output directories per solver,
mode and run are required; do not reuse another solver's resumable state.

`make -C cuda check` includes the SAT gradient finite-difference selftest.
`python cuda/tests/algos_smoke.py` runs bounded deterministic output tests at
n=1,5,19,70,1000 in all three modes, with independent C++ literal and roundtrip
acceptance. These correctness tests DO NOT establish quality improvements.

`python cuda/algos_bench.py --incumbent seed42.csv ...` applies equal wall
budgets to repeated short independent restart chunks. Startup/timeout waste
counts; certification is deferred and reported separately. The default 1-epoch
chunks measure restart quality, not sustained long adaptive trajectories. Increase
epochs AND per-trial budget to investigate sustained compression/basin hopping.
`chain_steps` for algos counts actual gradient-loop iterations across chains,
not requested iterations; packing's counter retains its existing definition.
Compare certified side lengths, not directly the two throughput definitions.

The optional `algos_sweep.py` is resumable and certifies all 1000 output puzzles
before emitting a final report. No Kaggle submission or main deployment occurs.
