# CUDA subproblem proof of concept

Standalone CUDA C++17 and cuRAND device Philox, **no PyTorch**. Supports n=1..50 only. This is a search demo, not a competitive solver or record claim. It never submits anything.

## Build and run

Install the repository's optimize and dev extras using its normal Python 3.12+ environment. CUDA 12.4 and g++ were used on the RTX 4070 SUPER through the desktop's existing Ubuntu WSL installation. No WSL restart or system configuration changes are needed.

From the repository root, on Linux/WSL:

~~~sh
nvcc -O3 -std=c++17 -arch=sm_89 cuda/packing.cu -o cuda/packing
nvcc -O3 -std=c++17 -arch=sm_89 cuda/selftest.cu -o cuda/selftest
./cuda/selftest
python cuda/demo.py run --executable ./cuda/packing \
  --epochs 20000 --steps 32 --chains 256
pytest -q
~~~

Use an architecture supported by your NVIDIA GPU; sm_89 is the measured 4070 SUPER. Do not use fast-math. Native Windows needs a compatible MSVC toolchain; this demo was compiled with WSL CUDA instead.

The driver generates seed files, runs each n sequentially, certifies GPU output with the repo's **unchanged official v20 interval certifier**, and writes outputs/cuda/report.json and gpu-certified.csv. It never substitutes a baseline if a GPU proposal fails. The direct executable takes seed-file, epochs, steps, chains, RNG seed, output JSON. Chains must be a multiple of eight, 8..4096; steps are 1..4096. Empty q means no zero-energy checkpoint was found and is rejected by the host driver.

For certification of checked-in results without a GPU:

~~~sh
python cuda/demo.py report --directory cuda/results
~~~

## Design and safety

One 64-thread block owns one chain; x,y,angle live in shared float32 arrays. A collision-free 16x16 uniform-grid bucket index with atomic linked lists filters neighbors. Cell width 2 exceeds the maximum center interaction radius sqrt(2). Final checkpoint energy is independently recomputed using **all pairs**, never the grid filter.

Threads propose single-square translation/rotation moves on a conservative independent set. Cyclic priorities select squares whose original centers are more than 3.3 apart; bounded per-coordinate translation is at most 0.12. Thus two simultaneously selected squares cannot interact either before or after their moves. Proposals read the same immutable shared snapshot and commit only after a block barrier. The priority rule is deliberately conservative and often selects only one square for small dense packings.

The objective is sum of squared positive SAT penetration depths plus squared wall violations. Metropolis moves use persistent per-thread Philox states. Every epoch uses a fixed common L and an eight-temperature geometric ladder, repeated across independent replica groups. A separate kernel proposes disjoint adjacent replica exchanges between completed walk kernels; it swaps configurations, not RNG streams. Alternating parity never exchanges across ladder-group boundaries. Cooling makes this a heuristic optimization schedule, not a stationary sampling claim.

A zero checkpoint copies its coordinates to the host and retains the latest zero-L snapshot. All chains are reseeded from that winner at L minus 0.002, scaled inward. The ladder always shares L. The executable retains only its latest zero checkpoint, so an eventual host rejection does not recover earlier checkpoints. There is no on-GPU exact validity claim.

The host converts float32 coordinates to float64, tries bounded center dilation factors 1.000002, 1.00001, 1.0001, and passes decimal strings to the repo's interval certifier. Dilation is explicitly included in measured L; it is not an allowed-overlap tolerance. No changes were made to the official metric.

Allocation is fixed and small, under 2 MB at 256 chains (under 32 MB at the chain limit), excluding CUDA context overhead. No per-step allocation, dataset input, background processes, extra GPU rental, or external service is used.

## Measured demo

November 1, 2026, RTX 4070 SUPER, CUDA 12.4, g++, 256 chains, 20,000 epochs, 32 steps, seed 42. Device was shared with normal Windows graphics applications. Times include loop kernel launches, checkpoint transfers, and replica exchanges, but exclude CUDA initialization, seed preparation and Python certification.

| n | Repo starting seed L | GPU certified L | Seconds | Million chain-steps/s | Million proposals/s |
|---|---:|---:|---:|---:|---:|
| 5 | 2.707106782894 | 2.707144336682 | 4.480 | 36.58 | 36.58 |
| 10 | 4 | 3.707824944118 | 6.384 | 25.66 | 25.82 |
| 11 | 4 | 3.889855843061 | 7.606 | 21.54 | 25.00 |
| 17 | 5 | 4.710065339847 | 9.187 | 17.83 | 25.93 |
| 50 | 8 | 8.000154190680 | 14.477 | 11.32 | 22.55 |

A chain-step is one block's independent-set proposal round, not a sweep over all n squares. Proposal counts measure individual square proposals, including rejected ones. Each run had 163,840,000 chain-steps. Floating atomic linked-list summation can vary slightly between runs even with a fixed RNG seed.

The baseline here is **the repo's existing deterministic starting construction**, not a timed comparison against its default 20-attempt SciPy optimizer. n=5 uses the existing credited tilted-five construction; all others use grid(n). GPU improves n=10,11,17 relative to these seeds. It does **not** improve n=5 or n=50; their final slightly larger L comes from seed clearance/dilation. Do not replace those better baseline entries. For n=5, the improvement over the trivial grid's L=3 is inherited from the CPU construction, not discovered by this kernel. No speedup over the CPU optimizer is claimed.

results/ contains raw device output, official-certified CSV, and the measured report. The measured target_L is the search box, while GPU certified L is the actual rotated bounding-box side including safety dilation.

## Validation

The existing 45 tests passed before adding host gate tests. The new tests validate seed generation, invalid/NaN/duplicate rejection, and certification of all saved GPU candidates.

selftest checks 250,000 GPU SAT pair depths against an independent float64 CPU oracle and 5,000 grid-neighbor energies against CUDA all-pairs energies, including cell-boundary and touching fixtures. Observed maximum absolute errors were 2.30535e-7 and 2.38419e-7 respectively. These are numerical checks, not certification tolerances.

CUDA compute-sanitizer memcheck passed on n=17, 16 chains, 20 epochs, 32 steps with zero errors. Racecheck passed on n=17, 8 chains, 4 epochs, 8 steps with zero hazards/errors/warnings.

~~~sh
compute-sanitizer --tool memcheck --error-exitcode 3 \
  ./cuda/packing outputs/cuda/seed17.txt 20 32 16 42 outputs/cuda/memcheck.json
compute-sanitizer --tool racecheck --error-exitcode 3 \
  ./cuda/packing outputs/cuda/seed17.txt 4 8 8 42 outputs/cuda/racecheck.json
~~~

## Limits and next experiments

This implementation favors transparent correctness over maximum throughput. The 16x16 direct grid, n<=50 storage, conservative proposal exclusion, repeated all-chain reseeding, fixed shrink amount and random-walk moves are deliberate POC limits. Longer runs and more chains are not guaranteed to improve results. n=50 stalls at its grid seed. Incremental neighbor maintenance, multiple retained feasible states, adaptive move scales and a GPU/CPU local-polish stage are possible next experiments, not implemented features.
