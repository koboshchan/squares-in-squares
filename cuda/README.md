Standalone CUDA C++17 packing research with cuRAND, without PyTorch. packing.cu handles n=1..1000; seeded.cu is a separate double-precision quality path limited to n<=384. Python host tools stay together here so existing imports and commands remain compatible. Device regression programs live in tests/. No solver behavior was changed by the layout cleanup.

The latest complete local artifact scores 24.21590689512412 with all 1000 packings accepted. See results/records-summary.json and ../KERNEL_BRIEF.md. The 129 strict record imports are credited geometry, not GPU discoveries. The 28-output seeded suite and four subsequent GPU attempts earned zero additional gains. Older results/rtx4070s-20261101/ describe the pre-import 25.48854662420754 result and must not be confused with the current artifact. Large CSVs, logs, binaries and generated output belong outside Git; small raw fixtures and compact evidence remain for regression tests.

From the repository root, install and test the host tools with the following commands.

```sh
python -m pip install -e '.[dev,optimize,records]'
python -m pytest -q
python -m compileall -q src cuda
python -m build
```

Build on a CUDA-equipped host. The default architecture is the measured RTX 4070 SUPER's sm_89. For RTX PRO 6000 Blackwell use CUDA 12.8+ and sm_120. No fast-math is enabled. Architecture changes require rebuilding, for example with make -B.

```sh
make -C cuda ARCH=sm_120
make -C cuda check ARCH=sm_120
```

The three device tests independently compare SAT against a float64 oracle, check grid/all-pairs energies and n=1000 strided exchanges, and compare seeded analytic gradients against finite differences. They require an NVIDIA GPU; Python host tests do not. Run compute-sanitizer memcheck and racecheck on a small finite fixture before accepting changed device code.

Both search executables take `seed-file epochs steps chains rng-seed output.json`. Seed text starts with `n L`, then n center-x, center-y, angle triples. Angles are radians. Raw JSON includes n and q triples. Generated outputs are proposals only, never certified geometry.

```sh
python cuda/demo.py run --executable ./cuda/packing --epochs 20000 --steps 32 --chains 256
python cuda/demo.py report --directory cuda/results
python cuda/scheduler.py --initialize --jobs 0 --directory outputs/research
python cuda/scheduler.py --coverage --jobs 10 --epochs 8 --steps 16 --chains 64 --directory outputs/research
python cuda/scheduler.py --finalize --jobs 0 --directory outputs/research
squares validate outputs/research/submission.csv --require-complete --report outputs/research/validation.json
```

Use a new research directory rather than overwrite an incumbent. Initialization creates analytic Göbel staircase, tilted-five and strip seeds, plus exact grids. The scheduler persists state.json and best/ and only merges officially certified improvements. --coverage visits unattempted nonsquares; a completed pass does not imply quality gains. batch.py prepare/merge supports finite GPU-only remote batches. autotune.py OUTPUT --prepare generates twelve launch/temperature configurations; analyze raw results with autotune.py OUTPUT/tuning. record_report.py compares certified decimal sides against cited published decimals, not exact algebraic records.

packing.cu uses shared float32 coordinates for small n and global storage for larger n, bounded by O(chains*n). A chain is a block; eight-temperature ladders exchange completed configurations. Conservative independent-set moves use immutable proposal snapshots and barriers. SAT penetration and wall penalties guide the search, with all-pairs checks before checkpoints. PACK_THREADS supports 32,64,128,256; PACK_TMIN, PACK_TMAX, PACK_MOVE, PACK_ROTATION and PACK_SHRINK expose runtime search settings. Its shared-memory grid is a small-n optimization, not a large-n scalable algorithm. The measured quality-ranked n=11 tuning result was 44.233 million chain-steps/s at 64 threads and 1024 chains on a 4070 SUPER. A chain-step is not a full n-square sweep.

seeded.cu adds adaptive individual/collective moves, reheats, parallel tempering and on-device gradient/backtracking polish in float64. SEED_COLLECTIVE=0 and SEED_POLISH=0 disable components for ablations; SEED_TMAX, SEED_MOVE, SEED_SHRINK and SEED_STRUCTURED configure bounded experiments. seeded.py write_seed preserves source and undilated certified L; certify_raw rejects invalid geometry and only awards GPU credit for strict improvement over both incumbent and source seed. Its rank_targets default remains 128, so pass a larger bound intentionally when researching up to 384. polish.py uses float64 CPU SLSQP with analytic SAT Jacobians; refine.py adds finite CPU basin hopping. CPU-assisted improvements must be labeled as such.

records.py imports linked SVG geometry from https://kingbird.myphotos.cc/packing/squares_in_squares.html using the separately obtained author's parser at https://github.com/Davidebyzero/packing_squares_in_squares__tools . Source URLs, hashes, attribution and parser revision are preserved in local manifests. No third-party parser or original SVGs are redistributed here; public access is not a reuse license. Install the records extra and check competition/source permissions before use. Most catalogue entries are below 325, but some extend higher. Missing targets are unknown, not fabricated grid records.

Every accepted candidate goes through the unchanged official v20 interval certifier in squares.core.certify. Float64 SAT is a diagnostic, not a substitute. Preliminary 1e-8 classification is not overlap allowance; interval contacts escalate through 50,100,200,500 digits and unresolved cases fail. Any safety dilation contributes to measured L. Perfect-square grids remain exact. Imported records do not earn GPU credit. A complete CSV has 500,500 rows plus header id,n,square_id,x,y,angle, consecutive global IDs, local IDs 0..n-1 and decimal-string coordinates. Lower total sum(L/sqrt(n)-1) is better.

The local current CSV is outputs/records/merged/submission.csv, SHA256 5d321184da9ebf7239acf8701f608694bd0fe3b856c92033652473aa8d3b222f. It is intentionally not bundled in Git. Saved small raw candidates remain under results/ and can be recertified with demo.py report; its regenerated gpu-certified.csv is ignored. No tool automatically submits to Kaggle, rents a GPU or launches an unbounded service.

### Endless 4070S passes

Double-click `cuda/endless-4070s.bat` on the desktop checkout, or run
`python cuda/endless.py --directory outputs/full-4070s --incumbent inputs/incumbent-safe.csv`
from the repo root under Linux/WSL. Install the project dependencies first.

The supervisor watches an existing seed-42 full_run process without signalling it.
It shows tqdm progress, current n, per-pass wins, and the sum of certified
per-n score contributions recomputed from best/*.json. The initial CSV is
independently verified while seed 42 keeps running. After that pass exits,
all 1000 puzzles must pass C++ literal and numeric-roundtrip certification
and both Python official metric paths before best-so-far.csv is atomically
replaced. Subsequent passes reuse the per-n bests, use seeds 43,44,...,
and require strictly smaller literal and roundtrip lengths for replacements.

endless.json persists pass, seed, and transition phase. state.json persists
completed cases and the active n. live.json is a small machine-readable status.
Each completed pass is archived under passes/ with its CSV, state, and score;
score-per-pass.json is rebuilt idempotently. An OS file lock prevents duplicate
supervisors and /proc process detection prevents duplicate workers. On reboot,
double-click the batch file to resume the unfinished pass. No automatic startup
is installed. If a worker or final certification fails, the supervisor exits
with an error instead of skipping a pass or replacing the known-good CSV.
The supervisor never submits to Kaggle.

The latest endless supervisor uses only the C++ certifier, both literal and
`--roundtrip`, including candidate replacement, live length metadata, and
1000/1000 final CSV promotion. It does not import the Python metric or certifier.
The legacy seed-42 process remains untouched and retains its original checks
until it exits. New passes use cpp_pass.py with up to four concurrent n workers.
Each owns separate seed, output, log and certification files. Only the coordinator
writes pass state. VRAM and available host RAM are checked before dispatch, with
2GiB reserved and conservative per-worker budgets. batch-tuning.json can cap
concurrency at the best measured throughput level. No fp32 kernel change is used.
