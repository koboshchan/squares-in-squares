# RTX 4070 Super finite suite, 2026-11-01

Official interval certification accepted all 1,000 packings, containing 500,500 squares.
All 969 nonsquare cases actually ran on the GPU; the 31 perfect squares use exact grids.
There were 988 finite-suite GPU launches: 969 coverage, 12 tuning, 7 post-tune quality.
Earlier exploratory runs are excluded from that count.

Official lower-is-better score: 25.4885466242075.
Grid baseline: 28.9376699165506.
Absolute gain: 3.44912329234305 (11.919147%).
Improved cases: 198.
Best-packing provenance: 197 analytic, 1 GPU, 802 grid, 0 other.
Coverage does not mean every GPU candidate improved its seed.

The quality-ranked tuning winner was 64 threads, 1,024 chains, warm ladder.
On n=11 with 4,000 epochs and 64 steps per epoch it measured 44.233 million
chain-steps/s and polished to L=3.88674602889437.
With the same chain count and work, 128 threads measured 43.632 million/s
and 256 threads 29.522 million/s. Those runs polished to approximately 4.0.
These are single-run measurements, not a repeated statistical benchmark.
The 44.252 versus 25.214 million/s cold figures use different chain counts,
and both polished candidates failed certification; they are not a valid
same-workload speedup claim or the quality winner.

Cases within 1e-14 of published decimal sides: 5, 10, 27.
These are analytic constructions, not CUDA record rediscoveries.
The other focus cases still miss their published targets. See record-comparison.json.
The 8-epoch, 16-step, 64-chain coverage pass is finite execution coverage,
not exhaustive search, a proof of optimality, or record-quality search.

CSV SHA-256: e202208231d31ac82efd36b04f7f59014ac505709729e72619fe2136bd8f8cac
Gzip SHA-256: d33d8e61836846d41fd1afc1ef0f52c313f83e36c45a7ef17b705f8728ae19e3

The bundled official metric was not modified. Certification ran with the
guarded finalize-only entrypoint after repairing an unguarded experimental
controller entrypoint incompatible with Python 3.14 forkserver spawning.
