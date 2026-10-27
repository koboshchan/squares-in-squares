# squares-in-squares

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![Version 0.1.0](https://img.shields.io/badge/version-0.1.0-informational)](pyproject.toml)
[![License MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Local Python tools for [Kaggle's Squares in Squares](https://www.kaggle.com/competitions/squares-in-squares). Generate packings, certify them with the bundled official metric, search small cases and merge the best valid packing for each n.

[Quick start](#quick-start) · [Commands](#commands) · [Scoring and validation](#scoring-and-validation) · [Architecture](#architecture) · [Testing](#testing)

## Features

- Complete grid baseline for n=1 through 1000, containing 500,500 unit squares.
- Decimal-preserving CSV handling and the unmodified official v20 interval certifier.
- Small-n rotation search with annealed restarts and certified acceptance.
- Best-per-n merging that rejects invalid candidates.
- Offline reference-coordinate adapter with source attribution.
- Atomic output replacement to protect existing files from failed validation.

No Kaggle login, API keys or automatic submissions are involved.

## Installation

Requires Python 3.12 or newer and Git. NumPy and SciPy are needed for optimization; mpmath and pandas support the metric.

```sh
git clone https://github.com/koboshchan/squares-in-squares.git
cd squares-in-squares
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev,optimize]'
```

On Windows, activate with `.venv\Scripts\activate` instead. For baseline, scoring and merge commands without search or developer tools, install with `python -m pip install -e .`.

## Quick start

Generate a full baseline and certify every packing.

```sh
squares baseline --output submission.csv
squares validate submission.csv --require-complete --report outputs/grid-report.json
```

The grid baseline scores **28.937669916550586**. All 1,000 packings passed the bundled official v20 evaluator. Perfect-square n score zero.

> [!NOTE]
> Grid generation uses exact half-integer geometry. Full interval verification takes longer because many edges touch exactly. Generated CSVs and reports are intentionally ignored by Git.

For a quick smoke test, work with n=1 through 5 and add a tilted construction.

```sh
squares baseline --max-n 5 --output outputs/small.csv
squares optimize --n 5 --attempts 0 --output outputs/tilted5.csv
squares merge outputs/small.csv outputs/tilted5.csv --output outputs/small-merged.csv
squares validate outputs/small-merged.csv --max-n 5 --require-complete
```

## Commands

Run `squares --help` or `squares <command> --help` for individual options. The CLI also works as `python -m squares`.

| Command | Purpose | Main options |
| --- | --- | --- |
| `baseline` | Generate exact grids for n=1..max-n | `--max-n` defaults to 1000; `--output` defaults to submission.csv |
| `validate INPUT` | Fail on any rejected submitted packing | `--max-n`, `--require-complete`, `--report` |
| `score INPUT` | Score with official fallback penalties | `--max-n`, `--report` |
| `optimize` | Search one small-n packing | Required `--n`, `--output`; optional `--input`, `--attempts`, `--maxiter`, `--seed` |
| `merge INPUTS...` | Keep the certified smallest L per n | Required `--output` |
| `references INPUT` | Import credited local JSON coordinates | Required `--output`, `--provenance` |

### Search and merge

```sh
squares optimize --n 12 --attempts 20 --seed 42 --output outputs/search12.csv
squares merge submission.csv outputs/search12.csv --output outputs/merged.csv
squares validate outputs/merged.csv --require-complete
squares score outputs/merged.csv --report outputs/merged-score.json
```

The optimizer starts from a grid or an optional `--input` CSV. It uses annealed random restarts and SciPy L-BFGS-B to reduce overlap and wall penalties while moving and rotating squares. Defaults are 20 attempts, 250 iterations per attempt and seed 42. Search is limited to n<=50 and may find no improvement.

For n=5 it also tries four corner squares around a central diamond, credited to Frits Goebel's 1979 construction via [Friedman's survey](https://erich-friedman.github.io/papers/squares.pdf). A small center dilation protects decimal-rounding contacts. The certified side is approximately 2.707106782894, better than the grid's 3 but not a new record.

Every proposed improvement must pass the official certifier. Search results are validated again before atomic file replacement. Numerical convergence alone never makes a candidate acceptable.

Merging certifies candidates, keeps the smallest measured L independently for each n and preserves the original decimal strings. Invalid candidates are skipped; ties keep the first input. Missing n stay missing. Include the complete baseline as an input when you want a complete submission.

### Import reference coordinates

[Erich Friedman's Packing Center](https://erich-friedman.github.io/packing/index.html) links to [David Ellsworth's square catalogue](https://kingbird.myphotos.cc/packing/squares_in_squares.html). The loader is an offline adapter stub, not a scraper or bundled record collection.

Transcribe permitted coordinates into the format in [examples/records.json](examples/records.json), keeping source and attribution.

```sh
squares references examples/records.json --output outputs/reference.csv --provenance outputs/provenance.json
squares merge submission.csv outputs/reference.csv --output outputs/merged.csv
```

The included n=1 record only demonstrates the format. Length-only tables are not packings; explicit decimal-string centers and angles are required. The loader certifies coordinates without evaluating symbolic expressions or executing SVG content.

> [!IMPORTANT]
> Check competition rules and source permissions before importing third-party records. Attribution does not establish permission, optimality or whether a record is current.

## Scoring and validation

For each n, the metric measures the rotated squares' bounding box and sets L to its larger dimension. The total score is the sum of `L / sqrt(n) - 1`; lower is better. Measurement is translation-invariant.

Missing or invalid n receive the trivial-grid penalty, `ceil(sqrt(n)) / sqrt(n) - 1`. Omitting a difficult packing therefore cannot improve the score.

> [!WARNING]
> A numerical score is not proof that every submitted packing passed. `score` applies the official fallback and reports accepted, missing and rejected counts. Use `validate --require-complete` before submission.

The wrapper uses the unmodified [official metric](https://www.kaggle.com/code/metric/squares-in-squares) version 20, retrieved October 27, 2026. The snapshot and its SHA256 are documented in [vendor/NOTICE](vendor/NOTICE); it does not automatically track upstream updates.

Float SAT provides a preliminary overlap filter. Undecided contacts escalate through interval checks at 50, 100, 200 and 500 decimal digits. The preliminary 1e-8 tolerance is not permission for overlap. Regression tests reject a 1e-30 interior overlap and accept certified edge contact. Unresolved interval cases are rejected.

### CSV format

The exact header is required.

```csv
id,n,square_id,x,y,angle
0,1,0,0.5,0.5,0
```

Row IDs must be sequential from zero. Each submitted n needs exactly n rows with square IDs 0 through n-1. Partial submissions are allowed unless `--require-complete` is supplied.

Coordinates and angles remain decimal strings throughout loading, merging and writing. The v20 snapshot limits absolute coordinates to 10000, absolute angles to 100 radians and numeric literals to 100 characters. Scientific decimal notation is accepted; symbolic expressions, NaN and infinity are rejected.

## Architecture

| File | Responsibility |
| --- | --- |
| `src/squares/core.py` | Decimal-preserving CSV IO, certification, scoring and merge gates |
| `src/squares/optimizer.py` | Rotation-search heuristic and tilted-five seed |
| `src/squares/references.py` | Credited offline coordinate adapter |
| `src/squares/cli.py` | CLI arguments and local command workflows |
| `src/squares/_official_metric.py` | Unmodified upstream metric snapshot |
| `tests/` | Geometry, IO, scoring, search and reference regressions |

mpmath is pinned to 1.4.1 because the snapshot's 50-decimal formatter uses that release's public mantissa and exponent properties. The remaining dependency bounds are in [pyproject.toml](pyproject.toml).

## Configuration

There are no required environment variables, services or credentials. Configure runs through CLI options.

Use `outputs/` for generated candidates and reports. Baseline `submission.csv`, build artifacts and virtual environments are ignored by Git. Keep credentials outside the repository. Reproducibility of numerical search assumes the same seed, attempt count and numeric environment.

## Testing

```sh
pytest -q
python -m compileall -q src tests
python -m build
```

The 45-test suite covers exact contact, microscopic overlap, malformed input, grids, fallback penalties, official-entrypoint parity, merging, references, search and atomic writes. Wheel and source builds include the upstream attribution and license.

## License

Project code is [MIT licensed](LICENSE). The bundled Kaggle metric retains its separate [Apache 2.0 license](vendor/LICENSE-APACHE-2.0) and [attribution notice](vendor/NOTICE).
