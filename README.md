squares-in-squares

Local Python tools for Kaggle's [Squares in Squares](https://www.kaggle.com/competitions/squares-in-squares). Generate a complete grid submission, certify geometry using the official metric, search small packings, import credited coordinates and merge the best valid packing for each n. Nothing here logs into Kaggle or submits a file.

Use Python 3.12 or newer.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev,optimize]'
pytest -q
```

Start with the exact half-integer grid. This writes 500,500 squares for every n from 1 through 1000, with sequential row IDs and the required CSV header. The generated submission.csv is intentionally ignored by Git.

```sh
squares baseline --output submission.csv
squares validate submission.csv --require-complete --report outputs/grid-report.json
squares score submission.csv --report outputs/grid-score.json
```

The grid's total score is 28.937669916550586. For each n its side is ceil(sqrt(n)); perfect-square n score zero. Generation uses exact rational grid geometry, not an approximate overlap test. Full official interval verification of a large grid takes much longer than generation because many edges touch exactly.

The score is the sum of L / sqrt(n) - 1. L is the maximum width or height of the bounding box over every rotated corner, independent of translation. Lower is better. Missing or invalid n receive the trivial-grid penalty, so dropping difficult packings cannot improve the score.

The bundled [official metric](https://www.kaggle.com/code/metric/squares-in-squares) is the unmodified version 20 code cell, retrieved October 27, 2026. Its source hash and Apache license attribution are in [vendor/NOTICE](vendor/NOTICE). It uses float SAT as a preliminary filter, then interval arithmetic at 50, 100, 200 and 500 decimal digits for undecided contacts. Its 1e-8 preliminary tolerance is not permission for an overlap. A 1e-30 interior overlap is rejected; exact edge contact is accepted when certified. Unresolved interval cases are rejected. The wrapper delegates geometry and scoring to this code rather than trying to reproduce it with Shapely tolerances.

Coordinates remain decimal strings through CSV loading, merging and writing. The current official snapshot bounds absolute coordinates by 10000, absolute angles by 100 radians and numeric literals by 100 characters. Scientific decimal notation is accepted by the metric. Symbolic expressions, NaN and infinity are not. The wrapper additionally requires the exact id,n,square_id,x,y,angle header and sequential row IDs; square IDs within each n must cover 0 through n-1.

Try the rotating-square search.

```sh
squares optimize --n 5 --attempts 0 --output outputs/tilted5.csv
squares optimize --n 12 --attempts 20 --seed 42 --output outputs/search12.csv
squares merge submission.csv outputs/tilted5.csv outputs/search12.csv --output outputs/merged.csv
squares validate outputs/merged.csv --require-complete
```

The optimizer starts from a grid or an optional --input CSV. For n=5 it also tries the classic four corner squares around a central diamond, credited to Frits Goebel's 1979 construction via [Friedman's survey](https://erich-friedman.github.io/papers/squares.pdf). A small center dilation protects contacts after decimal rounding. Its certified side is approximately 2.707106782894, an improvement over the grid's 3, not a new record.

Numerical search uses annealed random restarts and SciPy L-BFGS-B on overlap and wall penalties, allowing both translation and rotation. It is limited to n<=50 and may find no improvement. The seed and attempt count are reproducible within the same numeric environment. Every proposed improvement must pass the bundled certifier. Output is validated again before atomic file replacement, so a failed candidate cannot overwrite an existing good file.

The reference loader is deliberately an offline adapter stub, not a record scraper. [Erich Friedman's Packing Center](https://erich-friedman.github.io/packing/index.html) points to [David Ellsworth's square catalogue](https://kingbird.myphotos.cc/packing/squares_in_squares.html). Transcribe licensed coordinates into the format in [examples/records.json](examples/records.json), retaining source and attribution. The included n=1 example is only a format demonstration.

```sh
squares references examples/records.json --output outputs/reference.csv --provenance outputs/provenance.json
squares merge submission.csv outputs/reference.csv --output outputs/merged.csv
```

Length-only record tables are not packings. This adapter requires explicit decimal-string centers and angles, re-certifies them, and never evaluates symbolic formulas or executes SVG content. It does not assert that a reference is current, optimal or permitted for competition use. Check competition rules and source permissions before importing third-party records.

The merge command certifies candidates and keeps the smallest measured L independently for each n. Invalid candidates are skipped, ties retain the first input, and absent n stay absent. Use a full baseline as the first input when you want a complete submission. Partial submissions are supported; use score --max-n for small local smoke tests.

The strict validate command exits unsuccessfully on any rejected submitted packing. It permits missing n unless --require-complete is supplied. The score command instead follows official fallback semantics and reports accepted, missing and rejected counts, so a numerical score alone is not evidence that every packing was accepted. Full details can be saved with --report.

Application code lives in src/squares. core.py owns decimal-preserving CSV IO and certification gates, optimizer.py contains the heuristic, references.py implements the offline record adapter, and cli.py exposes the commands. The upstream metric lives in _official_metric.py and should only change with a documented notebook-version update. mpmath is pinned to 1.4.1 because the snapshot's 50-decimal formatter uses that release's public mpf mantissa and exponent properties.

Run pytest -q for contact precision, exact-grid, malformed-input, fallback-score, official-entrypoint parity, reference, merge, optimizer and atomic-write coverage. Run python -m build for distributable wheels and source archives. Generated CSVs, reports, virtual environments and credentials do not belong in commits. A wheel includes the upstream metric attribution and license text.

Project code is MIT licensed under [LICENSE](LICENSE). The bundled Kaggle metric remains Apache 2.0 licensed, with its separate attribution and license retained in vendor/.
