# Reproducing checks from the public artifact

The public table is `data/deidentified_events.csv`. It contains the reported phase, event-type, interaction, model-annotation, human-validation, institution, and manual-coding fields without source text or source identities.

## Requirements

Python 3.12 or newer and `uv`:

    uv sync

JSON-array columns can be parsed with the standard library:

```python
import json
import pandas as pd

events = pd.read_csv("data/deidentified_events.csv", dtype=str).fillna("")
events["manual_motive_codes"] = events["manual_motive_codes"].map(json.loads)
events["human_emotion_labels"] = events["human_emotion_labels"].map(json.loads)
```

## Integrity checks

The manifest records the CSV SHA-256 hash, byte size, source hashes used to construct this release, transformation policy, and expected counts. Verify the released CSV against it:

```python
import hashlib
import json
from pathlib import Path

manifest = json.loads(Path("data/deidentified_events.manifest.json").read_text())
payload = Path("data/deidentified_events.csv").read_bytes()
assert hashlib.sha256(payload).hexdigest() == manifest["output"]["sha256"]
assert len(payload) == manifest["output"]["bytes"]
```

Expected event counts are:

| Type | Count |
|---|---:|
| post | 3,874 |
| retweet | 20,427 |
| reply | 816 |

Expected relationship coverage is 20,381 retweets with a target event, all 20,427 retweets with a target account, and all 816 replies with both target fields. The 46 unresolved retweet event targets are empty by design.

Expected annotation coverage is 25,109 Gemma labels and 400 rows each for GPT-5, GPT-5.4, and human validation. Manual annotations cover 4,386 units; motive codes occur on 1,122 events; the paper's reconciled analytic motive sample contains 824 events, 1,266 detailed assignments, and 1,238 parent-category assignments.

## Building the release inside the restricted project

The following command is documented for custodians who possess the restricted inputs:

    uv run python scripts/build_deidentified_dataset.py

It creates fresh random tokens on every run and refuses to overwrite an existing release unless `--force` is supplied. The exporter validates the canonical row, phase, relation, and annotation counts before writing. The source-to-release crosswalks remain in memory and are not persisted.

Build the local OSF ZIP after regenerating the data:

    uv run python scripts/build_osf_package.py

The package builder withholds source-identity-bearing code and derived outputs, audits all staged text files against the restricted identity vocabulary, and writes `nisamprijavila-osf-deidentified.zip` only after that audit passes. It refuses to overwrite an existing ZIP unless `--force` is supplied.

## Restricted stages

Raw collection, post-text reconstruction, LLM inference from text, account verification, manual coding, and ID reconciliation require restricted inputs. Those stages are documented by the privacy-screened source files where possible, but the public package cannot reproduce them independently.
