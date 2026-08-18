# DRIFT-SENSE — Navigation-Error Recovery

AI-assisted localization of a reference site inside a larger search image for periodic semiconductor layouts.

## Quick start

```bash
pip install -r requirements.txt
python dataset_generator.py --architecture DRAM --num-pairs 1 --output-dir sample/generated
python inference.py sample/generated/reference_0000.png sample/generated/search_0000.png
```

The inference command outputs one coordinate `(x, y)` in Search-image pixel coordinates.

## Repository
- `inference.py` — evaluator-facing localization entry point
- `dataset_generator.py` — standalone DRAM/FinFET synthetic-pair generator
- `train.py` — optional PyTorch scaffold
- `src/matcher.py` — multi-scale localization baseline
- `requirements.txt` — dependencies
- `citations.md` — supporting references
- `docs/evaluation.md` — evaluation protocol

## Method
The baseline performs multi-scale normalized template correlation, candidate peak suppression, and a center-aware tie-break when multiple candidates have nearly equal scores. It is a runnable classical baseline; no fabricated accuracy or runtime claims are included.

## Critical interface
`python inference.py REFERENCE_IMAGE SEARCH_IMAGE`

No manual path edits are required.