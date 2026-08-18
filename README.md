# DRIFT-SENSE — Navigation-Error Recovery

AI-assisted localization of a reference site inside a larger search image for periodic semiconductor layouts.

## Quick start
```bash
pip install -r requirements.txt
python dataset_generator.py --architecture DRAM --num-pairs 5 --output-dir data/demo
python inference.py data/demo/reference_0000.png data/demo/search_0000.png
```

The inference command outputs one coordinate: `(x, y)`.

## Evaluator interface
```bash
python inference.py REFERENCE_IMAGE SEARCH_IMAGE
```

## Contents
- `inference.py` — evaluator-facing localization entry point
- `dataset_generator.py` — standalone DRAM/FinFET synthetic pair generator
- `train.py` — optional PyTorch scaffold
- `src/matcher.py` — multi-scale localization baseline
- `src/model.py` — optional Siamese encoder scaffold
- `requirements.txt` — dependencies
- `citations.md` — supporting references
- `docs/evaluation.md` — evaluation protocol

The baseline uses multi-scale normalized template correlation, nearby-peak suppression, and a center-aware tie-break for ambiguous periodic matches. No fabricated benchmark results or fake model weights are included.