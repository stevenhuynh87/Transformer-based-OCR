# jersey-rect

Spatio-Temporal Deformable Rectification and Transformer-based OCR for
in-the-wild sports jersey number recognition (target: IEEE Signal Processing Letters).

```
Tracklet (T frames)
 -> legibility filter -> pose (ViTPose) -> normalized torso crop
 -> rectifier (TPS-STN) -> deformable temporal alignment (DCNv2)
 -> recognizer (PARSeq, digits 0-9, length <= 2) -> tracklet head
 -> jersey number + confidence (or "illegible")
```

## Layout

Sprint 01 only uses the folders below; the others are empty (`.gitkeep`) and get filled in later sprints.

| Folder | Content (S01) |
|---|---|
| `scripts/` | `check_env.py`: GPU, PyTorch version, free memory |
| `utils/` | `seed.py` (seed + deterministic), `logging.py` (W&B / TensorBoard) |
| `third_party/` | `jersey-number-pipeline` (Koshkina & Elder), kept unmodified |
| `tests/` | pytest |
| `notes/` | reading notes `R01-R04.md`, `R06.md` |
| `reports/` | sprint reports `Sxx.md` |
| `docs/` | `setup_notes.md`, `jnr_onepager.md` |

Reserved for later sprints: `configs/`, `data/`, `models/`, `augment/`, `eval/`,
`analysis/`, `experiments/`, `baselines/`, `paper/`, `thesis/`, `demo/`.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
# Install PyTorch matching your CUDA first, e.g. CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pre-commit install
pytest
```

## Code style

`black` formats, `ruff` lints (line length 100). Both run on every commit through
pre-commit; run them on the whole repo with `pre-commit run --all-files`.
