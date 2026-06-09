# AGENTS.md

## Cursor Cloud specific instructions

### Product overview

RF-DETR is a **Python ML library** (not a web app). There are two developable artifacts:

| Artifact | Purpose |
|----------|---------|
| `rfdetr` package | Object detection inference, training, ONNX export |
| MkDocs site (`docs/`) | Documentation site |

### Environment setup

- **Python:** 3.9+ (CI uses 3.10; this VM uses 3.12).
- **Package manager:** [uv](https://github.com/astral-sh/uv) (matches CI). Activate the venv after install: `source .venv/bin/activate`.
- **Install (editable + docs):**
  ```bash
  uv pip install -r pyproject.toml -e . --extra docs
  uv pip install 'transformers>=4.56,<5'
  ```
  The `transformers<5` pin is required: `transformers` 5.x removed `find_pruneable_heads_and_indices`, which breaks `rfdetr.models.backbone.dinov2_with_windowed_attn`.

### Lint / test / build

| Check | Command | Notes |
|-------|---------|-------|
| Lint | N/A | No ruff/flake8/mypy configured |
| Unit tests | N/A | No `tests/` directory |
| Docs (CI) | `uv run mkdocs build --verbose` | Matches `.github/workflows/test-doc.yml` |
| Package build (CI) | `uv pip install -r pyproject.toml --extra build && uv build && uv run twine check --strict dist/*` | Matches publish workflow |
| Docs dev server | `uv run mkdocs serve` | Preview at http://127.0.0.1:8000 |

### Running the application (inference smoke test)

There is no long-running server. Core functionality is exercised via Python:

```bash
source .venv/bin/activate
python -c "
from rfdetr import RFDETRNano
from PIL import Image
import io, requests
url = 'https://media.roboflow.com/notebooks/examples/dog-2.jpeg'
img = Image.open(io.BytesIO(requests.get(url).content))
print(RFDETRNano().predict(img, threshold=0.5))
"
```

Use `RFDETRNano` on CPU (no GPU in Cloud VM). First run downloads ~350 MB weights from GCS.

### Training CLI

```bash
rfdetr --coco_dir /path/to/coco/dataset
```

Requires a local COCO-format dataset (`train/`, `valid/`, `test/` with `_annotations.coco.json`).

### Optional extras

- `rfdetr[metrics]` — TensorBoard / W&B logging
- `rfdetr[onnxexport]` — ONNX export
- `rfdetr[build]` — PyPI packaging tools

### Gotchas

- **No GPU** in Cloud VMs: inference works on CPU but is slower than documented latency numbers.
- **Network required** on first model load to download pretrained weights.
- **Roboflow API key** only needed for cloud dataset download / `deploy_to_roboflow()`, not for local inference.
