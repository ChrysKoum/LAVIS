# LAVIS Fork Changes - Dependency Modernization

This fork fixes installation and import issues on modern Python (3.10+/3.11+/3.12) and notebook environments (Kaggle/Colab).

## Installation

```bash
pip install git+https://github.com/ChrysKoum/LAVIS.git
```

For reproducible installs:
```bash
pip install -c constraints/constraints-py310-torch2.txt git+https://github.com/ChrysKoum/LAVIS.git
```

## Kaggle/Colab Quick Start

```python
!pip install git+https://github.com/ChrysKoum/LAVIS.git

# Smoke test
import lavis
from lavis.models import load_model_and_preprocess
print("Success!")
```

## Changes Made

### 1. Lazy Imports (A)

**`lavis/models/__init__.py`**: Complete refactor using `__getattr__` (PEP 562).
- Model classes only loaded when accessed
- `from lavis.models import load_model_and_preprocess` no longer triggers heavy imports
- Failed imports logged as warnings instead of crashing

**`lavis/__init__.py`**: Removed all star imports to prevent eager loading chain.

### 2. Transformers Compatibility (B)

**NEW: `lavis/models/compat.py`**: Compatibility shim for `apply_chunking_to_forward` and related functions.
- Works with transformers 4.28+ through 4.57+
- Falls back to local implementation if not found in either location

**Updated files**:
- `lavis/models/med.py`
- `lavis/models/blip_models/nlvr_encoder.py`
- `lavis/models/blip2_models/Qformer.py`

### 3. Optional Dependencies (C, D)

| Dependency | Before | After |
|------------|--------|-------|
| open3d | Required (0.13.0) | Optional `[3d]` extra |
| pycocoevalcap | Required at import | Lazy import in eval functions |
| peft/Vicuna | Crash if missing | Set to None if missing |

### 4. Requirements Split

| File | Purpose |
|------|---------|
| `requirements/base.txt` | Minimal core |
| `requirements/vision3d.txt` | 3D/point cloud |
| `requirements/dev.txt` | Development |
| `constraints/constraints-py310-torch2.txt` | Known-good pins |

### 5. CI Smoke Tests (E)

**NEW: `.github/workflows/smoke-test.yml`**
- Tests Python 3.10, 3.11, 3.12
- Verifies `import lavis` and `from lavis.models import load_model_and_preprocess`

## Version Changes

| Package | Before | After |
|---------|--------|-------|
| transformers | `==4.33.2` | `>=4.28.0,<5.0.0` |
| open3d | `==0.13.0` | Removed (optional) |
| timm | `==0.4.12` | `>=0.4.12` |
| opencv-python-headless | `==4.5.5.64` | No pin |

## What's NOT Changed

- Model behavior unchanged
- BLIP-2 API unchanged
- All existing code paths work the same

## Upstream

Based on: https://github.com/salesforce/LAVIS
