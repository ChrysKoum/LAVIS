"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import sys
import logging

from omegaconf import OmegaConf

from lavis.common.registry import registry

# Wrap model/dataset/processor imports in try/except to allow graceful degradation
# when optional dependencies are missing (e.g., peft, specific transformers versions)
_import_errors = []

try:
    from lavis.datasets.builders import *
except Exception as e:
    _import_errors.append(f"datasets.builders: {e}")

try:
    from lavis.models import *
except Exception as e:
    _import_errors.append(f"models: {e}")

try:
    from lavis.processors import *
except Exception as e:
    _import_errors.append(f"processors: {e}")

try:
    from lavis.tasks import *
except Exception as e:
    _import_errors.append(f"tasks: {e}")

if _import_errors:
    logging.warning(
        "LAVIS: Some modules failed to import. This is expected if optional "
        "dependencies are not installed. Failed imports:\n  - " + 
        "\n  - ".join(_import_errors)
    )


root_dir = os.path.dirname(os.path.abspath(__file__))
default_cfg = OmegaConf.load(os.path.join(root_dir, "configs/default.yaml"))

registry.register_path("library_root", root_dir)
repo_root = os.path.join(root_dir, "..")
registry.register_path("repo_root", repo_root)
cache_root = os.path.join(repo_root, default_cfg.env.cache_root)
registry.register_path("cache_root", cache_root)

registry.register("MAX_INT", sys.maxsize)
registry.register("SPLIT_NAMES", ["train", "val", "test"])

