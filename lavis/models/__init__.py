"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import importlib
import logging
import torch
from omegaconf import OmegaConf
from lavis.common.registry import registry
from lavis.processors.base_processor import BaseProcessor

# Lazy import mapping: model class name -> (module path, class name)
# These are loaded on-demand via __getattr__ to avoid import-time failures
_LAZY_IMPORTS = {
    # Base
    "BaseModel": ("lavis.models.base_model", "BaseModel"),
    
    # ALBEF
    "AlbefClassification": ("lavis.models.albef_models.albef_classification", "AlbefClassification"),
    "AlbefFeatureExtractor": ("lavis.models.albef_models.albef_feature_extractor", "AlbefFeatureExtractor"),
    "AlbefNLVR": ("lavis.models.albef_models.albef_nlvr", "AlbefNLVR"),
    "AlbefPretrain": ("lavis.models.albef_models.albef_pretrain", "AlbefPretrain"),
    "AlbefRetrieval": ("lavis.models.albef_models.albef_retrieval", "AlbefRetrieval"),
    "AlbefVQA": ("lavis.models.albef_models.albef_vqa", "AlbefVQA"),
    
    # ALPRO
    "AlproQA": ("lavis.models.alpro_models.alpro_qa", "AlproQA"),
    "AlproRetrieval": ("lavis.models.alpro_models.alpro_retrieval", "AlproRetrieval"),
    
    # BLIP
    "BlipBase": ("lavis.models.blip_models.blip", "BlipBase"),
    "BlipCaption": ("lavis.models.blip_models.blip_caption", "BlipCaption"),
    "BlipClassification": ("lavis.models.blip_models.blip_classification", "BlipClassification"),
    "BlipFeatureExtractor": ("lavis.models.blip_models.blip_feature_extractor", "BlipFeatureExtractor"),
    "BlipITM": ("lavis.models.blip_models.blip_image_text_matching", "BlipITM"),
    "BlipNLVR": ("lavis.models.blip_models.blip_nlvr", "BlipNLVR"),
    "BlipPretrain": ("lavis.models.blip_models.blip_pretrain", "BlipPretrain"),
    "BlipRetrieval": ("lavis.models.blip_models.blip_retrieval", "BlipRetrieval"),
    "BlipVQA": ("lavis.models.blip_models.blip_vqa", "BlipVQA"),
    
    # BLIP2
    "Blip2Base": ("lavis.models.blip2_models.blip2", "Blip2Base"),
    "Blip2OPT": ("lavis.models.blip2_models.blip2_opt", "Blip2OPT"),
    "Blip2T5": ("lavis.models.blip2_models.blip2_t5", "Blip2T5"),
    "Blip2Qformer": ("lavis.models.blip2_models.blip2_qformer", "Blip2Qformer"),
    "Blip2ITM": ("lavis.models.blip2_models.blip2_image_text_matching", "Blip2ITM"),
    "Blip2T5Instruct": ("lavis.models.blip2_models.blip2_t5_instruct", "Blip2T5Instruct"),
    "Blip2VicunaInstruct": ("lavis.models.blip2_models.blip2_vicuna_instruct", "Blip2VicunaInstruct"),
    "Blip2VicunaXInstruct": ("lavis.models.blip2_models.blip2_vicuna_xinstruct", "Blip2VicunaXInstruct"),
    
    # BLIP Diffusion
    "BlipDiffusion": ("lavis.models.blip_diffusion_models.blip_diffusion", "BlipDiffusion"),
    
    # PNP / Img2Prompt
    "PNPVQA": ("lavis.models.pnp_vqa_models.pnp_vqa", "PNPVQA"),
    "PNPUnifiedQAv2FiD": ("lavis.models.pnp_vqa_models.pnp_unifiedqav2_fid", "PNPUnifiedQAv2FiD"),
    "Img2PromptVQA": ("lavis.models.img2prompt_models.img2prompt_vqa", "Img2PromptVQA"),
    
    # Other
    "XBertLMHeadDecoder": ("lavis.models.med", "XBertLMHeadDecoder"),
    "VisionTransformerEncoder": ("lavis.models.vit", "VisionTransformerEncoder"),
    "CLIP": ("lavis.models.clip_models.model", "CLIP"),
    "GPTDialogue": ("lavis.models.gpt_models.gpt_dialogue", "GPTDialogue"),
}

# Cache for lazily imported classes
_LOADED_CLASSES = {}


def __getattr__(name):
    """Lazy import for model classes (PEP 562)."""
    if name in _LAZY_IMPORTS:
        if name not in _LOADED_CLASSES:
            module_path, class_name = _LAZY_IMPORTS[name]
            try:
                module = importlib.import_module(module_path)
                _LOADED_CLASSES[name] = getattr(module, class_name)
            except (ImportError, Exception) as e:
                logging.warning(f"Failed to import {name} from {module_path}: {e}")
                _LOADED_CLASSES[name] = None
        return _LOADED_CLASSES[name]
    raise AttributeError(f"module 'lavis.models' has no attribute '{name}'")


__all__ = [
    "load_model",
    "load_preprocess",
    "load_model_and_preprocess",
    "model_zoo",
] + list(_LAZY_IMPORTS.keys())


def load_model(name, model_type, is_eval=False, device="cpu", checkpoint=None):
    """
    Load supported models.

    To list all available models and types in registry:
    >>> from lavis.models import model_zoo
    >>> print(model_zoo)

    Args:
        name (str): name of the model.
        model_type (str): type of the model.
        is_eval (bool): whether the model is in eval mode. Default: False.
        device (str): device to use. Default: "cpu".
        checkpoint (str): path or to checkpoint. Default: None.
            Note that expecting the checkpoint to have the same keys in state_dict as the model.

    Returns:
        model (torch.nn.Module): model.
    """

    model = registry.get_model_class(name).from_pretrained(model_type=model_type)

    if checkpoint is not None:
        model.load_checkpoint(checkpoint)

    if is_eval:
        model.eval()

    if device == "cpu":
        model = model.float()

    return model.to(device)


def load_preprocess(config):
    """
    Load preprocessor configs and construct preprocessors.

    If no preprocessor is specified, return BaseProcessor, which does not do any preprocessing.

    Args:
        config (dict): preprocessor configs.

    Returns:
        vis_processors (dict): preprocessors for visual inputs.
        txt_processors (dict): preprocessors for text inputs.

        Key is "train" or "eval" for processors used in training and evaluation respectively.
    """

    def _build_proc_from_cfg(cfg):
        return (
            registry.get_processor_class(cfg.name).from_config(cfg)
            if cfg is not None
            else BaseProcessor()
        )

    vis_processors = dict()
    txt_processors = dict()

    vis_proc_cfg = config.get("vis_processor")
    txt_proc_cfg = config.get("text_processor")

    if vis_proc_cfg is not None:
        vis_train_cfg = vis_proc_cfg.get("train")
        vis_eval_cfg = vis_proc_cfg.get("eval")
    else:
        vis_train_cfg = None
        vis_eval_cfg = None

    vis_processors["train"] = _build_proc_from_cfg(vis_train_cfg)
    vis_processors["eval"] = _build_proc_from_cfg(vis_eval_cfg)

    if txt_proc_cfg is not None:
        txt_train_cfg = txt_proc_cfg.get("train")
        txt_eval_cfg = txt_proc_cfg.get("eval")
    else:
        txt_train_cfg = None
        txt_eval_cfg = None

    txt_processors["train"] = _build_proc_from_cfg(txt_train_cfg)
    txt_processors["eval"] = _build_proc_from_cfg(txt_eval_cfg)

    return vis_processors, txt_processors


def load_model_and_preprocess(name, model_type, is_eval=False, device="cpu"):
    """
    Load model and its related preprocessors.

    List all available models and types in registry:
    >>> from lavis.models import model_zoo
    >>> print(model_zoo)

    Args:
        name (str): name of the model.
        model_type (str): type of the model.
        is_eval (bool): whether the model is in eval mode. Default: False.
        device (str): device to use. Default: "cpu".

    Returns:
        model (torch.nn.Module): model.
        vis_processors (dict): preprocessors for visual inputs.
        txt_processors (dict): preprocessors for text inputs.
    """
    model_cls = registry.get_model_class(name)

    # load model
    model = model_cls.from_pretrained(model_type=model_type)

    if is_eval:
        model.eval()

    # load preprocess
    cfg = OmegaConf.load(model_cls.default_config_path(model_type))
    if cfg is not None:
        preprocess_cfg = cfg.preprocess

        vis_processors, txt_processors = load_preprocess(preprocess_cfg)
    else:
        vis_processors, txt_processors = None, None
        logging.info(
            f"""No default preprocess for model {name} ({model_type}).
                This can happen if the model is not finetuned on downstream datasets,
                or it is not intended for direct use without finetuning.
            """
        )

    if device == "cpu" or device == torch.device("cpu"):
        model = model.float()

    return model.to(device), vis_processors, txt_processors


class ModelZoo:
    """
    A utility class to create string representation of available model architectures and types.

    >>> from lavis.models import model_zoo
    >>> # list all available models
    >>> print(model_zoo)
    >>> # show total number of models
    >>> print(len(model_zoo))
    """

    def __init__(self) -> None:
        self.model_zoo = {
            k: list(v.PRETRAINED_MODEL_CONFIG_DICT.keys())
            for k, v in registry.mapping["model_name_mapping"].items()
        }

    def __str__(self) -> str:
        return (
            "=" * 50
            + "\n"
            + f"{'Architectures':<30} {'Types'}\n"
            + "=" * 50
            + "\n"
            + "\n".join(
                [
                    f"{name:<30} {', '.join(types)}"
                    for name, types in self.model_zoo.items()
                ]
            )
        )

    def __iter__(self):
        return iter(self.model_zoo.items())

    def __len__(self):
        return sum([len(v) for v in self.model_zoo.values()])


model_zoo = ModelZoo()
