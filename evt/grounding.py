"""Language grounding of the target: SigLIP image-text similarity between the episode instruction and each person crop.

EVT-Bench instructions describe the target's clothing ("Stay behind the man wearing a light gray suit and a blue tie").
The hybrid uses this to pick which detected person its memory learns, instead of trusting where the VLA heads in its
first steps (wrong in 18 of 68 DT pilot episodes). Same inputs as the VLA: RGB + instruction. The SigLIP checkpoint is
the one OmTrackVLA already loads (its vision tower); here the full model, text tower included, is used.
"""
from __future__ import annotations

import os
import re

import numpy as np

_VERBS = r"^(please\s+)?(follow|track|stay\s+behind|go\s+after|pursue|keep\s+up\s+with|walk\s+behind|chase|trail)\s+"


def description(instruction: str) -> str:
    """'Stay behind the man wearing a gray suit.' -> 'a photo of the man wearing a gray suit'"""
    s = (instruction or "a person").strip().rstrip(".")
    s = re.sub(_VERBS, "", s, flags=re.I)
    return "a photo of " + s


class TextGrounder:
    def __init__(self, model_dir: str, device: str | None = None):
        import torch
        from transformers import PreTrainedTokenizerFast, SiglipImageProcessor, SiglipModel
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = SiglipModel.from_pretrained(model_dir, torch_dtype=dtype).to(self.device).eval()
        # the container has no sentencepiece (slow SigLIP tokenizer): use the checkpoint's tokenizer.json directly
        self.tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_dir, "tokenizer.json"), pad_token="</s>")
        self.img_proc = SiglipImageProcessor.from_pretrained(model_dir)
        self.dtype = dtype
        self.text = None

    def set_instruction(self, instruction: str):
        text = re.sub(r"[^\w\s]", "", description(instruction).lower())     # SigLIP's own text canonicalisation
        ids = self.tok([text], padding="max_length", max_length=64, truncation=True, return_tensors="pt")["input_ids"]
        with self.torch.inference_mode():
            e = self.model.get_text_features(input_ids=ids.to(self.device))
        self.text = (e / e.norm(dim=-1, keepdim=True))[0]

    def scores(self, rgb: np.ndarray, boxes) -> np.ndarray:
        """rgb (H, W, 3) uint8, boxes [(r0, r1, c0, c1)] -> SigLIP logit per box (higher = matches the description)."""
        if self.text is None or len(boxes) == 0:
            return np.zeros(len(boxes))
        from PIL import Image
        h, w = rgb.shape[:2]
        crops = []
        for r0, r1, c0, c1 in boxes:
            pr, pc = int(0.1 * (r1 - r0)) + 2, int(0.1 * (c1 - c0)) + 2
            crops.append(Image.fromarray(np.ascontiguousarray(rgb[max(0, r0 - pr):min(h, r1 + pr), max(0, c0 - pc):min(w, c1 + pc)])))
        x = self.img_proc(images=crops, return_tensors="pt")["pixel_values"].to(self.device, self.dtype)
        with self.torch.inference_mode():
            e = self.model.get_image_features(pixel_values=x)
            e = e / e.norm(dim=-1, keepdim=True)
            logit = (e @ self.text) * self.model.logit_scale.exp() + self.model.logit_bias
        return logit.float().cpu().numpy().reshape(-1)
