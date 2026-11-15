"""Editing methods = how an editor is wrapped for one round.

baseline   plain editor (B1 with a watermark, B0 on the `none` track)
reembed    decode the payload from the round input, edit, re-embed the decoded payload into the
           output. Requires the watermark *encoder* at edit time. A strong, trivial control that any
           watermark-aware editor must be compared against.
"""
from __future__ import annotations

from PIL import Image

from .data import EditStep


class Method:
    def __init__(self, cfg: dict, editor, wm):
        self.cfg, self.editor, self.wm = cfg, editor, wm

    def run(self, img: Image.Image, step: EditStep, seed: int, state: dict) -> tuple[Image.Image, dict]:
        raise NotImplementedError


class Baseline(Method):
    def run(self, img, step, seed, state):
        return self.editor.edit(img, step, seed), {}


class ReEmbed(Method):
    def run(self, img, step, seed, state):
        if "payload" not in state or self.cfg.get("redecode_each_round", True):
            state["payload"] = self.wm.decode(img)
        out = self.editor.edit(img, step, seed)
        return self.wm.embed(out, state["payload"]), {}


def build_method(cfg: dict, editor, wm) -> Method:
    name = cfg["name"]
    if name == "baseline":
        return Baseline(cfg, editor, wm)
    if name == "reembed":
        return ReEmbed(cfg, editor, wm)
    if name.startswith("guided"):
        from .guidance import GuidedEdit

        return GuidedEdit(cfg, editor, wm)
    raise KeyError(name)
