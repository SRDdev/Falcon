"""Fast CPU tests for components that don't need model weights."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wmedit.data import build_plan, clean_target, decode_rle_mask, instruction_from_action  # noqa: E402
from wmedit.pipeline import load_config  # noqa: E402
from wmedit.utils import stable_seed  # noqa: E402
from wmedit.watermarks import detection_threshold, p_value  # noqa: E402


def test_threshold_controls_fpr():
    from scipy.stats import binom

    for k in (32, 100):
        m = detection_threshold(k, 1e-3)
        assert binom.sf(m - 1, k, 0.5) <= 1e-3 < binom.sf(m - 2, k, 0.5)
    assert detection_threshold(32, 1e-3) == 26
    assert p_value(32, 32) == pytest.approx(2.0 ** -32)


def test_instruction_templates():
    a = '{"motorcycle": {"position": 3, "edit_type": 1, "action": "bicycle"}, "on the road": {"position": 4, "edit_type": 3, "action": "-"}}'
    assert instruction_from_action(a) == "change the bicycle to motorcycle and remove the road"
    assert instruction_from_action('{"forest": {"position": 6, "edit_type": 8, "action": "white"}}') == \
        "change the background to forest"
    assert instruction_from_action('{"and scarf": {"position": 5, "edit_type": 2, "action": "+"}}') == "add scarf"
    two_styles = '{"baroque": {"position": 1, "edit_type": 9, "action": "+"}, "watercolor": {"position": 1, "edit_type": 9, "action": "cartoon"}}'
    assert instruction_from_action(two_styles) == "make it baroque watercolor"
    assert clean_target("a [white] raven  with [green] eyes") == "a white raven with green eyes"


def test_rle_mask():
    m = decode_rle_mask("0 3 10 2", 4, 4)
    assert m.sum() == 5 and m.flat[0] == 1 and m.flat[11] == 1 and m.flat[12] == 0


def test_plan_deterministic_and_chained():
    p1 = build_plan("x", "src", "tgt", "do it", "change_color", 5, 42)
    p2 = build_plan("x", "src", "tgt", "do it", "change_color", 5, 42)
    assert p1 == p2 and len(p1) == 5
    assert [s.category for s in p1[1:]] == ["global_attribute", "background", "style", "global_attribute"]
    for a, b in zip(p1, p1[1:]):
        assert b.source_caption == a.target_caption  # round r input caption = round r-1 output caption


def test_stable_seed():
    assert stable_seed("a", 1) == stable_seed("a", 1) != stable_seed("a", 2)


def test_config_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("name: x\ndataset: {max_images: 10}\n")
    assert load_config(p, ["dataset.max_images=3", "seed=7"]) == {"name": "x", "dataset": {"max_images": 3}, "seed": 7}


def test_dwtdct_roundtrip():
    pytest.importorskip("imwatermark")
    from PIL import Image

    from wmedit.watermarks import DWTDCT

    rng = np.random.default_rng(0)
    img = Image.fromarray((rng.random((256, 256, 3)) * 255).astype(np.uint8))
    wm, bits = DWTDCT(), rng.integers(0, 2, 32).astype(np.uint8)
    assert wm.score(wm.embed(img, bits), bits, 1e-3)["bit_acc"] > 0.9
