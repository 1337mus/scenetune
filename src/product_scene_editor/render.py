"""Original, deterministic synthetic product-scene rendering."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

SIZE = 512
SCENES = {
    "warm_limestone": ((239, 226, 204), (166, 137, 103), "warm limestone studio"),
    "garden_studio": ((210, 226, 202), (94, 132, 96), "soft green garden studio"),
    "sunset_coral": ((250, 207, 185), (176, 89, 79), "coral sunset studio"),
}
_LABELS = ("MORROW", "SORA", "ALDER", "VELA", "EMBER", "NORTH")
_FLAVORS = ("CITRUS", "BOTANICAL", "STILL", "TONIC", "MINT")
_COLORS = ((54, 115, 119), (123, 75, 94), (197, 133, 62), (74, 92, 135))


@dataclass(frozen=True)
class RenderedPair:
    source: Image.Image
    target: Image.Image
    mask: Image.Image
    bbox: tuple[int, int, int, int]
    prompt: str
    product_id: str
    target_scene: str
    seed: int


def _identity_number(product_id: str) -> int:
    return int.from_bytes(hashlib.sha256(product_id.encode("utf-8")).digest()[:8], "big")


def _background(top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    image = Image.new("RGB", (SIZE, SIZE))
    draw = ImageDraw.Draw(image)
    horizon = 386
    for y in range(SIZE):
        amount = min(1.0, y / horizon)
        color = tuple(
            round(a * (1 - amount) + b * amount) for a, b in zip(top, bottom, strict=True)
        )
        draw.line((0, y, SIZE, y), fill=color)
    draw.ellipse((74, 28, 438, 376), fill=tuple(min(255, c + 10) for c in top))
    draw.rectangle((0, horizon, SIZE, SIZE), fill=bottom)
    return image


def _foreground(
    product_id: str, seed: int
) -> tuple[Image.Image, Image.Image, tuple[int, int, int, int]]:
    rng = random.Random(seed ^ _identity_number(product_id))
    width = rng.randrange(142, 179)
    height = rng.randrange(245, 281)
    left = (SIZE - width) // 2
    top = 105 + rng.randrange(12)
    right = left + width
    bottom = top + height
    bottle = _identity_number(product_id) % 2 == 0
    body = _COLORS[_identity_number(product_id) % len(_COLORS)]
    ink = (35, 41, 40)
    mask = Image.new("L", (SIZE, SIZE), 0)
    product = Image.new("RGB", (SIZE, SIZE), (0, 0, 0))
    mask_draw = ImageDraw.Draw(mask)
    draw = ImageDraw.Draw(product)
    radius = 22 if bottle else 18
    mask_draw.rounded_rectangle((left, top, right, bottom), radius=radius, fill=255)
    draw.rounded_rectangle((left, top, right, bottom), radius=radius, fill=body)
    if bottle:
        cap_width = width // 3
        cap = (SIZE - cap_width) // 2
        mask_draw.rectangle((cap, top - 10, cap + cap_width, top + 22), fill=255)
        draw.rectangle((cap, top - 10, cap + cap_width, top + 22), fill=(225, 219, 194))
        draw.line((left + 10, top + 38, right - 10, top + 38), fill=(225, 219, 194), width=3)
    label_top = top + int(height * 0.38)
    label_bottom = top + int(height * 0.69)
    label = (left + 9, label_top, right - 9, label_bottom)
    paper = (243, 232, 203) if bottle else (238, 241, 226)
    draw.rectangle(label, fill=paper)
    draw.rectangle((left + 9, label_top, right - 9, label_top + 7), fill=body)
    label_text = _LABELS[_identity_number(product_id) % len(_LABELS)]
    flavor = _FLAVORS[(seed + _identity_number(product_id)) % len(_FLAVORS)]
    label_font = ImageFont.load_default(size=14)
    label_lines = ((label_text, ink), (flavor, body), ("SMALL BATCH", ink))
    line_tops = (label_top + 9, label_top + 30, label_top + 51)
    for (text, color), line_top in zip(label_lines, line_tops, strict=True):
        draw.text((left + 22, line_top), text, fill=color, font=label_font)
    bbox = mask.getbbox()
    assert bbox is not None
    return product, mask, bbox


def render_pair(product_id: str, seed: int, target_scene: str) -> RenderedPair:
    """Render an unchanged package over a neutral source and a scene-only target."""
    if target_scene not in SCENES:
        raise ValueError(f"unknown target scene: {target_scene}")
    foreground, mask, bbox = _foreground(product_id, seed)
    source_bg = _background((199, 218, 226), (122, 151, 165))
    scene_top, scene_bottom, scene_name = SCENES[target_scene]
    target_bg = _background(scene_top, scene_bottom)
    source = Image.composite(foreground, source_bg, mask)
    target = Image.composite(foreground, target_bg, mask)
    prompt = (
        f"Move the unchanged package into a {scene_name} with soft diffuse light; "
        "keep the package unchanged."
    )
    return RenderedPair(source, target, mask, bbox, prompt, product_id, target_scene, seed)
