"""CPU-only compositing with a mechanically verified protected pixel core."""

from __future__ import annotations

import hashlib
from io import BytesIO
from typing import Literal

from PIL import Image

_MAX_INPUT_BYTES = 16 * 1024 * 1024
_MAX_SIDE = 2048
_MAX_PIXELS = 4_000_000


def _check_input_bytes(data: bytes, role: str) -> None:
    if not isinstance(data, bytes) or not data:
        raise ValueError(f"{role} PNG bytes are empty or invalid")
    if len(data) > _MAX_INPUT_BYTES:
        raise ValueError(f"{role} PNG exceeds the 16 MiB byte limit")


def _decode_png(data: bytes, *, mode: Literal["RGB", "L"], role: str) -> Image.Image:
    try:
        with Image.open(BytesIO(data)) as opened:
            width, height = opened.size
            if not width or not height or width > _MAX_SIDE or height > _MAX_SIDE:
                raise ValueError(f"{role} PNG dimensions are empty or exceed 2048 per side")
            if width * height > _MAX_PIXELS:
                raise ValueError(f"{role} PNG dimensions exceed the 4 million pixel limit")
            if opened.format != "PNG" or opened.mode != mode:
                raise ValueError(f"{role} must be a {mode} PNG")
            if getattr(opened, "n_frames", 1) != 1:
                raise ValueError(f"{role} PNG must contain exactly one frame")
            opened.load()
            return opened.copy()
    except (OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ValueError(f"{role} is not a readable PNG") from exc


def compose_protected_product(
    source_png: bytes,
    background_png: bytes,
    alpha_png: bytes,
    protected_png: bytes,
) -> tuple[bytes, dict[str, int | str]]:
    """Blend a source over a background and prove the binary protected core is unchanged.

    Hashes bind the supplied PNG bytes (and returned output bytes). Counts describe mechanical
    compositing results only; they do not establish semantic mask correctness or image quality.
    """
    raw_inputs = (
        (source_png, "source"),
        (background_png, "background"),
        (alpha_png, "alpha"),
        (protected_png, "protected"),
    )
    for data, role in raw_inputs:
        _check_input_bytes(data, role)

    source = _decode_png(source_png, mode="RGB", role="source")
    background = _decode_png(background_png, mode="RGB", role="background")
    alpha = _decode_png(alpha_png, mode="L", role="alpha")
    protected = _decode_png(protected_png, mode="L", role="protected")
    images = (source, background, alpha, protected)
    if any(image.size != source.size for image in images[1:]):
        raise ValueError("source, background, alpha, and protected PNG dimensions must match")

    alpha_pixels, protected_pixels = alpha.tobytes(), protected.tobytes()
    if any(value not in (0, 255) for value in protected_pixels):
        raise ValueError("protected mask must contain only binary 0/255 values")
    protected_count = protected_pixels.count(255)
    if not protected_count or protected_count == len(protected_pixels):
        raise ValueError("protected mask must include protected and nonprotected pixels")
    if any(
        mask and amount != 255 for mask, amount in zip(protected_pixels, alpha_pixels, strict=True)
    ):
        raise ValueError("alpha must be 255 at every protected pixel")

    composited = Image.composite(source, background, alpha)
    stream = BytesIO()
    composited.save(stream, format="PNG", compress_level=6, optimize=False)
    output_png = stream.getvalue()

    decoded_output = _decode_png(output_png, mode="RGB", role="output")
    source_pixels, output_pixels = source.tobytes(), decoded_output.tobytes()
    fully_background_count = 0
    blended_count = 0
    changed_protected_pixels = 0
    max_abs_protected_change = 0
    for index, mask_value in enumerate(protected_pixels):
        alpha_value = alpha_pixels[index]
        if alpha_value == 0:
            fully_background_count += 1
        elif alpha_value < 255:
            blended_count += 1
        if mask_value:
            offset = index * 3
            differences = tuple(
                abs(output_pixels[offset + channel] - source_pixels[offset + channel])
                for channel in range(3)
            )
            if any(differences):
                changed_protected_pixels += 1
            max_abs_protected_change = max(max_abs_protected_change, *differences)
    if changed_protected_pixels != 0:
        raise AssertionError("composited output changed pixels inside the protected core")

    report: dict[str, int | str] = {
        "source_sha256": hashlib.sha256(source_png).hexdigest(),
        "background_sha256": hashlib.sha256(background_png).hexdigest(),
        "alpha_sha256": hashlib.sha256(alpha_png).hexdigest(),
        "protected_sha256": hashlib.sha256(protected_png).hexdigest(),
        "output_sha256": hashlib.sha256(output_png).hexdigest(),
        "protected_count": protected_count,
        "changed_protected_pixels": changed_protected_pixels,
        "max_abs_protected_change": max_abs_protected_change,
        "fully_background_count": fully_background_count,
        "blended_count": blended_count,
    }
    return output_png, report
