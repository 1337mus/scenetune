"""Draft color-connected catalog mattes; every result still needs visual review.

Low-contrast shadows connected to the border can be classified as background, and
products whose colors approach the border color can be eroded. This is not semantic
segmentation and must not be treated as accepted product geometry.
"""

from __future__ import annotations

import hashlib
from collections import deque
from io import BytesIO

from PIL import Image, ImageFilter

_WIDTH = 512
_HEIGHT = 512
_PIXELS = _WIDTH * _HEIGHT
_MAX_INPUT_BYTES = 4 * 1024 * 1024
_BORDER_TOLERANCE = 12
_BACKGROUND_TOLERANCE = 24
_MIN_BORDER_CONFORMITY = 0.98
_MIN_LIGHT_CHANNEL = 200
_MIN_FOREGROUND_FRACTION = 0.01
_MAX_FOREGROUND_FRACTION = 0.80
_MIN_MAJOR_COMPONENT_FRACTION = 0.10
_PROTECTED_EROSION_RADIUS = 2


def _decode_source(source_png: bytes) -> Image.Image:
    if not isinstance(source_png, bytes) or not source_png:
        raise ValueError("source PNG bytes are empty or invalid")
    if len(source_png) > _MAX_INPUT_BYTES:
        raise ValueError("source PNG exceeds the 4 MiB input limit")
    try:
        with Image.open(BytesIO(source_png)) as opened:
            if opened.format != "PNG" or opened.mode != "RGB" or opened.size != (_WIDTH, _HEIGHT):
                raise ValueError("source must be a 512x512 RGB PNG")
            if getattr(opened, "n_frames", 1) != 1:
                raise ValueError("source PNG must contain exactly one frame")
            opened.load()
            return opened.copy()
    except (OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ValueError("source is not a readable PNG") from exc


def _border_indices() -> list[int]:
    indices = list(range(_WIDTH))
    indices.extend((_HEIGHT - 1) * _WIDTH + x for x in range(_WIDTH))
    indices.extend(y * _WIDTH for y in range(1, _HEIGHT - 1))
    indices.extend(y * _WIDTH + _WIDTH - 1 for y in range(1, _HEIGHT - 1))
    return indices


def _robust_border_color(pixels: bytes, border: list[int]) -> tuple[int, int, int]:
    channels: list[list[int]] = [[], [], []]
    for index in border:
        offset = index * 3
        for channel in range(3):
            channels[channel].append(pixels[offset + channel])
    medians: list[int] = []
    for values in channels:
        values.sort()
        middle = len(values) // 2
        medians.append((values[middle - 1] + values[middle]) // 2)
    return medians[0], medians[1], medians[2]


def _max_channel_distance(pixels: bytes, index: int, color: tuple[int, int, int]) -> int:
    offset = index * 3
    return max(abs(pixels[offset + channel] - color[channel]) for channel in range(3))


def _neighbors(index: int) -> tuple[int, ...]:
    x, y = index % _WIDTH, index // _WIDTH
    result: list[int] = []
    if x:
        result.append(index - 1)
    if x + 1 < _WIDTH:
        result.append(index + 1)
    if y:
        result.append(index - _WIDTH)
    if y + 1 < _HEIGHT:
        result.append(index + _WIDTH)
    return tuple(result)


def _flood_border_background(
    pixels: bytes, border: list[int], color: tuple[int, int, int]
) -> bytearray:
    background = bytearray(_PIXELS)
    pending: deque[int] = deque()
    for index in border:
        if _max_channel_distance(pixels, index, color) <= _BACKGROUND_TOLERANCE:
            background[index] = 1
            pending.append(index)
    if not pending:
        raise ValueError("no border pixels match the estimated background color")
    while pending:
        index = pending.popleft()
        for neighbor in _neighbors(index):
            if background[neighbor]:
                continue
            if _max_channel_distance(pixels, neighbor, color) <= _BACKGROUND_TOLERANCE:
                background[neighbor] = 1
                pending.append(neighbor)
    return background


def _foreground_components(foreground: bytes) -> list[int]:
    visited = bytearray(_PIXELS)
    sizes: list[int] = []
    for start, value in enumerate(foreground):
        if not value or visited[start]:
            continue
        visited[start] = 1
        pending: deque[int] = deque((start,))
        size = 0
        while pending:
            index = pending.popleft()
            size += 1
            for neighbor in _neighbors(index):
                if foreground[neighbor] and not visited[neighbor]:
                    visited[neighbor] = 1
                    pending.append(neighbor)
        sizes.append(size)
    return sizes


def _encode_mask(pixels: bytes) -> bytes:
    stream = BytesIO()
    Image.frombytes("L", (_WIDTH, _HEIGHT), pixels).save(
        stream, format="PNG", compress_level=6, optimize=False
    )
    return stream.getvalue()


def draft_catalog_matte(
    source_png: bytes,
) -> tuple[bytes, bytes, dict[str, int | float | str | bool]]:
    """Return binary product alpha, eroded protected core, and unaccepted draft evidence."""
    source = _decode_source(source_png)
    rgb = source.tobytes()
    border = _border_indices()
    color = _robust_border_color(rgb, border)
    conformity = sum(
        _max_channel_distance(rgb, index, color) <= _BORDER_TOLERANCE for index in border
    ) / len(border)
    if conformity < _MIN_BORDER_CONFORMITY:
        raise ValueError("border is not sufficiently uniform around a robust border color")
    if min(color) < _MIN_LIGHT_CHANNEL:
        raise ValueError("border color is not light enough for this draft matte")

    background = _flood_border_background(rgb, border, color)
    alpha_pixels = bytes(0 if value else 255 for value in background)
    if any(alpha_pixels[index] for index in border):
        raise ValueError("foreground touches the image edge")
    foreground_count = alpha_pixels.count(255)
    foreground_fraction = foreground_count / _PIXELS
    if not _MIN_FOREGROUND_FRACTION <= foreground_fraction <= _MAX_FOREGROUND_FRACTION:
        raise ValueError("foreground area is empty, tiny, or too large for a catalog matte")

    components = sorted(_foreground_components(alpha_pixels), reverse=True)
    second_component_fraction = components[1] / foreground_count if len(components) > 1 else 0.0
    if second_component_fraction >= _MIN_MAJOR_COMPONENT_FRACTION:
        raise ValueError("foreground contains multiple major connected components")

    full_product = Image.frombytes("L", (_WIDTH, _HEIGHT), alpha_pixels)
    protected_pixels = full_product.filter(ImageFilter.MinFilter(5)).tobytes()
    protected_count = protected_pixels.count(255)
    if not protected_count or any(
        protected and not alpha
        for protected, alpha in zip(protected_pixels, alpha_pixels, strict=True)
    ):
        raise ValueError("foreground has no valid eroded protected core")

    alpha_png = _encode_mask(alpha_pixels)
    protected_png = _encode_mask(protected_pixels)
    report: dict[str, int | float | str | bool] = {
        "draft_status": "draft_unaccepted",
        "draft_requires_visual_review": True,
        "source_sha256": hashlib.sha256(source_png).hexdigest(),
        "alpha_sha256": hashlib.sha256(alpha_png).hexdigest(),
        "protected_sha256": hashlib.sha256(protected_png).hexdigest(),
        "border_color_rgb": ",".join(str(channel) for channel in color),
        "border_conformity_fraction": conformity,
        "border_pixel_count": len(border),
        "foreground_pixel_count": foreground_count,
        "foreground_fraction": foreground_fraction,
        "protected_pixel_count": protected_count,
        "protected_fraction": protected_count / _PIXELS,
        "component_count": len(components),
        "second_component_fraction": second_component_fraction,
        "border_tolerance_max_channel": _BORDER_TOLERANCE,
        "background_tolerance_max_channel": _BACKGROUND_TOLERANCE,
        "min_border_conformity_fraction": _MIN_BORDER_CONFORMITY,
        "min_light_border_channel": _MIN_LIGHT_CHANNEL,
        "min_foreground_fraction": _MIN_FOREGROUND_FRACTION,
        "max_foreground_fraction": _MAX_FOREGROUND_FRACTION,
        "min_major_component_fraction": _MIN_MAJOR_COMPONENT_FRACTION,
        "protected_erosion_radius": _PROTECTED_EROSION_RADIUS,
        "max_input_bytes": _MAX_INPUT_BYTES,
        "known_limitation_adjacent_shadow": (
            "Connected shadows and product areas within the background tolerance are removed."
        ),
    }
    return alpha_png, protected_png, report
