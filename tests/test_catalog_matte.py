from __future__ import annotations

import hashlib
from io import BytesIO

import pytest
from PIL import Image, ImageDraw

from product_scene_editor.catalog_matte import draft_catalog_matte


def png(image: Image.Image) -> bytes:
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def catalog_photo() -> Image.Image:
    image = Image.new("RGB", (512, 512), (248, 248, 244))
    draw = ImageDraw.Draw(image)
    draw.rectangle((120, 100, 391, 411), fill=(130, 90, 64))
    draw.rectangle((190, 190, 321, 229), fill=(245, 244, 240))
    return image


def decode_mask(data: bytes) -> bytes:
    with Image.open(BytesIO(data)) as image:
        assert image.format == "PNG"
        assert image.mode == "L"
        assert image.size == (512, 512)
        return image.tobytes()


def pixel_index(x: int, y: int) -> int:
    return y * 512 + x


def test_draft_keeps_disconnected_pale_label_and_marks_visual_review_required() -> None:
    source = png(catalog_photo())

    alpha_png, protected_png, report = draft_catalog_matte(source)

    alpha = decode_mask(alpha_png)
    protected = decode_mask(protected_png)
    assert set(alpha) == {0, 255}
    assert alpha[pixel_index(50, 50)] == 0
    assert alpha[pixel_index(150, 150)] == 255
    assert alpha[pixel_index(250, 205)] == 255
    assert protected[pixel_index(250, 205)] == 255
    assert all(value <= alpha[index] for index, value in enumerate(protected))
    assert report["draft_requires_visual_review"] is True
    assert report["border_color_rgb"] == "248,248,244"
    assert report["border_conformity_fraction"] == 1.0
    assert report["foreground_fraction"] > report["min_foreground_fraction"]
    assert report["foreground_fraction"] < report["max_foreground_fraction"]
    assert report["source_sha256"] == hashlib.sha256(source).hexdigest()
    assert report["alpha_sha256"] == hashlib.sha256(alpha_png).hexdigest()
    assert report["protected_sha256"] == hashlib.sha256(protected_png).hexdigest()
    for threshold in (
        "border_tolerance_max_channel",
        "background_tolerance_max_channel",
        "min_border_conformity_fraction",
        "min_foreground_fraction",
        "max_foreground_fraction",
        "protected_erosion_radius",
    ):
        assert threshold in report


def test_rejects_nonuniform_and_nonlight_border_samples() -> None:
    nonuniform = Image.new("RGB", (512, 512), (248, 248, 244))
    for x in range(256):
        nonuniform.putpixel((x, 0), (180, 180, 180))
    with pytest.raises(ValueError, match="uniform"):
        draft_catalog_matte(png(nonuniform))

    dark_border = Image.new("RGB", (512, 512), (170, 170, 170))
    ImageDraw.Draw(dark_border).rectangle((120, 100, 391, 411), fill=(70, 50, 40))
    with pytest.raises(ValueError, match="light"):
        draft_catalog_matte(png(dark_border))


@pytest.mark.parametrize("shape", ["empty", "tiny", "huge"])
def test_rejects_empty_tiny_or_huge_foreground(shape: str) -> None:
    image = Image.new("RGB", (512, 512), (248, 248, 244))
    if shape == "tiny":
        ImageDraw.Draw(image).rectangle((246, 246, 265, 265), fill=(80, 60, 40))
    elif shape == "huge":
        ImageDraw.Draw(image).rectangle((26, 26, 485, 485), fill=(80, 60, 40))
    with pytest.raises(ValueError, match="foreground area"):
        draft_catalog_matte(png(image))


def test_rejects_a_foreground_component_reaching_the_border() -> None:
    image = Image.new("RGB", (512, 512), (248, 248, 244))
    draw = ImageDraw.Draw(image)
    draw.rectangle((100, 100, 400, 400), fill=(80, 60, 40))
    draw.line((0, 250, 100, 250), fill=(80, 60, 40), width=1)
    with pytest.raises(ValueError, match="touches the image edge"):
        draft_catalog_matte(png(image))


def test_rejects_multiple_major_foreground_components() -> None:
    image = Image.new("RGB", (512, 512), (248, 248, 244))
    draw = ImageDraw.Draw(image)
    draw.rectangle((70, 140, 219, 369), fill=(80, 60, 40))
    draw.rectangle((292, 140, 441, 369), fill=(80, 60, 40))
    with pytest.raises(ValueError, match="multiple major"):
        draft_catalog_matte(png(image))


@pytest.mark.parametrize("kind", ["dimensions", "mode", "format", "corrupt", "oversized"])
def test_rejects_unbounded_or_mismatched_source_pngs(kind: str) -> None:
    if kind == "dimensions":
        source = png(Image.new("RGB", (511, 512), (248, 248, 244)))
    elif kind == "mode":
        source = png(Image.new("L", (512, 512), 248))
    elif kind == "format":
        stream = BytesIO()
        Image.new("RGB", (512, 512), (248, 248, 244)).save(stream, format="JPEG")
        source = stream.getvalue()
    elif kind == "oversized":
        source = b"\0" * (4 * 1024 * 1024 + 1)
    else:
        source = b"not a png"
    with pytest.raises(ValueError):
        draft_catalog_matte(source)


def test_border_connected_shadow_is_removed_and_reported_as_a_known_limit() -> None:
    image = Image.new("RGB", (512, 512), (248, 248, 244))
    draw = ImageDraw.Draw(image)
    draw.rectangle((160, 100, 391, 411), fill=(130, 90, 64))
    draw.rectangle((120, 248, 159, 271), fill=(230, 230, 230))

    alpha_png, _protected_png, report = draft_catalog_matte(png(image))

    alpha = decode_mask(alpha_png)
    assert alpha[250 * 512 + 140] == 0
    assert alpha[250 * 512 + 200] == 255
    assert "shadow" in str(report["known_limitation_adjacent_shadow"])
    assert report["draft_requires_visual_review"] is True
    assert report["draft_status"] == "draft_unaccepted"
