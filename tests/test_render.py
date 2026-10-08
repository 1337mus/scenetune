from PIL import Image, ImageChops

from product_scene_editor import render


def test_render_pair_keeps_identical_product_foreground_and_returns_mask_bbox() -> None:
    pair = render.render_pair("pack-000", seed=41, target_scene="warm_limestone")

    assert pair.source.size == pair.target.size == pair.mask.size == (512, 512)
    assert pair.mask.mode == "L"
    assert pair.mask.getbbox() == pair.bbox
    assert all(
        pair.source.getpixel((x, y)) == pair.target.getpixel((x, y))
        for y in range(512)
        for x in range(512)
        if pair.mask.getpixel((x, y)) > 0
    )
    assert any(
        pair.source.getpixel((x, y)) != pair.target.getpixel((x, y))
        for y in range(512)
        for x in range(512)
        if pair.mask.getpixel((x, y)) == 0
    )
    assert "keep the package unchanged" in pair.prompt


def test_same_product_identity_keeps_one_foreground_across_all_target_scenes() -> None:
    pairs = [render.render_pair("pack-004", seed=77, target_scene=scene) for scene in render.SCENES]
    first = pairs[0]
    black = Image.new("RGB", first.source.size, (0, 0, 0))

    for pair in pairs[1:]:
        assert pair.bbox == first.bbox
        assert pair.mask.tobytes() == first.mask.tobytes()
        difference = ImageChops.difference(first.target, pair.target)
        assert ImageChops.composite(difference, black, first.mask).getbbox() is None
        assert ImageChops.composite(difference, black, ImageChops.invert(first.mask)).getbbox()
