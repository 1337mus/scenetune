from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from product_scene_editor import real_photos


def _photo(
    root: Path,
    *,
    case_id: str,
    product_id: str,
    family_id: str,
    split: str,
    color: tuple[int, int, int],
    license_name: str,
    license_url: str | None,
    creator: str | None,
    rights_review: str,
) -> dict[str, object]:
    source_path = f"{split}/source/{case_id}.png"
    path = root / source_path
    path.parent.mkdir(parents=True, exist_ok=True)
    source = io.BytesIO()
    Image.new("RGB", (8, 6), color).save(source, format="PNG")
    content = source.getvalue()
    path.write_bytes(content)
    return {
        "case_id": case_id,
        "product_id": product_id,
        "family_id": family_id,
        "split": split,
        "source_path": source_path,
        "sha256": hashlib.sha256(content).hexdigest(),
        "source_url": (
            None if license_name == "owned" else f"https://images.example.test/{case_id}.png"
        ),
        "license": license_name,
        "license_url": license_url,
        "creator": creator,
        "attribution": f"Source record for {case_id}",
        "rights_review": rights_review,
    }


def _write_intake(
    root: Path,
    photos: list[dict[str, object]],
    reserved_groups: list[dict[str, str]] | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    intake_path = root / "intake.json"
    intake_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "photos": photos,
                "reserved_groups": reserved_groups or [],
            },
            allow_nan=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return intake_path


def _valid_photo(root: Path, *, case_id: str = "case", split: str = "train") -> dict[str, object]:
    return _photo(
        root,
        case_id=case_id,
        product_id=f"product-{case_id}",
        family_id=f"family-{case_id}",
        split=split,
        color=(12, 34, 56),
        license_name="owned",
        license_url=None,
        creator="Studio",
        rights_review="approved",
    )


def _rewrite_intake(root: Path, photos: list[dict[str, object]]) -> None:
    _write_intake(root, photos)


def test_verify_intake_reports_review_gaps_without_claiming_readiness(tmp_path: Path) -> None:
    photos = [
        _photo(
            tmp_path,
            case_id="case-train",
            product_id="product-train",
            family_id="family-train",
            split="train",
            color=(90, 110, 130),
            license_name="CC-BY-4.0",
            license_url="https://creativecommons.org/licenses/by/4.0/",
            creator=None,
            rights_review="pending",
        ),
        _photo(
            tmp_path,
            case_id="case-dev",
            product_id="product-dev",
            family_id="family-dev",
            split="dev",
            color=(140, 120, 100),
            license_name="owned",
            license_url=None,
            creator="Studio",
            rights_review="approved",
        ),
    ]
    _write_intake(
        tmp_path,
        photos,
        [{"product_id": "reserved-product", "family_id": "reserved-family"}],
    )

    summary = real_photos.verify_intake(tmp_path)

    assert summary["photo_count"] == 2
    assert summary["split_counts"] == {"train": 1, "dev": 1}
    assert summary["dimensions"] == {"case-train": [8, 6], "case-dev": [8, 6]}
    assert summary["pending_rights_case_ids"] == ["case-train"]
    assert summary["unresolved_creator_case_ids"] == ["case-train"]
    assert summary["ready_for_model_evaluation"] is False
    assert "training_pairs" not in summary
    assert any("near-duplicate" in reason for reason in summary["not_ready_reasons"])
    assert any(
        "masks" in reason.lower() and "text" in reason.lower()
        for reason in summary["not_ready_reasons"]
    )


@pytest.mark.parametrize(
    "raw",
    [
        '{"schema_version":1,"photos":[],"reserved_groups":[],"schema_version":1}',
        '{"schema_version":1,"photos":[],"reserved_groups":[],"extra":NaN}',
        '{"schema_version":1,"photos":[],"reserved_groups":[],"extra":Infinity}',
    ],
)
def test_rejects_duplicate_keys_and_nonfinite_json(tmp_path: Path, raw: str) -> None:
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "intake.json").write_text(raw, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key|non-finite"):
        real_photos.verify_intake(tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("split", [], "split must be train or dev"),
        ("license", {}, "unsupported license"),
        ("rights_review", [], "rights_review must be pending or approved"),
    ],
)
def test_malformed_enum_types_raise_targeted_value_error(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    photo = _valid_photo(tmp_path)
    photo[field] = value
    _rewrite_intake(tmp_path, [photo])

    with pytest.raises(ValueError, match=message):
        real_photos.verify_intake(tmp_path)


def test_external_photo_requires_source_url(tmp_path: Path) -> None:
    photo = _photo(
        tmp_path,
        case_id="external",
        product_id="product",
        family_id="family",
        split="train",
        color=(1, 2, 3),
        license_name="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        creator="Creator",
        rights_review="pending",
    )
    photo["source_url"] = None
    _rewrite_intake(tmp_path, [photo])

    with pytest.raises(ValueError, match="source_url may be null only for owned"):
        real_photos.verify_intake(tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_url", "http://example.test/photo.png", "source_url must be an HTTPS URL"),
        (
            "license_url",
            "https://user:pass@example.test/license",
            "license_url must be an HTTPS URL",
        ),
        ("license_url", None, "license_url may be null only for owned"),
        ("attribution", "  ", "attribution must be a non-empty string"),
    ],
)
def test_rejects_invalid_urls_and_attribution(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    photo = _photo(
        tmp_path,
        case_id="external",
        product_id="product",
        family_id="family",
        split="train",
        color=(1, 2, 3),
        license_name="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        creator="Creator",
        rights_review="pending",
    )
    photo[field] = value
    _rewrite_intake(tmp_path, [photo])

    with pytest.raises(ValueError, match=message):
        real_photos.verify_intake(tmp_path)


def test_rejects_source_hash_mismatch_and_invalid_image_bytes(tmp_path: Path) -> None:
    photo = _valid_photo(tmp_path)
    photo["sha256"] = "0" * 64
    _rewrite_intake(tmp_path, [photo])
    with pytest.raises(ValueError, match="source hash does not match"):
        real_photos.verify_intake(tmp_path)

    source = tmp_path / str(photo["source_path"])
    source.write_bytes(b"not an image")
    photo["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    _rewrite_intake(tmp_path, [photo])
    with pytest.raises(ValueError, match="not a readable image"):
        real_photos.verify_intake(tmp_path)


def test_rejects_duplicate_case_ids_and_photo_hashes(tmp_path: Path) -> None:
    first = _valid_photo(tmp_path, case_id="same-case")
    second = dict(first)
    second["source_path"] = "train/source/other.png"
    (tmp_path / str(second["source_path"])).write_bytes(
        (tmp_path / str(first["source_path"])).read_bytes()
    )
    _rewrite_intake(tmp_path, [first, second])
    with pytest.raises(ValueError, match="duplicate case_id"):
        real_photos.verify_intake(tmp_path)

    first = _valid_photo(tmp_path, case_id="first")
    second = _valid_photo(tmp_path, case_id="second")
    _rewrite_intake(tmp_path, [first, second])
    with pytest.raises(ValueError, match="duplicate photo hash"):
        real_photos.verify_intake(tmp_path)


@pytest.mark.parametrize("field", ["product_id", "family_id"])
def test_rejects_train_dev_group_leakage(tmp_path: Path, field: str) -> None:
    train = _valid_photo(tmp_path, case_id="train", split="train")
    dev = _valid_photo(tmp_path, case_id="dev", split="dev")
    dev[field] = train[field]
    _rewrite_intake(tmp_path, [train, dev])

    with pytest.raises(ValueError, match=f"{field} .* crosses train and dev"):
        real_photos.verify_intake(tmp_path)


@pytest.mark.parametrize("field", ["product_id", "family_id"])
def test_rejects_reserved_group_overlap(tmp_path: Path, field: str) -> None:
    photo = _valid_photo(tmp_path)
    group = {"product_id": "reserved-product", "family_id": "reserved-family"}
    group[field] = photo[field]
    _write_intake(tmp_path, [photo], [group])

    with pytest.raises(ValueError, match="overlaps a reserved product or family"):
        real_photos.verify_intake(tmp_path)


def test_rejects_malformed_reserved_groups(tmp_path: Path) -> None:
    photo = _valid_photo(tmp_path)
    _write_intake(tmp_path, [photo], [{"product_id": "one", "family_id": "two", "extra": 1}])

    with pytest.raises(ValueError, match="reserved_groups may contain only"):
        real_photos.verify_intake(tmp_path)

    _write_intake(tmp_path, [photo], [{"product_id": [], "family_id": "two"}])
    with pytest.raises(ValueError, match="reserved product_id must be a non-empty string"):
        real_photos.verify_intake(tmp_path)


def test_rejects_traversal_source_path(tmp_path: Path) -> None:
    photo = _valid_photo(tmp_path)
    photo["source_path"] = "train/source/../../sealed/photo.png"
    _rewrite_intake(tmp_path, [photo])

    with pytest.raises(ValueError, match="safe path below train/source"):
        real_photos.verify_intake(tmp_path)


def _assert_sealed_target_is_not_opened(
    monkeypatch: pytest.MonkeyPatch, sealed_file: Path
) -> list[Path]:
    opened: list[Path] = []
    original_open = Path.open

    def tracking_open(path: Path, *args: object, **kwargs: object):
        if path.resolve() == sealed_file.resolve():
            opened.append(path)
            raise AssertionError("reserved bytes must not be opened")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracking_open)
    return opened


@pytest.mark.parametrize("kind", ["file", "directory", "intake"])
def test_rejects_symlinks_to_reserved_files_before_reading_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    photo = _valid_photo(tmp_path)
    _rewrite_intake(tmp_path, [photo])
    sealed_dir = tmp_path / "sealed"
    sealed_dir.mkdir()
    sealed_file = sealed_dir / "case.png"
    sealed_file.write_bytes(b"reserved bytes")
    if kind == "file":
        source = tmp_path / str(photo["source_path"])
        source.unlink()
        source.symlink_to(sealed_file)
    elif kind == "directory":
        source_dir = tmp_path / "train/source"
        (source_dir / "case.png").unlink()
        source_dir.rmdir()
        source_dir.symlink_to(sealed_dir, target_is_directory=True)
        assert (source_dir / "case.png").resolve() == sealed_file.resolve()
    else:
        intake_path = tmp_path / "intake.json"
        intake_path.rename(sealed_dir / "intake.json")
        intake_path.symlink_to(sealed_dir / "intake.json")

    monitored_file = sealed_dir / "intake.json" if kind == "intake" else sealed_file
    opened = _assert_sealed_target_is_not_opened(monkeypatch, monitored_file)
    with pytest.raises(ValueError, match="missing or escapes"):
        real_photos.verify_intake(tmp_path)
    assert opened == []


def test_enforces_per_photo_and_total_byte_caps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(real_photos, "_MAX_PHOTO_BYTES", 100)
    monkeypatch.setattr(real_photos, "_MAX_TOTAL_PHOTO_BYTES", 100)
    first = _valid_photo(tmp_path, case_id="first")
    first_path = tmp_path / str(first["source_path"])
    first_bytes = first_path.read_bytes()
    second = _valid_photo(tmp_path, case_id="second")
    second_path = tmp_path / str(second["source_path"])
    second_bytes = second_path.read_bytes()
    assert len(first_bytes) <= 100 and len(second_bytes) <= 100

    _rewrite_intake(tmp_path, [first])
    over_limit = b"x" * 101
    first_path.write_bytes(over_limit)
    first["sha256"] = hashlib.sha256(over_limit).hexdigest()
    with pytest.raises(ValueError, match="per-photo or total size limit"):
        real_photos.verify_intake(tmp_path)

    first_path.write_bytes(first_bytes)
    first["sha256"] = hashlib.sha256(first_bytes).hexdigest()
    second_path.write_bytes(second_bytes)
    second["sha256"] = hashlib.sha256(second_bytes).hexdigest()
    _rewrite_intake(tmp_path, [first, second])
    with pytest.raises(ValueError, match="per-photo or total size limit"):
        real_photos.verify_intake(tmp_path)
