from __future__ import annotations

import hashlib
import json
import stat
import zipfile
from io import BytesIO

import pytest
from PIL import Image

from product_scene_editor import protected_comparison as comparison
from product_scene_editor import remote


def _png(mode: str, color: int | tuple[int, int, int], size: tuple[int, int] = (512, 512)) -> bytes:
    output = BytesIO()
    Image.new(mode, size, color).save(output, format="PNG")
    return output.getvalue()


def _mask_png(box: tuple[int, int, int, int]) -> bytes:
    image = Image.new("L", (512, 512), 0)
    image.paste(255, box)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _fixture() -> tuple[dict[str, object], dict[str, bytes]]:
    cases: list[dict[str, object]] = []
    files: dict[str, bytes] = {}
    for index in range(24):
        case_id = f"case-{index:02d}"
        prefix = f"cases/{case_id}"
        source_path = f"{prefix}/source-rgb.png"
        alpha_path = f"{prefix}/alpha-l.png"
        protected_path = f"{prefix}/protected-l.png"
        source = _png("RGB", (index + 1, index + 30, index + 60))
        alpha = _mask_png((64 + index, 0, 320 + index, 512))
        protected = _mask_png((80 + index, 20, 160 + index, 492))
        files.update({source_path: source, alpha_path: alpha, protected_path: protected})
        raw_sha = f"{index + 1:064x}"
        cases.append(
            {
                "case_id": case_id,
                "pool": "development" if index < 6 else "sealed_test",
                "brand_id": f"brand-{index}",
                "family_id": f"family-{index}",
                "product_id": f"product-{index}",
                "source_image_id": f"image-{index}",
                "source_rgb": {"path": source_path, "sha256": hashlib.sha256(source).hexdigest()},
                "alpha_l": {"path": alpha_path, "sha256": hashlib.sha256(alpha).hexdigest()},
                "protected_l": {
                    "path": protected_path,
                    "sha256": hashlib.sha256(protected).hexdigest(),
                },
                "source_provenance": {
                    "url": (
                        "https://amazon-berkeley-objects.s3.amazonaws.com/images/original/"
                        f"{index + 1:02x}/{index + 1:08x}.jpg"
                    ),
                    "raw_source_sha256": raw_sha,
                    "metadata_sha256": "c" * 64,
                },
                "mask_review": {
                    "status": "accepted_by_root_visual_review",
                    "sha256": f"{index + 100:064x}",
                },
            }
        )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "profile": "protected-product-comparison",
        "source_sha": "e" * 40,
        "context": {
            "profile": "reflex-personal",
            "workspace": "rajath-61258",
            "environment": "main",
        },
        "modal_sdk": "1.6.1",
        "diffsynth_commit": remote.DIFFSYNTH_COMMIT,
        "runtime_pins": dict(remote.RUNTIME_PINS),
        "model_components": remote.MODEL_COMPONENTS,
        "generation": {
            "steps": 16,
            "cfg": 4,
            "width": 512,
            "height": 512,
            "seed_start": 7321,
            "seed_rule": "sorted_case_id_index",
        },
        "prompts": {
            "full_image_edit": (
                "Replace only the background with a warm beige studio backdrop and a light stone "
                "display surface. Keep the product, text, logos, shape, size, orientation and "
                "position unchanged. Do not put any object in front of the product."
            ),
            "empty_scene": (
                "An empty light stone display surface against a warm beige studio backdrop. "
                "Soft studio light from the upper left. Straight-on product photograph. Leave "
                "the center empty. No products, objects, text or logos."
            ),
        },
        "cases": cases,
    }
    return manifest, files


def _zip_bytes(entries: list[tuple[str, bytes]]) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, content in entries:
            archive.writestr(name, content)
    return output.getvalue()


def _entries(payload: bytes) -> list[tuple[str, bytes]]:
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        return [(info.filename, archive.read(info)) for info in archive.infolist()]


def _canonical_zip_bytes(entries: list[tuple[str, bytes]]) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, content in sorted(entries):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            archive.writestr(info, content)
    return output.getvalue()


def test_payload_build_is_deterministic_and_round_trips() -> None:
    manifest, files = _fixture()

    first = comparison.build_payload(manifest, files)
    second = comparison.build_payload(manifest, files)
    parsed_manifest, parsed_files = comparison.validate_payload_bytes(first)
    reversed_manifest = {**manifest, "cases": list(reversed(manifest["cases"]))}

    assert first == second
    assert comparison.build_payload(reversed_manifest, files) == first
    assert parsed_manifest == manifest
    assert parsed_files == files


def test_generation_schedule_has_48_primary_calls_and_four_fresh_load_replays() -> None:
    manifest, _ = _fixture()

    schedule = comparison.generation_schedule(manifest)

    assert len(schedule) == 52
    primary = [task for task in schedule if task["purpose"] == "primary"]
    replays = [task for task in schedule if task["purpose"] == "fresh_load_replay"]
    assert len(primary) == 48
    assert len(replays) == 4
    assert [(task["case_id"], task["arm"]) for task in primary[:2]] == [
        ("case-00", "base_full_image_edit"),
        ("case-00", "empty_scene_composite"),
    ]
    assert [task["seed"] for task in primary[:4]] == [7321, 7321, 7322, 7322]
    assert all(task["pool"] == "development" for task in replays)
    assert {task["case_id"] for task in replays} == {"case-00", "case-01"}
    assert all(task["replay_of"] is not None for task in replays)


def test_build_rejects_unknown_manifest_fields() -> None:
    manifest, files = _fixture()
    manifest["unreviewed_launcher"] = "experiments/run-everything.py"

    with pytest.raises(ValueError, match="manifest fields"):
        comparison.build_payload(manifest, files)


@pytest.mark.parametrize(("field", "value"), (("steps", True), ("cfg", 4.0), ("seed_rule", None)))
def test_build_rejects_noncanonical_generation_types(field: str, value: object) -> None:
    manifest, files = _fixture()
    manifest["generation"] = {**manifest["generation"], field: value}
    with pytest.raises(ValueError):
        comparison.build_payload(manifest, files)


def test_generation_schedule_rejects_nonstring_and_unhashable_pools() -> None:
    for pool in ([], {}, 1):
        manifest, _ = _fixture()
        manifest["cases"][0]["pool"] = pool
        with pytest.raises(ValueError):
            comparison.generation_schedule(manifest)


def test_build_rejects_changed_frozen_prompt() -> None:
    manifest, files = _fixture()
    manifest["prompts"]["empty_scene"] = str(manifest["prompts"]["empty_scene"]) + " Add a vase."

    with pytest.raises(ValueError, match="frozen prompt"):
        comparison.build_payload(manifest, files)


@pytest.mark.parametrize("group_field", ("brand_id", "family_id"))
def test_generation_schedule_rejects_groups_shared_across_pools(group_field: str) -> None:
    manifest, _ = _fixture()
    cases = list(manifest["cases"])
    cases[6] = {**cases[6], group_field: cases[0][group_field]}
    manifest["cases"] = cases

    with pytest.raises(ValueError, match="crosses development and sealed test"):
        comparison.generation_schedule(manifest)


@pytest.mark.parametrize(
    "identity_path",
    [
        ("product_id",),
        ("source_image_id",),
        ("source_rgb", "sha256"),
        ("source_provenance", "raw_source_sha256"),
    ],
)
def test_generation_schedule_rejects_duplicate_source_and_product_identity(
    identity_path: tuple[str, ...],
) -> None:
    manifest, _ = _fixture()
    cases = list(manifest["cases"])
    first = cases[0]
    duplicate = dict(cases[1])
    if len(identity_path) == 1:
        duplicate[identity_path[0]] = first[identity_path[0]]
    else:
        field, key = identity_path
        duplicate[field] = {**duplicate[field], key: first[field][key]}
    cases[1] = duplicate
    manifest["cases"] = cases

    with pytest.raises(ValueError, match="must be unique"):
        comparison.generation_schedule(manifest)


def test_build_rejects_source_png_sha_mismatch() -> None:
    manifest, files = _fixture()
    case = manifest["cases"][0]
    case["source_rgb"] = {**case["source_rgb"], "sha256": "0" * 64}

    with pytest.raises(ValueError, match="source_rgb SHA-256 mismatch"):
        comparison.build_payload(manifest, files)


def test_build_allows_shared_mask_bytes_and_alpha_equal_to_protected_mask() -> None:
    manifest, files = _fixture()
    cases = manifest["cases"]
    shared_alpha = files[cases[0]["alpha_l"]["path"]]
    second = cases[1]
    for role in ("alpha_l", "protected_l"):
        path = second[role]["path"]
        files[path] = shared_alpha
        second[role] = {"path": path, "sha256": hashlib.sha256(shared_alpha).hexdigest()}

    payload = comparison.build_payload(manifest, files)

    assert comparison.validate_payload_bytes(payload)[1] == files


def test_build_rejects_malformed_case_rows_with_value_error() -> None:
    manifest, files = _fixture()
    manifest["cases"] = [None] * 24

    with pytest.raises(ValueError):
        comparison.build_payload(manifest, files)


def test_build_rejects_invalid_source_and_mask_provenance() -> None:
    host = "amazon-berkeley-objects.s3.amazonaws.com"
    path = "/images/original/01/00000001.jpg"
    invalid_values = (
        ("source_provenance", "url", f"https://example.com{path}"),
        ("source_provenance", "url", f"https://{host}.evil.com{path}"),
        ("source_provenance", "url", f"https://{host}:443{path}"),
        ("source_provenance", "url", f"https://{host}{path}?download=1"),
        ("mask_review", "status", "draft"),
    )
    for section, field, value in invalid_values:
        manifest, files = _fixture()
        manifest["cases"][0][section] = {**manifest["cases"][0][section], field: value}
        with pytest.raises(ValueError):
            comparison.build_payload(manifest, files)


@pytest.mark.parametrize(
    ("role", "color", "size", "message"),
    (
        ("alpha_l", 255, (512, 512), "10% fully background"),
        ("protected_l", 127, (512, 512), "binary 0/255"),
    ),
)
def test_build_rejects_invalid_masks_and_non_512_assets(
    role: str, color: int | tuple[int, int, int], size: tuple[int, int], message: str
) -> None:
    manifest, files = _fixture()
    path = manifest["cases"][0][role]["path"]
    data = _png("RGB" if role == "source_rgb" else "L", color, size)
    files[path] = data
    manifest["cases"][0][role] = {"path": path, "sha256": hashlib.sha256(data).hexdigest()}

    with pytest.raises(ValueError, match=message):
        comparison.build_payload(manifest, files)


def test_all_non_512_inputs_are_rejected_by_builder_and_payload_validator() -> None:
    manifest, files = _fixture()
    for role in ("source_rgb", "alpha_l", "protected_l"):
        case = manifest["cases"][0]
        path = case[role]["path"]
        data = _png("RGB" if role == "source_rgb" else "L", 0, (511, 512))
        files[path] = data
        case[role] = {"path": path, "sha256": hashlib.sha256(data).hexdigest()}

    with pytest.raises(ValueError, match="512x512"):
        comparison.build_payload(manifest, files)

    invalid_zip = _canonical_zip_bytes(
        [
            *files.items(),
            ("manifest.json", json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()),
        ]
    )
    with pytest.raises(ValueError, match="512x512"):
        comparison.validate_payload_bytes(invalid_zip)


@pytest.mark.parametrize(
    "extra",
    (("../escape.png", b"extra"), ("cases/case-00/extra.png", b"extra"), ("manifest.json", b"{}")),
)
def test_payload_rejects_unsafe_unexpected_or_duplicate_zip_members(
    extra: tuple[str, bytes],
) -> None:
    manifest, files = _fixture()
    valid_payload = comparison.build_payload(manifest, files)
    entries = _entries(valid_payload)
    archive_bytes = _zip_bytes([*entries, extra])

    with pytest.raises(ValueError):
        comparison.validate_payload_bytes(archive_bytes)


def test_payload_rejects_symbolic_link_zip_members() -> None:
    manifest, files = _fixture()
    payload = comparison.build_payload(manifest, files)
    entries = _entries(payload)
    link = zipfile.ZipInfo("cases/linked.png")
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, content in entries:
            archive.writestr(name, content)
        archive.writestr(link, b"elsewhere")

    with pytest.raises(ValueError, match="unsafe member"):
        comparison.validate_payload_bytes(output.getvalue())


def test_payload_rejects_oversized_uncompressed_member() -> None:
    manifest, files = _fixture()
    payload = comparison.build_payload(manifest, files)
    entries = _entries(payload)
    output = _zip_bytes([*entries, ("cases/bomb.bin", b"x" * (4 * 1024 * 1024 + 1))])

    with pytest.raises(ValueError, match="member exceeds its size limit"):
        comparison.validate_payload_bytes(output)


def test_payload_rejects_duplicate_json_keys() -> None:
    manifest, files = _fixture()
    payload = comparison.build_payload(manifest, files)
    entries = _entries(payload)
    name, raw = entries[-1]
    entries[-1] = (
        name,
        raw.replace(b'"schema_version":1', b'"schema_version":1,"schema_version":1'),
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        comparison.validate_payload_bytes(_zip_bytes(entries))


def test_payload_rejects_input_over_the_32_mib_limit() -> None:
    with pytest.raises(ValueError, match="32 MiB"):
        comparison.validate_payload_bytes(b"0" * (32 * 1024 * 1024 + 1))
