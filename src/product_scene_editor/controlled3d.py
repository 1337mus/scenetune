"""Registry-backed validation and explicit train-only export API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from product_scene_editor.controlled3d_assets import (
    export_train_only as _export_train_only,
)
from product_scene_editor.controlled3d_assets import (
    root_identity as _root_identity,
)
from product_scene_editor.controlled3d_assets import (
    validate_assets as _validate_assets,
)
from product_scene_editor.controlled3d_schema import (
    _DENIED_FAMILIES,
    _DENIED_PAIR_IDS,
    _DENIED_PRODUCTS,
    _MAX_JSON_BYTES,
    _STATES,
    _check_json,
    _digest,
    _dump,
    _fields,
    _id,
    _loads_any,
    _manifest_shape,
    _pins,
    _ref,
    _sha,
    _split,
)


def _registry(
    value: Any,
) -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, dict[str, Any]]]:
    registry = _fields(
        value, {"schema_version", "products", "families", "pairs"}, "reviewed registry"
    )
    if type(registry["schema_version"]) is not int or registry["schema_version"] != 1:
        raise ValueError("registry schema_version must be integer 1")
    if any(
        type(registry[key]) is not list or not registry[key]
        for key in ("products", "families", "pairs")
    ):
        raise ValueError("registry products, families, and pairs must be non-empty lists")
    products: dict[str, str] = {}
    for row in registry["products"]:
        item = _fields(row, {"product_id", "split"}, "registry product")
        product, split = _id(item["product_id"], "product ID"), _split(item["split"])
        if product in _DENIED_PRODUCTS:
            raise ValueError(f"excluded product: {product}")
        if product in products:
            raise ValueError(f"duplicate product: {product}")
        products[product] = split
    families: dict[str, str] = {}
    family_splits: dict[str, str] = {}
    canonical_ids: set[str] = set()
    alias_ids: set[str] = set()
    for row in registry["families"]:
        item = _fields(row, {"canonical_id", "split", "aliases"}, "registry family")
        canonical, split = _id(item["canonical_id"], "canonical family ID"), _split(item["split"])
        if canonical in _DENIED_FAMILIES:
            raise ValueError(f"excluded family: {canonical}")
        if canonical in canonical_ids or type(item["aliases"]) is not list:
            raise ValueError("duplicate family or aliases must be a list")
        canonical_ids.add(canonical)
        families[canonical], family_splits[canonical] = canonical, split
        for raw_alias in item["aliases"]:
            alias = _id(raw_alias, "family alias")
            if alias in _DENIED_FAMILIES:
                raise ValueError(f"excluded family: {alias}")
            if alias == canonical or alias in families:
                raise ValueError(f"duplicate or chained family alias: {alias}")
            families[alias] = canonical
            alias_ids.add(alias)
    if canonical_ids.intersection(alias_ids):
        raise ValueError("family aliases must bind directly to canonical families")
    pairs: dict[str, dict[str, Any]] = {}
    for row in registry["pairs"]:
        item = _fields(
            row,
            {"pair_id", "product_id", "background_family_id", "split", "expected_words", "pins"},
            "registry pair",
        )
        pair_id = _id(item["pair_id"], "pair ID")
        if pair_id in _DENIED_PAIR_IDS:
            raise ValueError(f"excluded pair: {pair_id}")
        product = _id(item["product_id"], "product ID")
        family = _id(item["background_family_id"], "family ID")
        split = _split(item["split"])
        if pair_id in pairs:
            raise ValueError(f"duplicate pair ID: {pair_id}")
        if (
            product not in products
            or family not in families
            or products[product] != split
            or family_splits[families.get(family, "")] != split
        ):
            raise ValueError(f"registry pair assignment mismatch: {pair_id}")
        if (
            type(item["expected_words"]) is not list
            or not item["expected_words"]
            or any(type(word) is not str or not word for word in item["expected_words"])
        ):
            raise ValueError("registry expected_words must be non-empty strings")
        pairs[pair_id] = item
    return products, families, family_splits, pairs


def _assignments(
    manifest: dict[str, Any],
    products: dict[str, str],
    families: dict[str, str],
    family_splits: dict[str, str],
    registry_pairs: dict[str, dict[str, Any]],
) -> None:
    assigned_products: dict[str, str] = {}
    assigned_families: dict[str, str] = {}
    for row in manifest["assignments"]:
        ident, split = row["id"], row["split"]
        if row["kind"] == "product":
            if ident in _DENIED_PRODUCTS:
                raise ValueError(f"excluded product: {ident}")
            if ident not in products or products[ident] != split or ident in assigned_products:
                raise ValueError(
                    f"missing, unknown, duplicate, or cross-split product assignment: {ident}"
                )
            assigned_products[ident] = split
        else:
            canonical = families.get(ident)
            if ident in _DENIED_FAMILIES:
                raise ValueError(f"excluded family: {ident}")
            if (
                canonical is None
                or family_splits[canonical] != split
                or canonical in assigned_families
            ):
                raise ValueError(
                    f"missing, unknown, duplicate canonical family assignment: {ident}"
                )
            assigned_families[canonical] = split
    if assigned_products != products or set(assigned_families) != set(family_splits):
        raise ValueError("manifest assignments are missing or unknown")
    rows = manifest["pairs"]
    if {row["pair_id"] for row in rows} != set(registry_pairs) or len(rows) != len(registry_pairs):
        raise ValueError("manifest pair IDs do not exactly match the reviewed registry")
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    splits: dict[str, set[str]] = {}
    for row in rows:
        approved = registry_pairs[row["pair_id"]]
        for key in ("product_id", "background_family_id", "split", "expected_words", "pins"):
            if row[key] != approved[key]:
                raise ValueError(f"pair differs from reviewed registry: {row['pair_id']} ({key})")
        product, family, split = (
            row["product_id"],
            families.get(row["background_family_id"]),
            row["split"],
        )
        if (
            family is None
            or assigned_products.get(product) != split
            or assigned_families.get(family) != split
        ):
            raise ValueError(f"pair endpoint split disagreement: {row['pair_id']}")
        left, right = find("p:" + product), find("f:" + family)
        parent[left] = right
        splits.setdefault(left, set()).add(split)
        splits.setdefault(right, set()).add(split)
    components: dict[str, set[str]] = {}
    for node, pool_splits in splits.items():
        components.setdefault(find(node), set()).update(pool_splits)
    if any(len(pool_splits) != 1 for pool_splits in components.values()):
        raise ValueError("connected product-family component crosses splits")


def _asset_refs(
    manifest: dict[str, Any],
) -> tuple[list[tuple[str, str, str, str]], dict[str, list[dict[str, Any]]]]:
    used: set[str] = set()
    refs: list[tuple[str, str, str, str]] = []
    pair_rows: dict[str, list[dict[str, Any]]] = {}
    for pair in manifest["pairs"]:
        split, pair_id = pair["split"], pair["pair_id"]
        _pins(pair["pins"], split, used)
        for row in pair["pins"]["local_files"]:
            refs.append((split, row["path"], row["sha256"], "pin"))
        snap = _ref(pair["snapshot_bundle"], split, {"path", "sha256"}, "snapshot bundle", used)
        refs.append((split, snap["path"], snap["sha256"], "snapshot"))
        by_name: dict[str, dict[str, Any]] = {}
        for raw in pair["renders"]:
            render = _fields(raw, {"name", "image", "mask", "bbox"}, "render")
            name = render["name"]
            if type(name) is not str or name not in _STATES or name in by_name:
                raise ValueError("renders must name source, cream, sage, and clay exactly once")
            image = _ref(render["image"], split, {"path", "sha256"}, "image", used)
            mask = _fields(render["mask"], {"path", "sha256", "decoded_sha256"}, "mask")
            _digest(mask["decoded_sha256"], "decoded mask sha256")
            mask = _ref(
                {"path": mask["path"], "sha256": mask["sha256"]},
                split,
                {"path", "sha256"},
                "mask",
                used,
            )
            bbox = render["bbox"]
            if (
                type(bbox) is not list
                or len(bbox) != 4
                or any(type(v) is not int or not 0 <= v <= 512 for v in bbox)
            ):
                raise ValueError("mask bbox must be four image-boundary integers")
            refs.extend(
                (
                    (split, image["path"], image["sha256"], "image"),
                    (split, mask["path"], mask["sha256"], "mask"),
                )
            )
            by_name[name] = {
                **render,
                "image": image,
                "mask": {**mask, "decoded_sha256": render["mask"]["decoded_sha256"]},
            }
        if set(by_name) != set(_STATES):
            raise ValueError("renders must name source, cream, sage, and clay exactly once")
        pair_rows[pair_id] = [by_name[name] for name in _STATES]
    return refs, pair_rows


def draft_manifest(path: str | Path) -> dict[str, Any]:
    """Parse bounded strict JSON only; referenced assets are never inspected."""
    with Path(path).open("rb") as file:
        raw = file.read(_MAX_JSON_BYTES + 1)
    value = _loads_any(raw)
    if type(value) is not dict:
        raise ValueError("manifest root must be an object")
    return _manifest_shape(value)


@dataclass(frozen=True)
class ValidatedControlled3D:
    """Immutable canonical inputs for explicit train-only export."""

    canonical_manifest: str
    canonical_registry: str
    registry_sha256: str
    root_identity: tuple[str, int, int]
    metrics_json: str


def validate_controlled3d(
    root: str | Path,
    manifest: dict[str, Any],
    *,
    reviewed_registry: bytes | str,
    registry_sha256: str,
) -> ValidatedControlled3D:
    """Validate frozen identity, provenance, captured state, and train/dev PNG assets."""
    if type(manifest) is not dict:
        raise ValueError("manifest must be an object")
    _check_json(manifest)
    manifest = _manifest_shape(manifest)
    if type(reviewed_registry) is bytes:
        registry_raw = reviewed_registry.decode("utf-8")
    elif type(reviewed_registry) is str:
        registry_raw = reviewed_registry
    else:
        raise ValueError("reviewed registry must be bytes or text")
    _digest(registry_sha256, "registry digest")
    if _sha(registry_raw.encode("utf-8")) != registry_sha256:
        raise ValueError("reviewed registry digest mismatch")
    registry = _loads_any(registry_raw)
    products, families, family_splits, registry_pairs = _registry(registry)
    _assignments(manifest, products, families, family_splits, registry_pairs)
    refs, pair_rows = _asset_refs(manifest)
    identity = _root_identity(root)
    metrics = _validate_assets(root, manifest, refs, pair_rows, _loads_any)
    return ValidatedControlled3D(
        _dump(manifest), registry_raw, registry_sha256, identity, _dump(metrics)
    )


def export_train_only(
    root: str | Path, validated_manifest: ValidatedControlled3D, destination: str | Path
) -> Path:
    """Revalidate and publish train-only assets to a new destination."""
    if not isinstance(validated_manifest, ValidatedControlled3D):
        raise ValueError("export requires a validated Controlled3D result")
    manifest = _loads_any(validated_manifest.canonical_manifest)
    return _export_train_only(
        root, validated_manifest, destination, manifest, validate_controlled3d, _dump
    )
