import io
import math
import os
import tempfile
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    import ezdxf
except Exception:  # pragma: no cover - handled at runtime for environments without ezdxf
    ezdxf = None


class UnsupportedCadFormat(ValueError):
    pass


DXF_UNIT_NAMES = {
    0: "unitless",
    1: "inches",
    2: "feet",
    3: "miles",
    4: "millimetres",
    5: "centimetres",
    6: "metres",
    7: "kilometres",
    8: "microinches",
    9: "mils",
    10: "yards",
    11: "angstroms",
    12: "nanometres",
    13: "microns",
    14: "decimetres",
    15: "decametres",
    16: "hectometres",
    17: "gigametres",
    18: "astronomical_units",
    19: "light_years",
    20: "parsecs",
}


def load_image(data: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Load PNG/JPG/WebP floor-plan images.

    Upgrade:
    - Applies EXIF orientation correction.
    - Converts transparent background to white.
    - Returns clearer metadata for image/OCR workflow.
    """
    pil = Image.open(io.BytesIO(data))
    pil = ImageOps.exif_transpose(pil)

    if pil.mode in {"RGBA", "LA"}:
        background = Image.new("RGBA", pil.size, (255, 255, 255, 255))
        background.alpha_composite(pil.convert("RGBA"))
        pil = background.convert("RGB")
    else:
        pil = pil.convert("RGB")

    image = np.array(pil)
    height, width = image.shape[:2]

    return image, {
        "source_type": "raster",
        "width_px": int(width),
        "height_px": int(height),
        "units": "pixel-derived",
        "layers": [],
        "entity_count": 0,
        "entity_types": {},
        "text_entities": [],
        "notes": [
            "Raster image loaded.",
            "CAD layer intelligence is unavailable for PNG/JPG/WebP inputs.",
            "Accuracy depends on wall clarity, room labels, area text and OCR quality.",
        ],
    }


def _safe_layer(entity: Any) -> str:
    try:
        return str(getattr(entity.dxf, "layer", "0"))
    except Exception:
        return "0"


def _entity_text(entity: Any) -> str:
    try:
        dtype = entity.dxftype()
        if dtype == "TEXT":
            return str(entity.dxf.text)
        if dtype == "MTEXT":
            return str(entity.text)
        if dtype == "ATTRIB":
            return str(entity.dxf.text)
    except Exception:
        return ""
    return ""


def _polyline_points(entity: Any) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []

    try:
        if entity.dxftype() == "LWPOLYLINE":
            for point in entity.get_points():
                points.append((float(point[0]), float(point[1])))
            return points

        if entity.dxftype() == "POLYLINE":
            for vertex in entity.vertices:
                loc = vertex.dxf.location
                points.append((float(loc.x), float(loc.y)))
            return points
    except Exception:
        pass

    return points


def _entity_points(entity: Any) -> list[tuple[float, float]]:
    """
    Extract points for DXF extents.

    Upgrade:
    - Includes TEXT/MTEXT/ATTRIB insert point.
    - Tries INSERT virtual entities for block extents.
    """
    try:
        dtype = entity.dxftype()

        if dtype == "LINE":
            return [
                (float(entity.dxf.start.x), float(entity.dxf.start.y)),
                (float(entity.dxf.end.x), float(entity.dxf.end.y)),
            ]

        if dtype in {"LWPOLYLINE", "POLYLINE"}:
            return _polyline_points(entity)

        if dtype == "CIRCLE":
            center = entity.dxf.center
            radius = float(entity.dxf.radius)
            return [
                (float(center.x - radius), float(center.y - radius)),
                (float(center.x + radius), float(center.y + radius)),
            ]

        if dtype == "ARC":
            center = entity.dxf.center
            radius = float(entity.dxf.radius)
            return [
                (float(center.x - radius), float(center.y - radius)),
                (float(center.x + radius), float(center.y + radius)),
            ]

        if dtype in {"TEXT", "MTEXT", "ATTRIB"}:
            insert = entity.dxf.insert
            height = float(getattr(entity.dxf, "height", 2.5) or 2.5)
            text = _entity_text(entity)
            approx_w = max(len(text) * height * 0.55, height)
            return [
                (float(insert.x), float(insert.y)),
                (float(insert.x + approx_w), float(insert.y + height)),
            ]

        if dtype == "INSERT":
            points: list[tuple[float, float]] = []

            try:
                insert = entity.dxf.insert
                points.append((float(insert.x), float(insert.y)))
            except Exception:
                pass

            try:
                for virtual in entity.virtual_entities():
                    points.extend(_entity_points(virtual))
            except Exception:
                pass

            return points

    except Exception:
        return []

    return []


def _inspect_entities(modelspace: Any) -> tuple[dict[str, int], dict[str, int], list[tuple[float, float]], list[dict[str, Any]]]:
    layers: dict[str, int] = {}
    entity_types: dict[str, int] = {}
    points: list[tuple[float, float]] = []
    text_entities: list[dict[str, Any]] = []

    for entity in modelspace:
        try:
            dtype = entity.dxftype()
            layer = _safe_layer(entity)

            layers[layer] = layers.get(layer, 0) + 1
            entity_types[dtype] = entity_types.get(dtype, 0) + 1

            points.extend(_entity_points(entity))

            text = _entity_text(entity).strip()
            if text:
                try:
                    insert = entity.dxf.insert
                    text_entities.append(
                        {
                            "text": text,
                            "layer": layer,
                            "type": dtype,
                            "x": float(insert.x),
                            "y": float(insert.y),
                        }
                    )
                except Exception:
                    text_entities.append(
                        {
                            "text": text,
                            "layer": layer,
                            "type": dtype,
                        }
                    )

            if dtype == "INSERT":
                try:
                    for virtual in entity.virtual_entities():
                        v_text = _entity_text(virtual).strip()
                        if v_text:
                            layer = _safe_layer(virtual)
                            insert = virtual.dxf.insert
                            text_entities.append(
                                {
                                    "text": v_text,
                                    "layer": layer,
                                    "type": virtual.dxftype(),
                                    "x": float(insert.x),
                                    "y": float(insert.y),
                                    "source": "block_insert",
                                }
                            )
                except Exception:
                    pass

        except Exception:
            continue

    return layers, entity_types, points, text_entities


def inspect_dxf(data: bytes) -> dict[str, Any]:
    if ezdxf is None:
        raise RuntimeError("ezdxf is not installed. Run `pip install -r requirements.txt`.")

    path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as file:
            file.write(data)
            path = file.name

        doc = ezdxf.readfile(path)
        modelspace = doc.modelspace()

        layers, entity_types, points, text_entities = _inspect_entities(modelspace)

        return {
            "doc": doc,
            "layers": layers,
            "entity_types": entity_types,
            "points": points,
            "text_entities": text_entities,
        }

    finally:
        if path and os.path.exists(path):
            os.unlink(path)


def _get_dxf_units(doc: Any) -> dict[str, Any]:
    try:
        code = int(doc.header.get("$INSUNITS", 0))
    except Exception:
        code = 0

    return {
        "code": code,
        "name": DXF_UNIT_NAMES.get(code, "unknown"),
    }


def _layer_category(layer: str, dtype: str = "") -> str:
    text = f"{layer} {dtype}".lower()

    if any(key in text for key in ["wall", "partition", "a-wall"]):
        return "wall"

    if any(key in text for key in ["door"]):
        return "door"

    if any(key in text for key in ["window", "glaz", "glass"]):
        return "window"

    if any(key in text for key in ["fire", "sprink", "alarm", "mcp", "smoke", "heat"]):
        return "fire"

    if any(key in text for key in ["hvac", "duct", "ahu", "return", "supply"]):
        return "hvac"

    if any(key in text for key in ["elec", "power", "light", "socket", "db", "panel"]):
        return "electrical"

    if any(key in text for key in ["plumb", "toilet", "wc", "sink", "water", "drain"]):
        return "plumbing"

    if dtype in {"TEXT", "MTEXT", "ATTRIB"}:
        return "text"

    return "default"


def _draw_style(layer: str, dtype: str = "") -> tuple[tuple[int, int, int], int]:
    category = _layer_category(layer, dtype)

    if category == "wall":
        return (0, 0, 0), 3

    if category == "door":
        return (45, 45, 45), 2

    if category == "window":
        return (0, 130, 190), 2

    if category == "fire":
        return (190, 30, 30), 2

    if category == "hvac":
        return (60, 150, 90), 2

    if category == "electrical":
        return (180, 95, 20), 2

    if category == "plumbing":
        return (35, 95, 170), 2

    if category == "text":
        return (40, 40, 40), 1

    return (20, 20, 20), 2


def _make_transform(
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    target_size: int,
) -> tuple[dict[str, float], Any]:
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    span = max(span_x, span_y)

    scale = (target_size - 120) / span

    image_width = max(240, int(span_x * scale + 120))
    image_height = max(240, int(span_y * scale + 120))

    meta = {
        "scale_px_per_drawing_unit": float(scale),
        "drawing_unit_per_px": float(1 / max(scale, 0.000001)),
        "image_width": int(image_width),
        "image_height": int(image_height),
    }

    def px(point: tuple[float, float]) -> tuple[int, int]:
        return (
            int((point[0] - min_x) * scale + 60),
            int((max_y - point[1]) * scale + 60),
        )

    return meta, px


def _draw_text_entity(image: np.ndarray, entity: Any, px: Any) -> None:
    text = _entity_text(entity).strip()

    if not text:
        return

    try:
        insert = entity.dxf.insert
        x, y = px((float(insert.x), float(insert.y)))
        layer = _safe_layer(entity)
        color, _ = _draw_style(layer, entity.dxftype())

        height = float(getattr(entity.dxf, "height", 2.5) or 2.5)
        font_scale = max(0.35, min(0.9, height / 6.0))

        clean = text.replace("\\P", " ").replace("\n", " ").strip()

        if len(clean) > 60:
            clean = clean[:57] + "..."

        cv2.putText(
            image,
            clean,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            1,
            cv2.LINE_AA,
        )

    except Exception:
        return


def _draw_entity(image: np.ndarray, entity: Any, px: Any) -> None:
    try:
        dtype = entity.dxftype()
        layer = _safe_layer(entity)
        color, thickness = _draw_style(layer, dtype)

        if dtype == "LINE":
            cv2.line(
                image,
                px((float(entity.dxf.start.x), float(entity.dxf.start.y))),
                px((float(entity.dxf.end.x), float(entity.dxf.end.y))),
                color,
                thickness,
                cv2.LINE_AA,
            )

        elif dtype in {"LWPOLYLINE", "POLYLINE"}:
            points = _polyline_points(entity)

            if len(points) >= 2:
                arr = np.array([px(point) for point in points], np.int32)
                closed = bool(getattr(entity, "closed", False))

                try:
                    closed = closed or bool(entity.is_closed)
                except Exception:
                    pass

                cv2.polylines(
                    image,
                    [arr],
                    closed,
                    color,
                    thickness,
                    cv2.LINE_AA,
                )

        elif dtype == "CIRCLE":
            center = px((float(entity.dxf.center.x), float(entity.dxf.center.y)))
            # Pixel radius from transformed point difference
            edge = px((float(entity.dxf.center.x + entity.dxf.radius), float(entity.dxf.center.y)))
            radius = max(1, int(math.dist(center, edge)))

            cv2.circle(
                image,
                center,
                radius,
                color,
                max(1, thickness - 1),
                cv2.LINE_AA,
            )

        elif dtype == "ARC":
            center = px((float(entity.dxf.center.x), float(entity.dxf.center.y)))
            edge = px((float(entity.dxf.center.x + entity.dxf.radius), float(entity.dxf.center.y)))
            radius = max(1, int(math.dist(center, edge)))

            cv2.ellipse(
                image,
                center,
                (radius, radius),
                0,
                float(entity.dxf.start_angle),
                float(entity.dxf.end_angle),
                color,
                max(1, thickness - 1),
                cv2.LINE_AA,
            )

        elif dtype in {"TEXT", "MTEXT", "ATTRIB"}:
            _draw_text_entity(image, entity, px)

        elif dtype == "INSERT":
            rendered = False

            try:
                for virtual in entity.virtual_entities():
                    _draw_entity(image, virtual, px)
                    rendered = True
            except Exception:
                rendered = False

            if not rendered:
                insert = entity.dxf.insert
                x, y = px((float(insert.x), float(insert.y)))
                cv2.rectangle(
                    image,
                    (x - 5, y - 5),
                    (x + 5, y + 5),
                    color,
                    1,
                    cv2.LINE_AA,
                )

    except Exception:
        return


def dxf_to_image(data: bytes, size: int = 2000) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Render DXF into an RGB image and return rich metadata.

    Upgrade:
    - Draws TEXT/MTEXT so OCR/room-label workflow can still work.
    - Expands simple block INSERT virtual entities where possible.
    - Preserves layer names, entity counts and text entities for analysis.
    - Uses layer-aware colours for fire/HVAC/electrical/plumbing hints.
    """
    if ezdxf is None:
        raise RuntimeError("ezdxf is not installed. Run `pip install -r requirements.txt`.")

    path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as file:
            file.write(data)
            path = file.name

        doc = ezdxf.readfile(path)
        modelspace = list(doc.modelspace())

        layers, entity_types, points, text_entities = _inspect_entities(modelspace)
        units = _get_dxf_units(doc)

        if not points:
            image = np.ones((900, 900, 3), np.uint8) * 255

            return image, {
                "source_type": "dxf",
                "width_px": 900,
                "height_px": 900,
                "layers": sorted(layers),
                "entity_count": len(modelspace),
                "entity_types": entity_types,
                "units": units,
                "text_entities": text_entities,
                "notes": [
                    "DXF loaded, but no drawable plan geometry was detected.",
                    "Check that the drawing contains modelspace geometry.",
                ],
            }

        xs, ys = zip(*points)

        min_x = float(min(xs))
        max_x = float(max(xs))
        min_y = float(min(ys))
        max_y = float(max(ys))

        transform_meta, px = _make_transform(min_x, max_x, min_y, max_y, size)

        image = np.ones(
            (
                transform_meta["image_height"],
                transform_meta["image_width"],
                3,
            ),
            np.uint8,
        ) * 255

        for entity in modelspace:
            _draw_entity(image, entity, px)

        height, width = image.shape[:2]

        return image, {
            "source_type": "dxf",
            "width_px": int(width),
            "height_px": int(height),
            "layers": sorted(layers),
            "layer_counts": layers,
            "entity_count": len(modelspace),
            "entity_types": entity_types,
            "units": units,
            "text_entities": text_entities[:500],
            "drawing_extents": {
                "min_x": round(min_x, 4),
                "max_x": round(max_x, 4),
                "min_y": round(min_y, 4),
                "max_y": round(max_y, 4),
            },
            "render_transform": {
                "scale_px_per_drawing_unit": round(transform_meta["scale_px_per_drawing_unit"], 6),
                "drawing_unit_per_px": round(transform_meta["drawing_unit_per_px"], 6),
            },
            "notes": [
                "DXF geometry rendered for review-preview analysis.",
                "Text entities were rendered where possible to support room OCR.",
                "Layer names were preserved for wall/door/window/MEP intelligence.",
                "Best accuracy is achieved when rooms are closed polylines or clearly bounded by WALL layers.",
            ],
        }

    finally:
        if path and os.path.exists(path):
            os.unlink(path)


def read_drawing(data: bytes, suffix: str) -> tuple[np.ndarray, dict[str, Any]]:
    suffix = (suffix or "").lower().strip().lstrip(".")

    if suffix == "dxf":
        return dxf_to_image(data)

    if suffix == "dwg":
        raise UnsupportedCadFormat(
            "DWG upload was received, but this open-source backend cannot parse proprietary DWG files directly. "
            "Convert the drawing to DXF, or connect an ODA/AutoCAD conversion microservice before analysis."
        )

    if suffix in {"png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"}:
        return load_image(data)

    raise UnsupportedCadFormat(f"Unsupported file format: {suffix}")