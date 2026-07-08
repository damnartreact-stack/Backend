from __future__ import annotations

"""
FireDesign / Ceasefire DXF writer.

Purpose
-------
Creates an updated DXF drawing that contains generated fire-safety and MEP
devices on clean CEASEFIRE_* layers.

This is a POC CAD export module. It is designed to be robust with the current
analysis.py output schema and with future richer schemas.

Main features
-------------
1. Preserves original DXF geometry if the uploaded file is DXF.
2. Creates a proof DXF for raster uploads.
3. Adds CEASEFIRE layers for:
   - devices
   - routes
   - text
   - rooms
   - review notes
4. Creates reusable block symbols for fire devices.
5. Inserts devices with labels and room IDs.
6. Inserts full polyline routes, not only start/end lines.
7. Adds room boxes if the source is raster or if a room overlay is useful.
8. Adds drawing legend and POC review stamp.
9. Returns a base64 DXF payload for frontend download.

Place this file at:
    FIRE_ALARM/backend/dxf_writer.py
"""

import base64
import math
import tempfile
from pathlib import Path
from typing import Any

try:
    import ezdxf
except Exception:  # pragma: no cover
    ezdxf = None


# ---------------------------------------------------------------------------
# Layer names
# ---------------------------------------------------------------------------

LAYER_BORDER = "POC_PLAN_BORDER"
LAYER_ROOMS = "CEASEFIRE_ROOM_REVIEW"
LAYER_FIRE_DEVICE = "CEASEFIRE_FIRE_SAFETY"
LAYER_FIRE_ALARM = "CEASEFIRE_FIRE_ALARM"
LAYER_SPRINKLER = "CEASEFIRE_SPRINKLER"
LAYER_HYDRANT = "CEASEFIRE_HYDRANT"
LAYER_HVAC = "CEASEFIRE_HVAC_REVIEW"
LAYER_ELECTRICAL = "CEASEFIRE_ELECTRICAL_REVIEW"
LAYER_PLUMBING = "CEASEFIRE_PLUMBING_REVIEW"
LAYER_TEXT = "CEASEFIRE_TEXT"
LAYER_ROUTE = "CEASEFIRE_ROUTES"
LAYER_REVIEW = "CEASEFIRE_REVIEW"

DEVICE_BLOCK_PREFIX = "CF_"


# ---------------------------------------------------------------------------
# Device mapping
# ---------------------------------------------------------------------------

DEVICE_TYPE_ALIASES: dict[str, str] = {
    # Sprinkler
    "SP": "SPRINKLER_HEAD",
    "SPRINKLER": "SPRINKLER_HEAD",
    "SPRINKLER_HEAD": "SPRINKLER_HEAD",
    "FIRE_SPRINKLER": "SPRINKLER_HEAD",
    "AUTOMATIC_SPRINKLER_HEAD": "SPRINKLER_HEAD",

    "RISER": "FIRE_RISER",
    "FIRE_RISER": "FIRE_RISER",
    "VALVE": "CONTROL_VALVE",
    "CONTROL_VALVE": "CONTROL_VALVE",
    "FLOW_SWITCH": "FLOW_SWITCH",
    "TAMPER_SWITCH": "TAMPER_SWITCH",
    "CHECK_VALVE": "CHECK_VALVE",
    "DRAIN_VALVE": "DRAIN_VALVE",
    "TEST_VALVE": "INSPECTOR_TEST_VALVE",
    "INSPECTOR_TEST_VALVE": "INSPECTOR_TEST_VALVE",
    "PRESSURE_GAUGE": "PRESSURE_GAUGE",
    "ALARM_VALVE": "ALARM_VALVE",
    "FDC": "FIRE_DEPARTMENT_CONNECTION",

    # Fire alarm
    "SD": "SMOKE_DETECTOR",
    "SMOKE": "SMOKE_DETECTOR",
    "SMOKE_DETECTOR": "SMOKE_DETECTOR",
    "HD": "HEAT_DETECTOR",
    "HEAT": "HEAT_DETECTOR",
    "HEAT_DETECTOR": "HEAT_DETECTOR",
    "MCP": "MANUAL_CALL_POINT",
    "MANUAL_CALL_POINT": "MANUAL_CALL_POINT",
    "CALL_POINT": "MANUAL_CALL_POINT",
    "HS": "HORN_STROBE",
    "HORN": "HORN_STROBE",
    "STROBE": "HORN_STROBE",
    "HORN_STROBE": "HORN_STROBE",
    "SOUNDER": "HORN_STROBE",
    "FACP": "FIRE_ALARM_CONTROL_PANEL",
    "PANEL": "FIRE_ALARM_CONTROL_PANEL",
    "FIRE_ALARM_PANEL": "FIRE_ALARM_CONTROL_PANEL",
    "ANNUNCIATOR": "REMOTE_ANNUNCIATOR",

    # Portable/egress/hydrant
    "EXT": "FIRE_EXTINGUISHER",
    "EXTINGUISHER": "FIRE_EXTINGUISHER",
    "FIRE_EXTINGUISHER": "FIRE_EXTINGUISHER",
    "SIGN": "EXIT_SIGNAGE",
    "EXIT_SIGN": "EXIT_SIGNAGE",
    "EXIT_SIGNAGE": "EXIT_SIGNAGE",
    "EM_LIGHT": "EMERGENCY_LIGHT",
    "EMERGENCY_LIGHT": "EMERGENCY_LIGHT",
    "HYDRANT": "FIRE_HYDRANT",
    "FIRE_HYDRANT": "FIRE_HYDRANT",
    "HOSE_REEL": "HOSE_REEL",
    "LANDING_VALVE": "LANDING_VALVE",

    # MEP placeholders
    "AHU": "AHU",
    "SDIFF": "SUPPLY_DIFFUSER",
    "RGRILLE": "RETURN_GRILLE",
    "EXH": "EXHAUST_GRILLE",
    "EXFAN": "EXHAUST_GRILLE",
    "EDB": "ELECTRICAL_DISTRIBUTION_BOARD",
    "MDB": "MAIN_DISTRIBUTION_BOARD",
    "LIGHT": "LIGHT_FIXTURE",
    "SWITCH": "LIGHT_SWITCH",
    "SO": "POWER_SOCKET",
    "POWER_SOCKET": "POWER_SOCKET",
    "PLUMBING_RISER": "PLUMBING_RISER",
    "LAV": "LAVATORY",
    "SINK": "SINK",
    "WC": "WC_FIXTURE",
    "FD": "FLOOR_DRAIN",
}


BLOCK_LABELS: dict[str, str] = {
    "SPRINKLER_HEAD": "SP",
    "FIRE_RISER": "RISER",
    "CONTROL_VALVE": "VLV",
    "FLOW_SWITCH": "FS",
    "TAMPER_SWITCH": "TS",
    "CHECK_VALVE": "CV",
    "INSPECTOR_TEST_VALVE": "ITV",
    "PRESSURE_GAUGE": "PG",
    "FIRE_DEPARTMENT_CONNECTION": "FDC",
    "DRAIN_VALVE": "DV",
    "ALARM_VALVE": "AV",

    "SMOKE_DETECTOR": "SD",
    "HEAT_DETECTOR": "HD",
    "MANUAL_CALL_POINT": "MCP",
    "HORN_STROBE": "HS",
    "FIRE_ALARM_CONTROL_PANEL": "FACP",
    "REMOTE_ANNUNCIATOR": "ANN",

    "FIRE_EXTINGUISHER": "EXT",
    "EXIT_SIGNAGE": "EXIT",
    "EMERGENCY_LIGHT": "EM",
    "FIRE_HYDRANT": "HYD",
    "HOSE_REEL": "HR",
    "LANDING_VALVE": "LV",

    "AHU": "AHU",
    "SUPPLY_DIFFUSER": "SDIF",
    "RETURN_GRILLE": "RG",
    "EXHAUST_GRILLE": "EXH",
    "ELECTRICAL_DISTRIBUTION_BOARD": "EDB",
    "MAIN_DISTRIBUTION_BOARD": "MDB",
    "LIGHT_FIXTURE": "LGT",
    "LIGHT_SWITCH": "SW",
    "POWER_SOCKET": "SO",
    "PLUMBING_RISER": "PR",
    "LAVATORY": "LAV",
    "SINK": "SINK",
    "WC_FIXTURE": "WC",
    "FLOOR_DRAIN": "FD",

    "UNKNOWN": "?",
}


DEVICE_LAYER_MAP: dict[str, str] = {
    "SPRINKLER_HEAD": LAYER_SPRINKLER,
    "FIRE_RISER": LAYER_SPRINKLER,
    "CONTROL_VALVE": LAYER_SPRINKLER,
    "FLOW_SWITCH": LAYER_SPRINKLER,
    "TAMPER_SWITCH": LAYER_SPRINKLER,
    "CHECK_VALVE": LAYER_SPRINKLER,
    "DRAIN_VALVE": LAYER_SPRINKLER,
    "ALARM_VALVE": LAYER_SPRINKLER,
    "INSPECTOR_TEST_VALVE": LAYER_SPRINKLER,
    "PRESSURE_GAUGE": LAYER_SPRINKLER,
    "FIRE_DEPARTMENT_CONNECTION": LAYER_SPRINKLER,

    "SMOKE_DETECTOR": LAYER_FIRE_ALARM,
    "HEAT_DETECTOR": LAYER_FIRE_ALARM,
    "MANUAL_CALL_POINT": LAYER_FIRE_ALARM,
    "HORN_STROBE": LAYER_FIRE_ALARM,
    "FIRE_ALARM_CONTROL_PANEL": LAYER_FIRE_ALARM,
    "REMOTE_ANNUNCIATOR": LAYER_FIRE_ALARM,

    "FIRE_EXTINGUISHER": LAYER_FIRE_DEVICE,
    "EXIT_SIGNAGE": LAYER_FIRE_DEVICE,
    "EMERGENCY_LIGHT": LAYER_FIRE_DEVICE,
    "FIRE_HYDRANT": LAYER_HYDRANT,
    "HOSE_REEL": LAYER_HYDRANT,
    "LANDING_VALVE": LAYER_HYDRANT,

    "AHU": LAYER_HVAC,
    "SUPPLY_DIFFUSER": LAYER_HVAC,
    "RETURN_GRILLE": LAYER_HVAC,
    "EXHAUST_GRILLE": LAYER_HVAC,

    "ELECTRICAL_DISTRIBUTION_BOARD": LAYER_ELECTRICAL,
    "MAIN_DISTRIBUTION_BOARD": LAYER_ELECTRICAL,
    "LIGHT_FIXTURE": LAYER_ELECTRICAL,
    "LIGHT_SWITCH": LAYER_ELECTRICAL,
    "POWER_SOCKET": LAYER_ELECTRICAL,

    "PLUMBING_RISER": LAYER_PLUMBING,
    "LAVATORY": LAYER_PLUMBING,
    "SINK": LAYER_PLUMBING,
    "WC_FIXTURE": LAYER_PLUMBING,
    "FLOOR_DRAIN": LAYER_PLUMBING,
}


ROUTE_LAYER_MAP: dict[str, str] = {
    "MAIN": LAYER_SPRINKLER,
    "BRANCH": LAYER_SPRINKLER,
    "DROP": LAYER_SPRINKLER,
    "SLC": LAYER_FIRE_ALARM,
    "NAC": LAYER_FIRE_ALARM,
    "DUCT_MAIN": LAYER_HVAC,
    "DUCT_BRANCH": LAYER_HVAC,
    "RETURN_DUCT": LAYER_HVAC,
    "EXHAUST_DUCT": LAYER_HVAC,
    "LIGHTING_CIRCUIT": LAYER_ELECTRICAL,
    "POWER_CIRCUIT": LAYER_ELECTRICAL,
    "EMERGENCY_CIRCUIT": LAYER_ELECTRICAL,
    "WATER_PIPE": LAYER_PLUMBING,
    "DRAIN_PIPE": LAYER_PLUMBING,
    "VENT_PIPE": LAYER_PLUMBING,
}


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def _require_ezdxf() -> None:
    if ezdxf is None:
        raise RuntimeError("ezdxf is not installed. Install it with: pip install ezdxf")


def _normalise(value: Any) -> str:
    if value is None:
        return "UNKNOWN"

    text = str(value).strip().upper()
    text = text.replace("-", "_").replace("/", "_").replace(" ", "_")
    text = "".join(ch for ch in text if ch.isalnum() or ch == "_")

    while "__" in text:
        text = text.replace("__", "_")

    return text or "UNKNOWN"


def canonical_device_type(value: Any) -> str:
    key = _normalise(value)
    return DEVICE_TYPE_ALIASES.get(key, key if key in BLOCK_LABELS else "UNKNOWN")


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        if value is None:
            return fallback
        if isinstance(value, str):
            value = value.replace(",", "").strip()
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return fallback
        return number
    except (TypeError, ValueError):
        return fallback


def _get_case_insensitive(row: dict[str, Any], possible_keys: list[str], default: Any = None) -> Any:
    if not isinstance(row, dict):
        return default

    lower_map = {str(key).lower(): key for key in row.keys()}

    for key in possible_keys:
        actual_key = lower_map.get(str(key).lower())
        if actual_key is not None:
            value = row.get(actual_key)
            if value not in (None, ""):
                return value

    return default


def _safe_text(value: Any, fallback: str = "") -> str:
    text = str(value if value is not None else fallback).strip()
    return text or fallback


def _format_number(value: Any, digits: int = 2) -> str:
    number = _safe_float(value, 0.0)
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def _add_polyline(
    layout: Any,
    points: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    dxfattribs: dict[str, Any] | None = None,
    close: bool = False,
) -> Any:
    """
    Add a polyline safely across old and new DXF versions.

    Some customer DXF files may be saved in old DXF releases such as R12.
    R12 does not support LWPOLYLINE, so direct layout.add_lwpolyline()
    can fail with:

        LWPOLYLINE requires DXF R2000

    This helper first tries LWPOLYLINE. If the loaded DXF version does not
    support it, it falls back to ordinary LINE segments, which are compatible
    with older DXF versions.
    """
    attrs = dict(dxfattribs or {})
    clean_points = [(float(p[0]), float(p[1])) for p in points if len(p) >= 2]

    if len(clean_points) < 2:
        return None

    if close and clean_points[0] != clean_points[-1]:
        clean_points.append(clean_points[0])

    try:
        return layout.add_lwpolyline(clean_points, dxfattribs=attrs)
    except Exception as exc:
        message = str(exc).lower()

        # Fallback only for DXF-version/polyline support problems.
        # For other issues, keep backend alive by drawing line segments too.
        if "lwpolyline" not in message and "r2000" not in message and "dxf" not in message:
            pass

        entities = []
        for start, end in zip(clean_points, clean_points[1:]):
            try:
                entities.append(layout.add_line(start, end, dxfattribs=attrs))
            except Exception:
                continue

        return entities[-1] if entities else None


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

def extract_device_type(device: dict[str, Any]) -> str:
    for key in ["device_type", "type", "symbol", "name", "label", "item", "description"]:
        value = device.get(key)
        if value:
            canonical = canonical_device_type(value)
            if canonical != "UNKNOWN":
                return canonical

    return "UNKNOWN"


def extract_xy(item: dict[str, Any]) -> tuple[float, float]:
    direct_pairs = [
        ("x", "y"),
        ("cx", "cy"),
        ("center_x", "center_y"),
        ("pixel_x", "pixel_y"),
    ]

    for x_key, y_key in direct_pairs:
        if x_key in item and y_key in item:
            return _safe_float(item.get(x_key)), _safe_float(item.get(y_key))

    for parent in ["position", "location", "point", "center"]:
        value = item.get(parent)

        if isinstance(value, dict) and "x" in value and "y" in value:
            return _safe_float(value.get("x")), _safe_float(value.get("y"))

        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return _safe_float(value[0]), _safe_float(value[1])

    value = item.get("coordinates")
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return _safe_float(value[0]), _safe_float(value[1])

    geometry = item.get("geometry")
    if isinstance(geometry, dict) and "x" in geometry and "y" in geometry:
        return _safe_float(geometry.get("x")), _safe_float(geometry.get("y"))

    return 0.0, 0.0


def extract_route_polyline(route: dict[str, Any]) -> list[tuple[float, float]]:
    if not isinstance(route, dict):
        return []

    points = route.get("points")
    if isinstance(points, list) and len(points) >= 2:
        output: list[tuple[float, float]] = []

        for point in points:
            if isinstance(point, dict):
                output.append(extract_xy(point))
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                output.append((_safe_float(point[0]), _safe_float(point[1])))

        if len(output) >= 2:
            return output

    start = route.get("start")
    end = route.get("end")

    if isinstance(start, dict) and isinstance(end, dict):
        return [extract_xy(start), extract_xy(end)]

    if isinstance(start, (list, tuple)) and isinstance(end, (list, tuple)) and len(start) >= 2 and len(end) >= 2:
        return [
            (_safe_float(start[0]), _safe_float(start[1])),
            (_safe_float(end[0]), _safe_float(end[1])),
        ]

    keys = ["start_x", "start_y", "end_x", "end_y"]
    if all(key in route for key in keys):
        return [
            (_safe_float(route.get("start_x")), _safe_float(route.get("start_y"))),
            (_safe_float(route.get("end_x")), _safe_float(route.get("end_y"))),
        ]

    return []


def _get_report_bounds(report: dict[str, Any]) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []

    for device in report.get("devices") or []:
        if isinstance(device, dict):
            x, y = extract_xy(device)
            xs.append(x)
            ys.append(y)

    for route in report.get("routes") or []:
        if isinstance(route, dict):
            for x, y in extract_route_polyline(route):
                xs.append(x)
                ys.append(y)

    for room in report.get("rooms") or []:
        if isinstance(room, dict):
            x = _safe_float(room.get("x"), 0.0)
            y = _safe_float(room.get("y"), 0.0)
            w = _safe_float(room.get("w") or room.get("width_px") or room.get("width"), 0.0)
            h = _safe_float(room.get("h") or room.get("height_px") or room.get("height"), 0.0)
            xs.extend([x, x + w])
            ys.extend([y, y + h])

    if not xs or not ys:
        return 0.0, 0.0, 1000.0, 700.0

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    if abs(max_x - min_x) < 1:
        max_x += 100.0

    if abs(max_y - min_y) < 1:
        max_y += 100.0

    return min_x, min_y, max_x, max_y


def _drawing_text_height(report: dict[str, Any]) -> float:
    min_x, min_y, max_x, max_y = _get_report_bounds(report)
    size = max(max_x - min_x, max_y - min_y)
    return max(2.0, min(size / 180.0, 12.0))


def _block_scale(report: dict[str, Any]) -> float:
    min_x, min_y, max_x, max_y = _get_report_bounds(report)
    size = max(max_x - min_x, max_y - min_y)
    return max(1.0, min(size / 450.0, 10.0))


# ---------------------------------------------------------------------------
# Document creation/loading
# ---------------------------------------------------------------------------

def load_or_create_document(
    original_file_bytes: bytes | None = None,
    original_filename: str | None = None,
    report: dict[str, Any] | None = None,
):
    _require_ezdxf()

    original_filename = original_filename or ""
    suffix = Path(original_filename).suffix.lower()

    if original_file_bytes and suffix == ".dxf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
            tmp.write(original_file_bytes)
            tmp_path = Path(tmp.name)

        try:
            return ezdxf.readfile(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 6  # metres
    msp = doc.modelspace()

    if report:
        min_x, min_y, max_x, max_y = _get_report_bounds(report)
    else:
        min_x, min_y, max_x, max_y = 0.0, 0.0, 1000.0, 700.0

    pad = max(50.0, (max_x - min_x) * 0.04)
    points = [
        (min_x - pad, min_y - pad),
        (max_x + pad, min_y - pad),
        (max_x + pad, max_y + pad),
        (min_x - pad, max_y + pad),
        (min_x - pad, min_y - pad),
    ]

    if LAYER_BORDER not in doc.layers:
        doc.layers.new(name=LAYER_BORDER, dxfattribs={"color": 8})

    _add_polyline(msp, points, dxfattribs={"layer": LAYER_BORDER})
    return doc


def ensure_layers(doc: Any) -> None:
    layer_specs = [
        (LAYER_BORDER, 8),
        (LAYER_ROOMS, 9),
        (LAYER_FIRE_DEVICE, 1),
        (LAYER_FIRE_ALARM, 1),
        (LAYER_SPRINKLER, 5),
        (LAYER_HYDRANT, 1),
        (LAYER_HVAC, 3),
        (LAYER_ELECTRICAL, 30),
        (LAYER_PLUMBING, 4),
        (LAYER_TEXT, 7),
        (LAYER_ROUTE, 6),
        (LAYER_REVIEW, 2),
    ]

    for name, color in layer_specs:
        if name not in doc.layers:
            doc.layers.new(name=name, dxfattribs={"color": color})


# ---------------------------------------------------------------------------
# Block creation
# ---------------------------------------------------------------------------

def _add_centered_text(block: Any, text: str, insert: tuple[float, float], height: float) -> None:
    entity = block.add_text(text, dxfattribs={"height": height, "layer": LAYER_TEXT})

    try:
        entity.set_placement(insert, align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER)
    except Exception:
        entity.dxf.insert = insert


def _create_round_symbol(block: Any, label: str, radius: float = 1.2) -> None:
    block.add_circle((0, 0), radius, dxfattribs={"layer": LAYER_FIRE_DEVICE})
    block.add_line((-radius, 0), (radius, 0), dxfattribs={"layer": LAYER_FIRE_DEVICE})
    block.add_line((0, -radius), (0, radius), dxfattribs={"layer": LAYER_FIRE_DEVICE})
    _add_centered_text(block, label, (0, -radius - 1.4), 0.9)


def _create_square_symbol(block: Any, label: str, size: float = 2.6) -> None:
    half = size / 2.0
    _add_polyline(
        block,
        [(-half, -half), (half, -half), (half, half), (-half, half), (-half, -half)],
        dxfattribs={"layer": LAYER_FIRE_DEVICE},
    )
    _add_centered_text(block, label, (0, 0), 0.8)


def _create_triangle_symbol(block: Any, label: str, size: float = 2.8) -> None:
    h = size / 2
    points = [(0, h), (-h, -h), (h, -h), (0, h)]
    _add_polyline(block, points, dxfattribs={"layer": LAYER_FIRE_DEVICE})
    _add_centered_text(block, label, (0, -h - 1.0), 0.75)


def _create_extinguisher_symbol(block: Any) -> None:
    _add_polyline(
        block,
        [(-0.8, -1.5), (0.8, -1.5), (0.8, 1.2), (-0.8, 1.2), (-0.8, -1.5)],
        dxfattribs={"layer": LAYER_FIRE_DEVICE},
    )
    block.add_line((-0.5, 1.2), (0.5, 1.2), dxfattribs={"layer": LAYER_FIRE_DEVICE})
    block.add_line((0, 1.2), (0, 1.8), dxfattribs={"layer": LAYER_FIRE_DEVICE})
    _add_centered_text(block, "EXT", (0, -2.5), 0.8)


def _create_exit_sign_symbol(block: Any) -> None:
    _add_polyline(
        block,
        [(-2.4, -0.8), (2.4, -0.8), (2.4, 0.8), (-2.4, 0.8), (-2.4, -0.8)],
        dxfattribs={"layer": LAYER_FIRE_DEVICE},
    )
    _add_centered_text(block, "EXIT", (0, 0), 0.7)


def _create_riser_symbol(block: Any) -> None:
    block.add_circle((0, 0), 1.7, dxfattribs={"layer": LAYER_FIRE_DEVICE})
    block.add_circle((0, 0), 0.7, dxfattribs={"layer": LAYER_FIRE_DEVICE})
    block.add_line((-1.7, 0), (1.7, 0), dxfattribs={"layer": LAYER_FIRE_DEVICE})
    block.add_line((0, -1.7), (0, 1.7), dxfattribs={"layer": LAYER_FIRE_DEVICE})
    _add_centered_text(block, "RISER", (0, -2.8), 0.8)


def _create_hydrant_symbol(block: Any) -> None:
    block.add_circle((0, 0), 1.5, dxfattribs={"layer": LAYER_FIRE_DEVICE})
    block.add_line((-2.0, 0), (2.0, 0), dxfattribs={"layer": LAYER_FIRE_DEVICE})
    block.add_line((0, -2.0), (0, 2.0), dxfattribs={"layer": LAYER_FIRE_DEVICE})
    _add_centered_text(block, "HYD", (0, -2.8), 0.8)


def _create_generic_symbol(block: Any, label: str) -> None:
    block.add_circle((0, 0), 1.1, dxfattribs={"layer": LAYER_FIRE_DEVICE})
    _add_centered_text(block, label[:6], (0, -1.9), 0.75)


def ensure_device_blocks(doc: Any) -> None:
    for device_type, label in BLOCK_LABELS.items():
        block_name = DEVICE_BLOCK_PREFIX + device_type

        if block_name in doc.blocks:
            continue

        block = doc.blocks.new(name=block_name)

        if device_type == "SPRINKLER_HEAD":
            _create_round_symbol(block, "SP", radius=1.0)
        elif device_type == "SMOKE_DETECTOR":
            _create_round_symbol(block, "SD", radius=1.2)
        elif device_type == "HEAT_DETECTOR":
            _create_round_symbol(block, "HD", radius=1.2)
        elif device_type == "MANUAL_CALL_POINT":
            _create_square_symbol(block, "MCP", size=2.2)
        elif device_type == "HORN_STROBE":
            _create_square_symbol(block, "HS", size=2.4)
        elif device_type == "FIRE_ALARM_CONTROL_PANEL":
            _create_square_symbol(block, "FACP", size=4.0)
        elif device_type in {"CONTROL_VALVE", "FLOW_SWITCH", "TAMPER_SWITCH", "CHECK_VALVE", "INSPECTOR_TEST_VALVE"}:
            _create_square_symbol(block, label, size=2.4)
        elif device_type == "FIRE_EXTINGUISHER":
            _create_extinguisher_symbol(block)
        elif device_type == "EXIT_SIGNAGE":
            _create_exit_sign_symbol(block)
        elif device_type == "FIRE_RISER":
            _create_riser_symbol(block)
        elif device_type in {"FIRE_HYDRANT", "HOSE_REEL", "LANDING_VALVE"}:
            _create_hydrant_symbol(block)
        elif device_type in {"AHU", "ELECTRICAL_DISTRIBUTION_BOARD", "MAIN_DISTRIBUTION_BOARD", "PLUMBING_RISER"}:
            _create_square_symbol(block, label, size=3.6)
        elif device_type in {"SUPPLY_DIFFUSER", "RETURN_GRILLE", "EXHAUST_GRILLE"}:
            _create_triangle_symbol(block, label, size=2.5)
        else:
            _create_generic_symbol(block, label)


# ---------------------------------------------------------------------------
# Insert overlays
# ---------------------------------------------------------------------------

def insert_rooms(doc: Any, report: dict[str, Any]) -> int:
    msp = doc.modelspace()
    count = 0
    height = _drawing_text_height(report) * 0.65

    for room in report.get("rooms") or []:
        if not isinstance(room, dict):
            continue

        x = _safe_float(room.get("x"), 0.0)
        y = _safe_float(room.get("y"), 0.0)
        w = _safe_float(room.get("w") or room.get("width_px") or room.get("width"), 0.0)
        h = _safe_float(room.get("h") or room.get("height_px") or room.get("height"), 0.0)

        if w <= 0 or h <= 0:
            continue

        _add_polyline(
            msp,
            [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)],
            dxfattribs={"layer": LAYER_ROOMS},
        )

        label = f'{room.get("id", "ROOM")} {room.get("area_class", "")} {room.get("area_m2", "")}m2'
        text = msp.add_text(
            label[:80],
            dxfattribs={"layer": LAYER_TEXT, "height": height},
        )
        text.dxf.insert = (x + max(2.0, height), y + max(2.0, height * 1.5))
        count += 1

    return count


def _device_scale(device: dict[str, Any], report: dict[str, Any]) -> float:
    scale = _safe_float(device.get("scale"), 0.0)
    if scale > 0:
        return min(max(scale, 1.0), 12.0)
    return _block_scale(report)


def insert_devices(doc: Any, report: dict[str, Any]) -> int:
    msp = doc.modelspace()
    count = 0
    text_height = _drawing_text_height(report)

    for device in report.get("devices") or []:
        if not isinstance(device, dict):
            continue

        device_type = extract_device_type(device)
        block_name = DEVICE_BLOCK_PREFIX + device_type
        x, y = extract_xy(device)

        if block_name not in doc.blocks:
            block_name = DEVICE_BLOCK_PREFIX + "UNKNOWN"

        scale = _device_scale(device, report)
        layer = DEVICE_LAYER_MAP.get(device_type, LAYER_FIRE_DEVICE)

        msp.add_blockref(
            block_name,
            (x, y),
            dxfattribs={
                "layer": layer,
                "xscale": scale,
                "yscale": scale,
            },
        )

        label = str(
            device.get("type")
            or device.get("label")
            or device.get("name")
            or BLOCK_LABELS.get(device_type, device_type)
        )

        room_id = device.get("room_id")
        if room_id:
            label = f"{label} {room_id}"

        if label:
            text = msp.add_text(
                label[:28],
                dxfattribs={"layer": LAYER_TEXT, "height": text_height},
            )
            try:
                text.set_placement(
                    (x + 2.2 * scale, y + 2.2 * scale),
                    align=ezdxf.enums.TextEntityAlignment.LEFT,
                )
            except Exception:
                text.dxf.insert = (x + 2.2 * scale, y + 2.2 * scale)

        count += 1

    return count


def insert_routes(doc: Any, report: dict[str, Any]) -> int:
    msp = doc.modelspace()
    count = 0
    text_height = _drawing_text_height(report) * 0.75

    for route in report.get("routes") or []:
        if not isinstance(route, dict):
            continue

        points = extract_route_polyline(route)
        if len(points) < 2:
            continue

        raw_type = _normalise(route.get("type") or route.get("route_type") or "route")
        layer = ROUTE_LAYER_MAP.get(raw_type, LAYER_ROUTE)

        _add_polyline(
            msp,
            points,
            dxfattribs={
                "layer": layer,
            },
        )

        mid = points[len(points) // 2]
        length_m = route.get("length_m")
        diameter = route.get("diameter")

        label_parts = [raw_type]
        if diameter:
            label_parts.append(str(diameter))
        if length_m not in (None, ""):
            label_parts.append(f'{_format_number(length_m)}m')

        label = " ".join(label_parts)

        text = msp.add_text(
            label[:40],
            dxfattribs={"layer": LAYER_TEXT, "height": text_height},
        )

        try:
            text.set_placement(mid, align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER)
        except Exception:
            text.dxf.insert = mid

        count += 1

    return count


def add_legend_and_review_stamp(
    doc: Any,
    report: dict[str, Any],
    room_count: int,
    device_count: int,
    route_count: int,
) -> None:
    msp = doc.modelspace()
    min_x, min_y, max_x, max_y = _get_report_bounds(report)
    text_h = _drawing_text_height(report)

    x = min_x
    y = min_y - max(60.0, text_h * 9.0)

    title = "FIREDESIGN AUTOMATION / CEASEFIRE POC CAD EXPORT"
    note = "POC ONLY - ENGINEER/AHJ REVIEW REQUIRED BEFORE CLIENT SUBMISSION, PROCUREMENT OR CONSTRUCTION."

    rows = [
        title,
        note,
        f"Rooms overlaid: {room_count}",
        f"Generated devices: {device_count}",
        f"Generated routes: {route_count}",
        "Layers: CEASEFIRE_SPRINKLER, CEASEFIRE_FIRE_ALARM, CEASEFIRE_FIRE_SAFETY, CEASEFIRE_ROUTES, CEASEFIRE_REVIEW",
        "Blocks: POC symbols only. Replace with official Ceasefire CAD block library for production.",
    ]

    for index, row in enumerate(rows):
        entity = msp.add_text(
            row[:180],
            dxfattribs={"layer": LAYER_REVIEW, "height": text_h},
        )
        entity.dxf.insert = (x, y - index * text_h * 1.8)

    # Simple legend
    legend_x = min_x
    legend_y = y - len(rows) * text_h * 1.8 - text_h * 2.0
    legend_items = [
        ("SP", "Sprinkler head"),
        ("SD/HD", "Smoke/heat detector"),
        ("MCP/HS", "Manual call point / horn-strobe"),
        ("EXT/SIGN", "Extinguisher / signage"),
        ("MAIN/BRANCH/DROP", "Sprinkler routes"),
        ("SLC/NAC", "Fire alarm circuits"),
    ]

    for idx, (symbol, description) in enumerate(legend_items):
        ly = legend_y - idx * text_h * 1.6
        msp.add_text(
            f"{symbol}: {description}",
            dxfattribs={"layer": LAYER_REVIEW, "height": text_h * 0.85},
        ).dxf.insert = (legend_x, ly)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def document_to_bytes(doc: Any) -> bytes:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
        tmp_path = Path(tmp.name)

    try:
        doc.saveas(tmp_path)
        return tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)


def build_updated_dxf(
    report: dict[str, Any],
    original_file_bytes: bytes | None = None,
    original_filename: str | None = None,
) -> bytes:
    """
    Creates an updated DXF containing generated room overlays, devices and routes.

    If original file is DXF:
        The original geometry is preserved and CEASEFIRE_* layers are added.

    If original file is raster:
        A proof DXF is created with detected room boxes, generated devices and routes.
    """
    _require_ezdxf()

    if not isinstance(report, dict):
        raise ValueError("report must be a dictionary")

    doc = load_or_create_document(
        original_file_bytes=original_file_bytes,
        original_filename=original_filename,
        report=report,
    )

    ensure_layers(doc)
    ensure_device_blocks(doc)

    room_count = insert_rooms(doc, report)
    device_count = insert_devices(doc, report)
    route_count = insert_routes(doc, report)

    add_legend_and_review_stamp(
        doc=doc,
        report=report,
        room_count=room_count,
        device_count=device_count,
        route_count=route_count,
    )

    return document_to_bytes(doc)


def dxf_bytes_to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def dxf_base64_to_data_url(data: bytes) -> str:
    encoded = dxf_bytes_to_base64(data)
    return f"data:application/dxf;base64,{encoded}"


def build_dxf_download_payload(
    report: dict[str, Any],
    original_file_bytes: bytes | None = None,
    original_filename: str | None = None,
) -> dict[str, Any]:
    dxf_bytes = build_updated_dxf(
        report=report,
        original_file_bytes=original_file_bytes,
        original_filename=original_filename,
    )

    return {
        "filename": "ceasefire_updated_layout.dxf",
        "content_base64": dxf_bytes_to_base64(dxf_bytes),
        "mime_type": "application/dxf",
        "note": (
            "POC DXF export. Original DXF geometry is preserved when uploaded as DXF. "
            "Generated rooms, devices and routes are placed on CEASEFIRE_* layers. "
            "Old DXF versions are handled safely by falling back to LINE segments when LWPOLYLINE is unavailable. "
            "Engineer/AHJ review is required before use."
        ),
        "compatibility": {
            "old_dxf_safe_polyline_fallback": True,
            "prevents_error": "LWPOLYLINE requires DXF R2000",
        },
        "layers": [
            LAYER_ROOMS,
            LAYER_FIRE_DEVICE,
            LAYER_FIRE_ALARM,
            LAYER_SPRINKLER,
            LAYER_HYDRANT,
            LAYER_HVAC,
            LAYER_ELECTRICAL,
            LAYER_PLUMBING,
            LAYER_ROUTE,
            LAYER_TEXT,
            LAYER_REVIEW,
        ],
    }