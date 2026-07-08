from __future__ import annotations

"""
Ceasefire-style BOM mapper for FireDesign Automation.

Purpose
-------
This module converts the generic backend BOM/devices/routes into a stronger
Ceasefire-style proof-of-concept material schedule.

It supports:
- device product-code mapping
- cable/pipe length mapping
- sprinkler pipe diameter grouping
- estimated fittings/accessories takeoff
- dummy POC prices
- CSV export

Place this file at:
    FIRE_ALARM/backend/bom_mapper.py

Then restart backend:
    uvicorn backend.main:app --reload --host 127.0.0.1 --port 8020

Important:
    Product codes and prices are dummy POC data only.
    Replace PRODUCT_MASTER with the official Ceasefire product master later.
"""

import csv
import io
import math
from collections import defaultdict
from typing import Any


CSV_HEADER = (
    "item_no,category,device_type,product_code,description,quantity,unit,"
    "unit_price,total_price,mapping_status,takeoff_basis\n"
)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _normalise(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip().upper()
    text = text.replace("²", "2")
    text = text.replace("Ø", "DIA")
    text = text.replace("-", "_").replace("/", "_").replace(" ", "_")
    text = "".join(ch for ch in text if ch.isalnum() or ch == "_")

    while "__" in text:
        text = text.replace("__", "_")

    return text.strip("_")


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


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return fallback
        if isinstance(value, str):
            value = value.replace(",", "").replace("m", "").strip()
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return fallback
        return number
    except (TypeError, ValueError):
        return fallback


def _clean_quantity(value: Any) -> int | float:
    quantity = _safe_float(value, 1.0)
    if abs(quantity - round(quantity)) < 0.000001:
        return int(round(quantity))
    return round(quantity, 3)


def _route_length(route: dict[str, Any]) -> float:
    direct = _get_case_insensitive(
        route,
        ["length_m", "route_length_m", "length", "quantity", "qty"],
        default=None,
    )

    if direct is not None:
        return _safe_float(direct, 0.0)

    points = route.get("points")
    if isinstance(points, list) and len(points) >= 2:
        total = 0.0

        for p1, p2 in zip(points, points[1:]):
            try:
                x1, y1 = p1[:2] if isinstance(p1, (list, tuple)) else (p1.get("x"), p1.get("y"))
                x2, y2 = p2[:2] if isinstance(p2, (list, tuple)) else (p2.get("x"), p2.get("y"))
                total += math.dist((_safe_float(x1), _safe_float(y1)), (_safe_float(x2), _safe_float(y2)))
            except Exception:
                continue

        # If route had no scale, do not guess metres from pixels here.
        return 0.0 if total > 5000 else total

    return 0.0


# ---------------------------------------------------------------------------
# Alias mapping
# ---------------------------------------------------------------------------

DEVICE_ALIASES: dict[str, str] = {
    # Fire alarm system
    "SD": "SMOKE_DETECTOR",
    "SMOKE": "SMOKE_DETECTOR",
    "SMOKE_DETECTOR": "SMOKE_DETECTOR",
    "PHOTOELECTRIC_SMOKE_DETECTOR": "SMOKE_DETECTOR",
    "OPTICAL_SMOKE_DETECTOR": "SMOKE_DETECTOR",
    "ADDRESSABLE_SMOKE_DETECTOR": "SMOKE_DETECTOR",

    "HD": "HEAT_DETECTOR",
    "HEAT": "HEAT_DETECTOR",
    "HEAT_DETECTOR": "HEAT_DETECTOR",
    "ADDRESSABLE_HEAT_DETECTOR": "HEAT_DETECTOR",

    "MCP": "MANUAL_CALL_POINT",
    "MANUAL_CALL_POINT": "MANUAL_CALL_POINT",
    "CALL_POINT": "MANUAL_CALL_POINT",
    "BREAK_GLASS": "MANUAL_CALL_POINT",

    "HS": "HORN_STROBE",
    "HORN": "HORN_STROBE",
    "STROBE": "HORN_STROBE",
    "HORN_STROBE": "HORN_STROBE",
    "SOUNDER": "HORN_STROBE",
    "SOUNDER_STROBE": "HORN_STROBE",
    "VISUAL_ALARM_STROBE": "HORN_STROBE",

    "FACP": "FIRE_ALARM_CONTROL_PANEL",
    "PANEL": "FIRE_ALARM_CONTROL_PANEL",
    "FIRE_PANEL": "FIRE_ALARM_CONTROL_PANEL",
    "FIRE_ALARM_PANEL": "FIRE_ALARM_CONTROL_PANEL",
    "FIRE_ALARM_CONTROL_PANEL": "FIRE_ALARM_CONTROL_PANEL",

    "ANNUNCIATOR": "REMOTE_ANNUNCIATOR",
    "REMOTE_ANNUNCIATOR": "REMOTE_ANNUNCIATOR",

    "FIRE_ALARM_CABLE_M": "FIRE_ALARM_CABLE",
    "FIRE_ALARM_CABLE": "FIRE_ALARM_CABLE",
    "CABLE": "FIRE_ALARM_CABLE",
    "SLC": "FIRE_ALARM_CABLE",
    "NAC": "FIRE_ALARM_CABLE",
    "SLC_CABLE": "FIRE_ALARM_CABLE",
    "NAC_CABLE": "FIRE_ALARM_CABLE",

    "FIRE_ALARM_BACKBOX": "FIRE_ALARM_BACKBOX",
    "BACKBOX": "FIRE_ALARM_BACKBOX",
    "FIRE_ALARM_BATTERY": "FIRE_ALARM_BATTERY",
    "BATTERY": "FIRE_ALARM_BATTERY",

    # Portable fire and signage
    "EXT": "FIRE_EXTINGUISHER",
    "EXTINGUISHER": "FIRE_EXTINGUISHER",
    "PORTABLE_FIRE_EXTINGUISHER": "FIRE_EXTINGUISHER",
    "FIRE_EXTINGUISHER": "FIRE_EXTINGUISHER",
    "ABC_EXTINGUISHER": "FIRE_EXTINGUISHER",

    "SIGN": "EXIT_SIGNAGE",
    "EXIT_SIGN": "EXIT_SIGNAGE",
    "EXIT_SIGNAGE": "EXIT_SIGNAGE",
    "FIRE_SAFETY_EXIT_SIGNAGE": "EXIT_SIGNAGE",
    "FIRE_EXIT_SIGN": "EXIT_SIGNAGE",

    "EM_LIGHT": "EMERGENCY_LIGHT",
    "EMERGENCY_LIGHT": "EMERGENCY_LIGHT",
    "EMERGENCY_LIGHTING": "EMERGENCY_LIGHT",

    # Sprinkler / hydrant system
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
    "FIRE_DEPARTMENT_CONNECTION": "FIRE_DEPARTMENT_CONNECTION",

    "HYDRANT": "FIRE_HYDRANT",
    "FIRE_HYDRANT": "FIRE_HYDRANT",
    "HOSE_REEL": "HOSE_REEL",
    "FIRE_HOSE_REEL": "HOSE_REEL",
    "LANDING_VALVE": "LANDING_VALVE",
    "FIRE_PUMP": "FIRE_PUMP",

    "PIPE_M": "FIRE_PIPE",
    "PIPE": "FIRE_PIPE",
    "SPRINKLER_PIPE_M": "FIRE_PIPE",
    "SPRINKLER_PIPE_ROUTE_ALLOWANCE_12": "FIRE_PIPE",
    "SPRINKLER_PIPE_ROUTE_ALLOWANCE_12_": "FIRE_PIPE",
    "BRANCH": "FIRE_PIPE",
    "MAIN": "FIRE_PIPE",
    "DROP": "FIRE_PIPE",
    "BRANCH_PIPE_M": "FIRE_PIPE",
    "MAIN_PIPE_M": "FIRE_PIPE",
    "DROP_PIPE_M": "FIRE_PIPE",

    "FITTING": "PIPE_FITTINGS_ALLOWANCE",
    "FITTINGS": "PIPE_FITTINGS_ALLOWANCE",
    "PIPE_FITTINGS_ALLOWANCE": "PIPE_FITTINGS_ALLOWANCE",
    "ELBOW": "PIPE_ELBOW",
    "TEE": "PIPE_TEE",
    "COUPLING": "PIPE_COUPLING",
    "HANGER": "PIPE_HANGER",

    # HVAC/electrical/plumbing can appear in full-package BOM.
    "AHU": "AHU",
    "SDIFF": "SUPPLY_DIFFUSER",
    "RGRILLE": "RETURN_GRILLE",
    "EXH": "EXHAUST_GRILLE",
    "EXFAN": "EXHAUST_GRILLE",
    "DUCT_ROUTE_M": "DUCT_ROUTE",

    "EDB": "ELECTRICAL_DISTRIBUTION_BOARD",
    "MDB": "MAIN_DISTRIBUTION_BOARD",
    "LIGHT": "LIGHT_FIXTURE",
    "SWITCH": "LIGHT_SWITCH",
    "SO": "POWER_SOCKET",
    "POWER_SOCKET": "POWER_SOCKET",
    "ELECTRICAL_CABLE_M": "ELECTRICAL_CABLE",

    "PLUMBING_RISER": "PLUMBING_RISER",
    "LAV": "LAVATORY",
    "SINK": "SINK",
    "WC": "WC_FIXTURE",
    "FD": "FLOOR_DRAIN",
    "WATER_PIPE": "WATER_PIPE",
    "DRAIN_PIPE": "DRAIN_PIPE",
    "WATER_PIPE_M": "WATER_PIPE",
    "DRAIN_PIPE_M": "DRAIN_PIPE",
}


# ---------------------------------------------------------------------------
# Dummy Ceasefire-style product master
# ---------------------------------------------------------------------------

PRODUCT_MASTER: dict[str, dict[str, Any]] = {
    # Fire alarm
    "SMOKE_DETECTOR": {
        "product_code": "CF-SD-001",
        "description": "Ceasefire addressable smoke detector, dummy POC item",
        "unit": "nos",
        "unit_price": 1250.0,
        "category": "Fire Alarm System",
    },
    "HEAT_DETECTOR": {
        "product_code": "CF-HD-001",
        "description": "Ceasefire addressable heat detector, dummy POC item",
        "unit": "nos",
        "unit_price": 1150.0,
        "category": "Fire Alarm System",
    },
    "MANUAL_CALL_POINT": {
        "product_code": "CF-MCP-001",
        "description": "Ceasefire manual call point, dummy POC item",
        "unit": "nos",
        "unit_price": 950.0,
        "category": "Fire Alarm System",
    },
    "HORN_STROBE": {
        "product_code": "CF-HS-001",
        "description": "Ceasefire horn/strobe sounder, dummy POC item",
        "unit": "nos",
        "unit_price": 1450.0,
        "category": "Fire Alarm System",
    },
    "FIRE_ALARM_CONTROL_PANEL": {
        "product_code": "CF-FACP-001",
        "description": "Ceasefire fire alarm control panel, dummy POC item",
        "unit": "nos",
        "unit_price": 18500.0,
        "category": "Fire Alarm System",
    },
    "REMOTE_ANNUNCIATOR": {
        "product_code": "CF-ANN-001",
        "description": "Ceasefire remote annunciator panel, dummy POC item",
        "unit": "nos",
        "unit_price": 6500.0,
        "category": "Fire Alarm System",
    },
    "FIRE_ALARM_CABLE": {
        "product_code": "CF-CBL-FAS-001",
        "description": "Ceasefire fire alarm cable route allowance, dummy POC item",
        "unit": "m",
        "unit_price": 65.0,
        "category": "Fire Alarm System",
    },
    "FIRE_ALARM_BACKBOX": {
        "product_code": "CF-BBOX-001",
        "description": "Fire alarm device back box allowance, dummy POC item",
        "unit": "nos",
        "unit_price": 90.0,
        "category": "Fire Alarm Accessories",
    },
    "FIRE_ALARM_BATTERY": {
        "product_code": "CF-BAT-FA-001",
        "description": "Fire alarm standby battery allowance, dummy POC item",
        "unit": "set",
        "unit_price": 2800.0,
        "category": "Fire Alarm Accessories",
    },

    # Portable fire / signage
    "FIRE_EXTINGUISHER": {
        "product_code": "CF-EXT-ABC-6KG",
        "description": "Ceasefire ABC fire extinguisher 6 kg, dummy POC item",
        "unit": "nos",
        "unit_price": 2800.0,
        "category": "Portable Fire Extinguisher",
    },
    "EXIT_SIGNAGE": {
        "product_code": "CF-SIGN-EXIT-001",
        "description": "Ceasefire fire safety / exit signage, dummy POC item",
        "unit": "nos",
        "unit_price": 650.0,
        "category": "Signage",
    },
    "EMERGENCY_LIGHT": {
        "product_code": "CF-EML-001",
        "description": "Ceasefire emergency light fitting, dummy POC item",
        "unit": "nos",
        "unit_price": 1750.0,
        "category": "Emergency Lighting",
    },

    # Sprinkler system
    "SPRINKLER_HEAD": {
        "product_code": "CF-SPK-001",
        "description": "Ceasefire pendent sprinkler head, dummy POC item",
        "unit": "nos",
        "unit_price": 450.0,
        "category": "Sprinkler System",
    },
    "FIRE_RISER": {
        "product_code": "CF-RISER-001",
        "description": "Ceasefire fire riser marker/assembly, dummy POC item",
        "unit": "set",
        "unit_price": 15000.0,
        "category": "Fire System",
    },
    "CONTROL_VALVE": {
        "product_code": "CF-VLV-001",
        "description": "Ceasefire control valve assembly, dummy POC item",
        "unit": "nos",
        "unit_price": 4200.0,
        "category": "Sprinkler Accessories",
    },
    "FLOW_SWITCH": {
        "product_code": "CF-FS-001",
        "description": "Ceasefire flow switch, dummy POC item",
        "unit": "nos",
        "unit_price": 3600.0,
        "category": "Sprinkler Accessories",
    },
    "TAMPER_SWITCH": {
        "product_code": "CF-TS-001",
        "description": "Ceasefire tamper switch, dummy POC item",
        "unit": "nos",
        "unit_price": 3100.0,
        "category": "Sprinkler Accessories",
    },
    "CHECK_VALVE": {
        "product_code": "CF-CV-001",
        "description": "Ceasefire check valve allowance, dummy POC item",
        "unit": "nos",
        "unit_price": 3800.0,
        "category": "Sprinkler Accessories",
    },
    "INSPECTOR_TEST_VALVE": {
        "product_code": "CF-ITV-001",
        "description": "Inspector test and drain valve allowance, dummy POC item",
        "unit": "nos",
        "unit_price": 2600.0,
        "category": "Sprinkler Accessories",
    },
    "PRESSURE_GAUGE": {
        "product_code": "CF-PG-001",
        "description": "Pressure gauge allowance, dummy POC item",
        "unit": "nos",
        "unit_price": 750.0,
        "category": "Sprinkler Accessories",
    },
    "FIRE_DEPARTMENT_CONNECTION": {
        "product_code": "CF-FDC-001",
        "description": "Fire department connection allowance, dummy POC item",
        "unit": "nos",
        "unit_price": 9500.0,
        "category": "Sprinkler Accessories",
    },
    "FIRE_PIPE": {
        "product_code": "CF-PIPE-FIRE-001",
        "description": "Ceasefire fire pipe route allowance, dummy POC item",
        "unit": "m",
        "unit_price": 350.0,
        "category": "Fire Pipework",
    },
    "PIPE_FITTINGS_ALLOWANCE": {
        "product_code": "CF-FIT-ALLOW-001",
        "description": "Fire pipe fittings allowance, dummy POC item",
        "unit": "lot",
        "unit_price": 1200.0,
        "category": "Fire Pipework",
    },
    "PIPE_ELBOW": {
        "product_code": "CF-ELB-001",
        "description": "Fire pipe elbow allowance, dummy POC item",
        "unit": "nos",
        "unit_price": 180.0,
        "category": "Fire Pipework",
    },
    "PIPE_TEE": {
        "product_code": "CF-TEE-001",
        "description": "Fire pipe tee allowance, dummy POC item",
        "unit": "nos",
        "unit_price": 240.0,
        "category": "Fire Pipework",
    },
    "PIPE_COUPLING": {
        "product_code": "CF-CPL-001",
        "description": "Fire pipe coupling allowance, dummy POC item",
        "unit": "nos",
        "unit_price": 160.0,
        "category": "Fire Pipework",
    },
    "PIPE_HANGER": {
        "product_code": "CF-HGR-001",
        "description": "Fire pipe hanger/support allowance, dummy POC item",
        "unit": "nos",
        "unit_price": 150.0,
        "category": "Fire Pipework",
    },

    # Hydrant / hose
    "FIRE_HYDRANT": {
        "product_code": "CF-HYD-001",
        "description": "Ceasefire fire hydrant point, dummy POC item",
        "unit": "nos",
        "unit_price": 12500.0,
        "category": "Hydrant System",
    },
    "HOSE_REEL": {
        "product_code": "CF-HR-001",
        "description": "Ceasefire hose reel assembly, dummy POC item",
        "unit": "nos",
        "unit_price": 9500.0,
        "category": "Hydrant System",
    },
    "LANDING_VALVE": {
        "product_code": "CF-LV-001",
        "description": "Landing valve allowance, dummy POC item",
        "unit": "nos",
        "unit_price": 7200.0,
        "category": "Hydrant System",
    },
    "FIRE_PUMP": {
        "product_code": "CF-PUMP-001",
        "description": "Fire pump placeholder, dummy POC item",
        "unit": "set",
        "unit_price": 120000.0,
        "category": "Hydrant System",
    },

    # MEP placeholder mapping
    "AHU": {
        "product_code": "CF-MEP-AHU-001",
        "description": "AHU/indoor unit placeholder, dummy POC item",
        "unit": "nos",
        "unit_price": 0.0,
        "category": "HVAC Placeholder",
    },
    "SUPPLY_DIFFUSER": {
        "product_code": "CF-MEP-SDIF-001",
        "description": "Supply diffuser placeholder, dummy POC item",
        "unit": "nos",
        "unit_price": 0.0,
        "category": "HVAC Placeholder",
    },
    "RETURN_GRILLE": {
        "product_code": "CF-MEP-RG-001",
        "description": "Return grille placeholder, dummy POC item",
        "unit": "nos",
        "unit_price": 0.0,
        "category": "HVAC Placeholder",
    },
    "EXHAUST_GRILLE": {
        "product_code": "CF-MEP-EXH-001",
        "description": "Exhaust grille placeholder, dummy POC item",
        "unit": "nos",
        "unit_price": 0.0,
        "category": "HVAC Placeholder",
    },
    "DUCT_ROUTE": {
        "product_code": "CF-MEP-DUCT-001",
        "description": "Duct route placeholder, dummy POC item",
        "unit": "m",
        "unit_price": 0.0,
        "category": "HVAC Placeholder",
    },
    "ELECTRICAL_DISTRIBUTION_BOARD": {
        "product_code": "CF-MEP-EDB-001",
        "description": "Electrical DB placeholder, dummy POC item",
        "unit": "nos",
        "unit_price": 0.0,
        "category": "Electrical Placeholder",
    },
    "MAIN_DISTRIBUTION_BOARD": {
        "product_code": "CF-MEP-MDB-001",
        "description": "Main DB placeholder, dummy POC item",
        "unit": "nos",
        "unit_price": 0.0,
        "category": "Electrical Placeholder",
    },
    "LIGHT_FIXTURE": {
        "product_code": "CF-MEP-LGT-001",
        "description": "Light fixture placeholder, dummy POC item",
        "unit": "nos",
        "unit_price": 0.0,
        "category": "Electrical Placeholder",
    },
    "LIGHT_SWITCH": {
        "product_code": "CF-MEP-SW-001",
        "description": "Switch placeholder, dummy POC item",
        "unit": "nos",
        "unit_price": 0.0,
        "category": "Electrical Placeholder",
    },
    "POWER_SOCKET": {
        "product_code": "CF-MEP-SO-001",
        "description": "Power socket placeholder, dummy POC item",
        "unit": "nos",
        "unit_price": 0.0,
        "category": "Electrical Placeholder",
    },
    "ELECTRICAL_CABLE": {
        "product_code": "CF-MEP-CBL-001",
        "description": "Electrical cable/conduit placeholder, dummy POC item",
        "unit": "m",
        "unit_price": 0.0,
        "category": "Electrical Placeholder",
    },
    "PLUMBING_RISER": {
        "product_code": "CF-MEP-PR-001",
        "description": "Plumbing riser placeholder, dummy POC item",
        "unit": "nos",
        "unit_price": 0.0,
        "category": "Plumbing Placeholder",
    },
    "LAVATORY": {
        "product_code": "CF-MEP-LAV-001",
        "description": "Lavatory placeholder, dummy POC item",
        "unit": "nos",
        "unit_price": 0.0,
        "category": "Plumbing Placeholder",
    },
    "SINK": {
        "product_code": "CF-MEP-SINK-001",
        "description": "Sink placeholder, dummy POC item",
        "unit": "nos",
        "unit_price": 0.0,
        "category": "Plumbing Placeholder",
    },
    "WC_FIXTURE": {
        "product_code": "CF-MEP-WC-001",
        "description": "WC fixture placeholder, dummy POC item",
        "unit": "nos",
        "unit_price": 0.0,
        "category": "Plumbing Placeholder",
    },
    "FLOOR_DRAIN": {
        "product_code": "CF-MEP-FD-001",
        "description": "Floor drain placeholder, dummy POC item",
        "unit": "nos",
        "unit_price": 0.0,
        "category": "Plumbing Placeholder",
    },
    "WATER_PIPE": {
        "product_code": "CF-MEP-WP-001",
        "description": "Water pipe placeholder, dummy POC item",
        "unit": "m",
        "unit_price": 0.0,
        "category": "Plumbing Placeholder",
    },
    "DRAIN_PIPE": {
        "product_code": "CF-MEP-DP-001",
        "description": "Drain pipe placeholder, dummy POC item",
        "unit": "m",
        "unit_price": 0.0,
        "category": "Plumbing Placeholder",
    },
    "UNKNOWN": {
        "product_code": "CF-REVIEW-001",
        "description": "Unmapped generated item - review product mapping",
        "unit": "nos",
        "unit_price": 0.0,
        "category": "Review Required",
    },
}


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------

def canonical_device_type(value: Any) -> str:
    key = _normalise(value)
    if not key:
        return "UNKNOWN"
    return DEVICE_ALIASES.get(key, key if key in PRODUCT_MASTER else "UNKNOWN")


def map_product(device_type: Any) -> dict[str, Any]:
    canonical = canonical_device_type(device_type)
    product = PRODUCT_MASTER.get(canonical, PRODUCT_MASTER["UNKNOWN"])

    return {
        "device_type": canonical,
        "product_code": product["product_code"],
        "description": product["description"],
        "unit": product["unit"],
        "unit_price": float(product["unit_price"]),
        "category": product["category"],
    }


def extract_device_type(row: dict[str, Any]) -> str:
    type_value = _get_case_insensitive(
        row,
        [
            "type",
            "device_type",
            "symbol",
            "code",
            "item_code",
            "tag",
            "route_type",
        ],
    )

    if type_value:
        canonical = canonical_device_type(type_value)
        if canonical != "UNKNOWN":
            return canonical

    name_value = _get_case_insensitive(
        row,
        [
            "item",
            "name",
            "label",
            "description",
            "device",
            "material",
            "equipment",
        ],
    )

    if name_value:
        canonical = canonical_device_type(name_value)
        if canonical != "UNKNOWN":
            return canonical

        text = _normalise(name_value)
        keyword_rules = [
            ("SMOKE", "SMOKE_DETECTOR"),
            ("HEAT", "HEAT_DETECTOR"),
            ("MANUAL_CALL", "MANUAL_CALL_POINT"),
            ("CALL_POINT", "MANUAL_CALL_POINT"),
            ("HORN", "HORN_STROBE"),
            ("STROBE", "HORN_STROBE"),
            ("CONTROL_PANEL", "FIRE_ALARM_CONTROL_PANEL"),
            ("ALARM_PANEL", "FIRE_ALARM_CONTROL_PANEL"),
            ("ANNUNCIATOR", "REMOTE_ANNUNCIATOR"),
            ("EXTINGUISHER", "FIRE_EXTINGUISHER"),
            ("EXIT_SIGN", "EXIT_SIGNAGE"),
            ("SIGNAGE", "EXIT_SIGNAGE"),
            ("EMERGENCY_LIGHT", "EMERGENCY_LIGHT"),
            ("CABLE", "FIRE_ALARM_CABLE"),
            ("SPRINKLER", "SPRINKLER_HEAD"),
            ("RISER", "FIRE_RISER"),
            ("FLOW_SWITCH", "FLOW_SWITCH"),
            ("TAMPER", "TAMPER_SWITCH"),
            ("VALVE", "CONTROL_VALVE"),
            ("HYDRANT", "FIRE_HYDRANT"),
            ("HOSE_REEL", "HOSE_REEL"),
            ("PIPE", "FIRE_PIPE"),
            ("FITTING", "PIPE_FITTINGS_ALLOWANCE"),
            ("ELBOW", "PIPE_ELBOW"),
            ("TEE", "PIPE_TEE"),
            ("COUPLING", "PIPE_COUPLING"),
            ("HANGER", "PIPE_HANGER"),
            ("DIFFUSER", "SUPPLY_DIFFUSER"),
            ("GRILLE", "RETURN_GRILLE"),
            ("AHU", "AHU"),
            ("LIGHT", "LIGHT_FIXTURE"),
            ("SOCKET", "POWER_SOCKET"),
            ("LAV", "LAVATORY"),
            ("SINK", "SINK"),
            ("FLOOR_DRAIN", "FLOOR_DRAIN"),
            ("DRAIN", "DRAIN_PIPE"),
            ("WATER", "WATER_PIPE"),
        ]

        for keyword, device_type in keyword_rules:
            if keyword in text:
                return device_type

    return "UNKNOWN"


def extract_quantity(row: dict[str, Any]) -> int | float:
    value = _get_case_insensitive(
        row,
        [
            "quantity",
            "qty",
            "count",
            "total",
            "amount",
            "length",
            "length_m",
            "route_length_m",
        ],
        default=1,
    )
    return _clean_quantity(value)


def extract_unit(row: dict[str, Any], fallback_unit: str) -> str:
    value = _get_case_insensitive(row, ["unit", "uom", "units"], default=fallback_unit)
    return str(value or fallback_unit).strip() or fallback_unit


def _candidate_bom_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    possible_keys = [
        "bom",
        "bill_of_materials",
        "materials",
        "material_schedule",
        "equipment_schedule",
        "fire_design_takeoff",
    ]

    for key in possible_keys:
        rows = report.get(key)
        if isinstance(rows, list) and rows:
            return [row for row in rows if isinstance(row, dict)]

    return []


def _candidate_device_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    possible_keys = [
        "devices",
        "device_rows",
        "equipment",
        "placed_devices",
        "symbols",
    ]

    for key in possible_keys:
        rows = report.get(key)
        if isinstance(rows, list) and rows:
            return [row for row in rows if isinstance(row, dict)]

    return []


def _candidate_route_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = report.get("routes")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    return []


def _add_counter(
    counter: defaultdict[tuple[str, str, str], float],
    device_type: str,
    quantity: float,
    unit: str,
    basis: str,
) -> None:
    product = map_product(device_type)
    counter[(product["device_type"], unit or product["unit"], basis)] += float(quantity)


def _build_rows_from_counter(counter: dict[tuple[str, str, str], float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped: defaultdict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {"quantity": 0.0, "basis": []})

    for (device_type, unit, basis), quantity in counter.items():
        grouped[(device_type, unit)]["quantity"] += quantity
        if basis:
            grouped[(device_type, unit)]["basis"].append(basis)

    sorted_items = sorted(grouped.items(), key=lambda item: (map_product(item[0][0])["category"], item[0][0]))

    for item_no, ((device_type, source_unit), data) in enumerate(sorted_items, start=1):
        product = map_product(device_type)
        quantity = float(data["quantity"])

        unit = source_unit or product["unit"]

        if product["device_type"] in {
            "FIRE_ALARM_CABLE",
            "FIRE_PIPE",
            "DUCT_ROUTE",
            "ELECTRICAL_CABLE",
            "WATER_PIPE",
            "DRAIN_PIPE",
        }:
            unit = "m"

        if product["device_type"] == "PIPE_FITTINGS_ALLOWANCE":
            unit = "lot"

        basis = "; ".join(sorted(set(data["basis"])))[:240]

        rows.append(
            {
                "item_no": item_no,
                "category": product["category"],
                "device_type": product["device_type"],
                "product_code": product["product_code"],
                "description": product["description"],
                "quantity": _clean_quantity(quantity),
                "unit": unit,
                "unit_price": product["unit_price"],
                "total_price": round(quantity * float(product["unit_price"]), 2),
                "mapping_status": "mapped" if product["device_type"] != "UNKNOWN" else "review_required",
                "takeoff_basis": basis or "mapped from generated backend output",
            }
        )

    return rows


# ---------------------------------------------------------------------------
# Takeoff builders
# ---------------------------------------------------------------------------

def build_ceasefire_bom_from_existing_bom(bom_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter: defaultdict[tuple[str, str, str], float] = defaultdict(float)

    for row in bom_rows or []:
        if not isinstance(row, dict):
            continue

        device_type = extract_device_type(row)
        product = map_product(device_type)
        quantity = float(extract_quantity(row))
        unit = extract_unit(row, product["unit"])

        item_name = str(_get_case_insensitive(row, ["item", "name", "description"], default="backend BOM row"))
        _add_counter(counter, product["device_type"], quantity, unit, f"from BOM: {item_name[:80]}")

    return _build_rows_from_counter(counter)


def build_ceasefire_bom_from_devices(devices: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter: defaultdict[tuple[str, str, str], float] = defaultdict(float)

    for device in devices or []:
        if not isinstance(device, dict):
            continue

        device_type = extract_device_type(device)
        product = map_product(device_type)
        unit = extract_unit(device, product["unit"])
        _add_counter(counter, product["device_type"], 1.0, unit, "counted from placed devices")

    return _build_rows_from_counter(counter)


def build_route_takeoff(routes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Creates additional FireDesign-style takeoff rows from routes.

    This is intentionally approximate:
    - pipe/cable/duct length is based on generated route length
    - fittings are estimated from number of route bends/branches
    - hangers are estimated from pipe length

    It is useful for POC bidding/BOM demonstration, not procurement.
    """
    counter: defaultdict[tuple[str, str, str], float] = defaultdict(float)

    fire_pipe_m = 0.0
    branch_routes = 0
    main_routes = 0
    drop_routes = 0
    fire_alarm_cable_m = 0.0
    duct_m = 0.0
    electrical_cable_m = 0.0
    water_pipe_m = 0.0
    drain_pipe_m = 0.0
    total_bends = 0

    for route in routes or []:
        if not isinstance(route, dict):
            continue

        route_type = canonical_device_type(_get_case_insensitive(route, ["type", "route_type"], default=""))
        raw_type = _normalise(_get_case_insensitive(route, ["type", "route_type"], default=""))
        length = _route_length(route)

        if raw_type in {"MAIN", "BRANCH", "DROP"}:
            fire_pipe_m += length
            if raw_type == "MAIN":
                main_routes += 1
            elif raw_type == "BRANCH":
                branch_routes += 1
            elif raw_type == "DROP":
                drop_routes += 1

        elif raw_type in {"SLC", "NAC"}:
            fire_alarm_cable_m += length

        elif raw_type in {"DUCT_MAIN", "DUCT_BRANCH", "RETURN_DUCT", "EXHAUST_DUCT"}:
            duct_m += length

        elif raw_type in {"LIGHTING_CIRCUIT", "POWER_CIRCUIT", "EMERGENCY_CIRCUIT"}:
            electrical_cable_m += length

        elif raw_type == "WATER_PIPE":
            water_pipe_m += length

        elif raw_type == "DRAIN_PIPE":
            drain_pipe_m += length

        points = route.get("points") or []
        if isinstance(points, list) and len(points) >= 3:
            total_bends += max(0, len(points) - 2)

    if fire_pipe_m > 0:
        pipe_qty = round(fire_pipe_m * 1.12, 2)
        _add_counter(counter, "FIRE_PIPE", pipe_qty, "m", "fire pipe length from MAIN/BRANCH/DROP routes +12% allowance")

        elbows = max(1, total_bends)
        tees = max(1, branch_routes)
        couplings = max(1, math.ceil(pipe_qty / 6.0))
        hangers = max(1, math.ceil(pipe_qty / 3.0))

        _add_counter(counter, "PIPE_ELBOW", elbows, "nos", "estimated from generated route bends")
        _add_counter(counter, "PIPE_TEE", tees, "nos", "estimated from branch route count")
        _add_counter(counter, "PIPE_COUPLING", couplings, "nos", "estimated one coupling per 6 m of fire pipe")
        _add_counter(counter, "PIPE_HANGER", hangers, "nos", "estimated one hanger/support per 3 m of fire pipe")
        _add_counter(counter, "PIPE_FITTINGS_ALLOWANCE", 1, "lot", "fittings allowance for preliminary sprinkler pipework")

        if main_routes or branch_routes or drop_routes:
            _add_counter(counter, "CONTROL_VALVE", 1, "nos", "sprinkler system control valve allowance")
            _add_counter(counter, "FLOW_SWITCH", 1, "nos", "sprinkler waterflow switch allowance")
            _add_counter(counter, "TAMPER_SWITCH", 1, "nos", "sprinkler valve tamper switch allowance")
            _add_counter(counter, "PRESSURE_GAUGE", 2, "nos", "two pressure gauges allowance at riser/control valve")
            _add_counter(counter, "INSPECTOR_TEST_VALVE", 1, "nos", "inspector test/drain allowance")

    if fire_alarm_cable_m > 0:
        _add_counter(counter, "FIRE_ALARM_CABLE", round(fire_alarm_cable_m * 1.15, 2), "m", "SLC/NAC route length +15% allowance")

    if duct_m > 0:
        _add_counter(counter, "DUCT_ROUTE", round(duct_m * 1.10, 2), "m", "duct route length +10% allowance")

    if electrical_cable_m > 0:
        _add_counter(counter, "ELECTRICAL_CABLE", round(electrical_cable_m * 1.10, 2), "m", "electrical route length +10% allowance")

    if water_pipe_m > 0:
        _add_counter(counter, "WATER_PIPE", round(water_pipe_m * 1.10, 2), "m", "water pipe route length +10% allowance")

    if drain_pipe_m > 0:
        _add_counter(counter, "DRAIN_PIPE", round(drain_pipe_m * 1.10, 2), "m", "drain pipe route length +10% allowance")

    return _build_rows_from_counter(counter)


def build_accessory_takeoff_from_devices(devices: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Adds logical accessory rows from placed device counts.

    Examples:
    - one back box per fire alarm field device
    - one battery set for FACP
    """
    counter: defaultdict[tuple[str, str, str], float] = defaultdict(float)

    fire_alarm_field_devices = 0
    facp_count = 0

    for device in devices or []:
        if not isinstance(device, dict):
            continue

        dtype = extract_device_type(device)

        if dtype in {"SMOKE_DETECTOR", "HEAT_DETECTOR", "MANUAL_CALL_POINT", "HORN_STROBE"}:
            fire_alarm_field_devices += 1

        if dtype == "FIRE_ALARM_CONTROL_PANEL":
            facp_count += 1

    if fire_alarm_field_devices:
        _add_counter(
            counter,
            "FIRE_ALARM_BACKBOX",
            fire_alarm_field_devices,
            "nos",
            "one back box per generated fire alarm field device",
        )

    if facp_count:
        _add_counter(
            counter,
            "FIRE_ALARM_BATTERY",
            facp_count,
            "set",
            "one standby battery set per generated FACP",
        )

    return _build_rows_from_counter(counter)


def _merge_rows(row_groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    counter: defaultdict[tuple[str, str, str], float] = defaultdict(float)

    for rows in row_groups:
        for row in rows or []:
            if not isinstance(row, dict):
                continue

            dtype = str(row.get("device_type") or "UNKNOWN")
            unit = str(row.get("unit") or map_product(dtype)["unit"])
            basis = str(row.get("takeoff_basis") or row.get("mapping_status") or "")
            quantity = _safe_float(row.get("quantity"), 0.0)

            _add_counter(counter, dtype, quantity, unit, basis)

    return _build_rows_from_counter(counter)


def build_ceasefire_bom(report: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Main function called by routes.py.

    Priority:
    1. Map report["bom"] because it has summarized quantities.
    2. Add route takeoff rows for pipes/cables/fittings/accessories.
    3. Add device accessories from report["devices"].
    4. If BOM is missing, fallback to devices.
    """
    if not isinstance(report, dict):
        return []

    bom_rows = _candidate_bom_rows(report)
    device_rows = _candidate_device_rows(report)
    route_rows = _candidate_route_rows(report)

    mapped_groups: list[list[dict[str, Any]]] = []

    if bom_rows:
        mapped_groups.append(build_ceasefire_bom_from_existing_bom(bom_rows))
    elif device_rows:
        mapped_groups.append(build_ceasefire_bom_from_devices(device_rows))

    if route_rows:
        mapped_groups.append(build_route_takeoff(route_rows))

    if device_rows:
        mapped_groups.append(build_accessory_takeoff_from_devices(device_rows))

    if not mapped_groups:
        return []

    return _merge_rows(mapped_groups)


def bom_to_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return CSV_HEADER

    output = io.StringIO()
    fieldnames = [
        "item_no",
        "category",
        "device_type",
        "product_code",
        "description",
        "quantity",
        "unit",
        "unit_price",
        "total_price",
        "mapping_status",
        "takeoff_basis",
    ]

    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})

    return output.getvalue()


# Quick self-test:
# Run from project root:
#     python backend/bom_mapper.py
if __name__ == "__main__":
    sample_report = {
        "bom": [
            {"item": "Portable Fire Extinguisher", "type": "EXT", "quantity": 4, "unit": "nos"},
            {"item": "Fire Alarm Control Panel", "type": "FACP", "quantity": 1, "unit": "nos"},
            {"item": "Heat Detector", "type": "HD", "quantity": 10, "unit": "nos"},
            {"item": "Horn/Strobe", "type": "HS", "quantity": 6, "unit": "nos"},
            {"item": "Manual Call Point", "type": "MCP", "quantity": 6, "unit": "nos"},
            {"item": "Smoke Detector", "type": "SD", "quantity": 11, "unit": "nos"},
            {"item": "Fire Safety / Exit Signage", "type": "SIGN", "quantity": 4, "unit": "nos"},
            {"item": "Fire alarm cable route allowance +15%", "type": "FIRE_ALARM_CABLE_M", "quantity": 832.63, "unit": "m"},
        ],
        "devices": [
            {"type": "FACP"},
            {"type": "SD"},
            {"type": "MCP"},
            {"type": "HS"},
        ],
        "routes": [
            {"type": "MAIN", "length_m": 21},
            {"type": "BRANCH", "length_m": 18},
            {"type": "DROP", "length_m": 6},
            {"type": "SLC", "length_m": 120},
            {"type": "NAC", "length_m": 80},
        ],
    }

    mapped = build_ceasefire_bom(sample_report)
    print(bom_to_csv(mapped))