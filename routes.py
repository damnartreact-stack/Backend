from __future__ import annotations

import base64
import inspect
import io
import json
import zipfile
from copy import deepcopy
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from analysis import analyze_floor_plan
from config import Settings, get_settings
from constants import HAZARD_PROFILES, MODULES

try:
    from .constants import (
        FIREDESIGN_STYLE_FEATURES,
        REVIEW_GATE_DEFINITIONS,
        SPRINKLER_STANDARD_PROFILES,
    )
except Exception:  # pragma: no cover
    FIREDESIGN_STYLE_FEATURES = {}
    REVIEW_GATE_DEFINITIONS = []
    SPRINKLER_STANDARD_PROFILES = {}


try:
    from .bom_mapper import build_ceasefire_bom, bom_to_csv

    BOM_MAPPING_AVAILABLE = True
    BOM_MAPPING_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - keeps backend alive if file is missing
    BOM_MAPPING_AVAILABLE = False
    BOM_MAPPING_IMPORT_ERROR = str(exc)

    def build_ceasefire_bom(report: dict[str, Any]) -> list[dict[str, Any]]:
        return []

    def bom_to_csv(rows: list[dict[str, Any]]) -> str:
        return (
            "item_no,category,device_type,product_code,description,quantity,"
            "unit,unit_price,total_price,mapping_status,takeoff_basis\n"
        )


try:
    from .dxf_writer import build_dxf_download_payload

    DXF_EXPORT_AVAILABLE = True
    DXF_EXPORT_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - keeps backend alive if file is missing
    DXF_EXPORT_AVAILABLE = False
    DXF_EXPORT_IMPORT_ERROR = str(exc)

    def build_dxf_download_payload(
        report: dict[str, Any],
        original_file_bytes: bytes | None = None,
        original_filename: str | None = None,
    ) -> dict[str, Any]:
        raise RuntimeError(f"DXF export module is not available: {DXF_EXPORT_IMPORT_ERROR}")


router = APIRouter()


DISCIPLINE_ALIASES = {
    "full": "full_package",
    "full-package": "full_package",
    "full_package": "full_package",
    "mep": "full_package",
    "full_mep": "full_package",
    "full-mep": "full_package",
    "fire": "fire_alarm",
    "fire_alarm": "fire_alarm",
    "fire-alarm": "fire_alarm",
    "alarm": "fire_alarm",
    "sprinkler": "sprinklers",
    "sprinklers": "sprinklers",
    "hvac": "hvac",
    "electrical": "electrical",
    "electric": "electrical",
    "plumbing": "plumbing",
    "plumb": "plumbing",
}


HAZARD_ALIASES = {
    "light": "light",
    "light_hazard": "light",
    "light-hazard": "light",
    "light hazard": "light",
    "ordinary1": "ordinary_1",
    "ordinary_1": "ordinary_1",
    "ordinary-1": "ordinary_1",
    "ordinary 1": "ordinary_1",
    "ordinary hazard 1": "ordinary_1",
    "oh1": "ordinary_1",
    "ordinary2": "ordinary_2",
    "ordinary_2": "ordinary_2",
    "ordinary-2": "ordinary_2",
    "ordinary 2": "ordinary_2",
    "ordinary hazard 2": "ordinary_2",
    "oh2": "ordinary_2",
}


SYSTEM_TYPE_ALIASES = {
    "wet": "wet_pipe",
    "wet_pipe": "wet_pipe",
    "wet-pipe": "wet_pipe",
    "wet pipe": "wet_pipe",
    "dry": "dry_pipe",
    "dry_pipe": "dry_pipe",
    "dry-pipe": "dry_pipe",
    "dry pipe": "dry_pipe",
    "preaction": "preaction",
    "pre-action": "preaction",
    "pre action": "preaction",
    "deluge": "deluge",
}


SPRINKLER_STANDARD_ALIASES = {
    "nfpa13": "nfpa_13",
    "nfpa_13": "nfpa_13",
    "nfpa 13": "nfpa_13",
    "nfpa-13": "nfpa_13",
    "13": "nfpa_13",
    "commercial": "nfpa_13",
    "nfpa13r": "nfpa_13r",
    "nfpa_13r": "nfpa_13r",
    "nfpa 13r": "nfpa_13r",
    "nfpa-13r": "nfpa_13r",
    "13r": "nfpa_13r",
    "residential low-rise": "nfpa_13r",
    "nfpa13d": "nfpa_13d",
    "nfpa_13d": "nfpa_13d",
    "nfpa 13d": "nfpa_13d",
    "nfpa-13d": "nfpa_13d",
    "13d": "nfpa_13d",
    "dwelling": "nfpa_13d",
}


SUPPORTED_SYSTEM_TYPES = {"wet_pipe", "dry_pipe", "preaction", "deluge"}
RASTER_SUFFIXES = {"png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"}
DXF_SUFFIXES = {"dxf"}
DWG_SUFFIXES = {"dwg"}


def _normalise_suffix(filename: str) -> str:
    filename = filename or ""
    return filename.rsplit(".", 1)[-1].lower().strip() if "." in filename else ""


def _normalise_discipline(value: str) -> str:
    key = (value or "").strip().lower()
    return DISCIPLINE_ALIASES.get(key, key)


def _normalise_hazard(value: str) -> str:
    key = (value or "").strip().lower()
    return HAZARD_ALIASES.get(key, key)


def _normalise_system_type(value: str) -> str:
    key = (value or "").strip().lower()
    return SYSTEM_TYPE_ALIASES.get(key, key)


def _normalise_sprinkler_standard_profile(value: str) -> str:
    key = (value or "").strip().lower()
    return SPRINKLER_STANDARD_ALIASES.get(key, key or "nfpa_13")


def _safe_content_length(request: Request) -> int | None:
    raw = request.headers.get("content-length")

    if not raw:
        return None

    try:
        return int(raw)
    except Exception:
        return None


def _validate_positive_number(name: str, value: float) -> None:
    if value <= 0:
        raise HTTPException(
            status_code=400,
            detail=f"{name} must be greater than zero",
        )


def _validate_upload_size(size_bytes: int, settings: Settings) -> None:
    if size_bytes > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Upload must be {settings.max_upload_mb} MB or smaller",
        )


def _recommended_settings() -> dict[str, Any]:
    return {
        "for_png_jpg_test_plans": {
            "metres_per_pixel": 0.01,
            "min_room_area": 4.0,
            "socket_spacing": 4.0,
            "discipline": "full_package",
            "hazard_class": "light",
            "system_type": "wet_pipe",
            "sprinkler_standard_profile": "nfpa_13",
        },
        "for_higher_accuracy": [
            "Use high-resolution PNG/JPG or DXF.",
            "Keep room labels visible inside each room.",
            "Include known area text such as 32 m² or 48 m².",
            "Label wet areas as Pantry, Toilet, WC, Janitor or Cafeteria.",
            "Label Electrical DB, Server/IT, AHU, Fire Riser, Store and Stair clearly.",
            "For real 9/10 accuracy, DXF with clean WALL, ROOM, DOOR and MEP layers is preferred.",
        ],
        "firedesign_style_next_steps": [
            "Use a sprinkler-focused run when benchmarking against FireDesign.ai.",
            "Use NFPA 13 for commercial/office/retail/warehouse test cases.",
            "Use NFPA 13R/13D only when the residential scope is confirmed.",
            "Review pipe/fitting/accessory takeoff as an estimate, not procurement-ready output.",
        ],
    }


def _accuracy_guidance(suffix: str) -> dict[str, Any]:
    raster = suffix in RASTER_SUFFIXES
    dxf = suffix in DXF_SUFFIXES

    return {
        "expected_accuracy_mode": (
            "image_ocr_geometry"
            if raster
            else "cad_layer_geometry"
            if dxf
            else "unsupported_or_conversion_required"
        ),
        "notes": [
            "Raster PNG/JPG accuracy depends on OCR quality, wall clarity and visible room labels.",
            "DXF accuracy improves when WALL, ROOM, DOOR, TEXT and MEP layers are cleanly named.",
            "This system generates a feasibility/review package, not a final stamped fire design.",
            "Final sprinkler, alarm, HVAC, electrical and plumbing design must be reviewed by a qualified engineer.",
        ],
        "nine_plus_accuracy_requirements": [
            "Correct room boundary detection.",
            "Correct room name and area reading from CAD text or DXF data.",
            "Correct room classification: corridor, toilet, pantry, electrical, server, store and stair.",
            "Sprinkler count based on corrected room area and hazard class.",
            "Fire alarm detector type based on room use.",
            "HVAC separated into supply, return and exhaust logic.",
            "Plumbing connected only to true wet rooms.",
            "Routes following corridor/service trunks instead of crossing walls directly.",
        ],
    }


def _ensure_report(result: dict[str, Any]) -> dict[str, Any]:
    report = result.get("report")

    if not isinstance(report, dict):
        report = {}
        result["report"] = report

    return report


def _ensure_downloads(result: dict[str, Any]) -> dict[str, Any]:
    downloads = result.get("downloads")

    if not isinstance(downloads, dict):
        downloads = {}
        result["downloads"] = downloads

    return downloads


def _ensure_warnings(report: dict[str, Any]) -> list[str]:
    warnings = report.get("warnings")

    if not isinstance(warnings, list):
        warnings = []
        report["warnings"] = warnings

    return warnings


def _safe_error_text(exc: Exception) -> str:
    text = str(exc).strip()
    return text or exc.__class__.__name__


def _call_analyze_floor_plan(**kwargs: Any) -> dict[str, Any]:
    """
    Calls analyze_floor_plan safely.

    This lets routes.py pass newer settings such as
    sprinkler_standard_profile and occupancy_type only when analysis.py supports them.
    It avoids breaking older versions of analysis.py.
    """
    signature = inspect.signature(analyze_floor_plan)

    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )

    if accepts_kwargs:
        return analyze_floor_plan(**kwargs)

    accepted_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }

    return analyze_floor_plan(**accepted_kwargs)


def _decode_data_url_or_base64(value: str | None) -> bytes:
    if not value:
        return b""

    text = str(value)

    if "," in text and text.lower().startswith("data:"):
        text = text.split(",", 1)[1]

    try:
        return base64.b64decode(text)
    except Exception:
        return b""


def _make_json_safe(value: Any) -> Any:
    """
    Removes very large nested zip payloads before writing design_package.json.
    Keeps DXF payload metadata but avoids recursive export package bloat.
    """
    cloned = deepcopy(value)

    if isinstance(cloned, dict):
        downloads = cloned.get("downloads")
        if isinstance(downloads, dict):
            downloads.pop("export_package_zip", None)

        # Avoid repeated megabytes inside the report JSON saved into the ZIP.
        if isinstance(downloads, dict) and isinstance(downloads.get("updated_dxf"), dict):
            dxf = dict(downloads["updated_dxf"])
            if "content_base64" in dxf:
                dxf["content_base64"] = "[included separately as ceasefire_updated_layout.dxf]"
            downloads["updated_dxf"] = dxf

    return cloned


def _build_export_package_zip(result: dict[str, Any]) -> dict[str, Any]:
    """
    Creates a FireDesign-style deliverable ZIP in memory.

    Includes:
    - annotated PNG
    - SVG preview
    - standard BOM CSV
    - Ceasefire mapped BOM CSV
    - calculation schedule CSV
    - engineering report TXT
    - design package JSON
    - updated DXF if available
    """
    report = _ensure_report(result)
    downloads = _ensure_downloads(result)

    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as package:
        annotated_png = _decode_data_url_or_base64(result.get("annotated_png"))
        if annotated_png:
            package.writestr("01_annotated_layout.png", annotated_png)

        svg = downloads.get("svg")
        if svg:
            package.writestr("02_review_preview.svg", str(svg))

        engineering_report = downloads.get("engineering_report_txt")
        if engineering_report:
            package.writestr("03_engineering_report.txt", str(engineering_report))

        bom_csv = downloads.get("bom_csv")
        if bom_csv:
            package.writestr("04_standard_bom.csv", str(bom_csv))

        ceasefire_bom_csv = downloads.get("ceasefire_bom_csv")
        if ceasefire_bom_csv:
            package.writestr("05_ceasefire_mapped_bom.csv", str(ceasefire_bom_csv))

        hydraulic_csv = downloads.get("hydraulic_csv")
        if hydraulic_csv:
            package.writestr("06_calculation_schedule.csv", str(hydraulic_csv))

        updated_dxf = downloads.get("updated_dxf")
        if isinstance(updated_dxf, dict):
            dxf_bytes = _decode_data_url_or_base64(updated_dxf.get("content_base64"))
            if dxf_bytes:
                package.writestr(updated_dxf.get("filename") or "07_ceasefire_updated_layout.dxf", dxf_bytes)

        design_package = _make_json_safe(result)
        package.writestr(
            "08_design_package.json",
            json.dumps(design_package, indent=2, ensure_ascii=False, default=str),
        )

        manifest = {
            "package_name": "FireDesign Automation POC Export Package",
            "disclaimer": "POC only. Engineer/AHJ review required before client submission, procurement or construction.",
            "included_files": [
                "01_annotated_layout.png",
                "02_review_preview.svg",
                "03_engineering_report.txt",
                "04_standard_bom.csv",
                "05_ceasefire_mapped_bom.csv",
                "06_calculation_schedule.csv",
                "07_ceasefire_updated_layout.dxf when available",
                "08_design_package.json",
            ],
            "summary": report.get("summary", {}),
            "ceasefire_bom_status": result.get("ceasefire_bom_status", {}),
            "dxf_export_status": result.get("dxf_export_status", {}),
        }
        package.writestr("00_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False, default=str))

    data = buffer.getvalue()

    return {
        "filename": "firedesign_ceasefire_export_package.zip",
        "content_base64": base64.b64encode(data).decode("ascii"),
        "mime_type": "application/zip",
        "size_bytes": len(data),
        "note": "Contains preview, report, BOM files, design JSON and updated DXF when available.",
    }


def _add_export_package_output(result: dict[str, Any]) -> None:
    report = _ensure_report(result)
    downloads = _ensure_downloads(result)
    warnings = _ensure_warnings(report)

    try:
        payload = _build_export_package_zip(result)
        downloads["export_package_zip"] = payload
        result["export_package_status"] = {
            "status": "created",
            "filename": payload["filename"],
            "size_bytes": payload["size_bytes"],
        }
    except Exception as exc:
        reason = _safe_error_text(exc)
        warnings.append(f"Export package ZIP failed: {reason}")
        result["export_package_status"] = {
            "status": "failed",
            "reason": reason,
        }


def _add_ceasefire_bom_outputs(result: dict[str, Any]) -> None:
    """
    Adds a Ceasefire-style mapped BOM without removing the existing backend BOM.
    """
    report = _ensure_report(result)
    downloads = _ensure_downloads(result)
    warnings = _ensure_warnings(report)

    if not BOM_MAPPING_AVAILABLE:
        report.setdefault("ceasefire_bom", [])
        downloads.setdefault("ceasefire_bom_csv", bom_to_csv([]))
        warnings.append(
            "Ceasefire BOM mapping skipped because backend/bom_mapper.py is missing or could not be imported."
        )
        result["ceasefire_bom_status"] = {
            "status": "skipped",
            "rows": 0,
            "reason": BOM_MAPPING_IMPORT_ERROR,
        }
        return

    try:
        ceasefire_bom = build_ceasefire_bom(report)
        report["ceasefire_bom"] = ceasefire_bom
        downloads["ceasefire_bom_csv"] = bom_to_csv(ceasefire_bom)

        if ceasefire_bom:
            status = "created"
            note = "Dummy Ceasefire-style product codes and prices for POC only."
        else:
            status = "empty"
            note = "No report.devices, report.routes or report.bom rows were available for Ceasefire mapping."
            warnings.append(
                "Ceasefire mapped BOM is empty because the analysis output did not include devices, routes or BOM rows."
            )

        result["ceasefire_bom_status"] = {
            "status": status,
            "rows": len(ceasefire_bom),
            "note": note,
        }
    except Exception as exc:
        reason = _safe_error_text(exc)
        report.setdefault("ceasefire_bom", [])
        downloads.setdefault("ceasefire_bom_csv", bom_to_csv([]))
        warnings.append(f"Ceasefire BOM mapping failed: {reason}")
        result["ceasefire_bom_status"] = {
            "status": "failed",
            "rows": 0,
            "reason": reason,
        }


def _add_updated_dxf_output(
    result: dict[str, Any],
    original_file_bytes: bytes,
    original_filename: str,
) -> None:
    """
    Adds updated DXF export.

    For original DXF files, it overlays generated device blocks on the original drawing.
    For raster files, it creates a proof DXF containing rooms, devices and routes.
    """
    report = _ensure_report(result)
    downloads = _ensure_downloads(result)
    warnings = _ensure_warnings(report)

    if not DXF_EXPORT_AVAILABLE:
        warnings.append(
            "Updated DXF export skipped because backend/dxf_writer.py is missing or could not be imported."
        )
        result["cad_export"] = {
            "status": "skipped",
            "reason": DXF_EXPORT_IMPORT_ERROR,
        }
        result["dxf_export_status"] = {
            "available": False,
            "reason": DXF_EXPORT_IMPORT_ERROR,
        }
        return

    try:
        payload = build_dxf_download_payload(
            report=report,
            original_file_bytes=original_file_bytes,
            original_filename=original_filename,
        )

        # Object shape used by the upgraded frontend.
        downloads["updated_dxf"] = payload

        # Legacy/flat keys kept for compatibility.
        downloads["updated_dxf_base64"] = payload.get("content_base64", "")
        downloads["updated_dxf_filename"] = payload.get("filename", "ceasefire_updated_layout.dxf")
        downloads["updated_dxf_mime_type"] = payload.get("mime_type", "application/dxf")
        downloads["updated_dxf_note"] = payload.get("note", "")

        result["cad_export"] = {
            "status": "created",
            "filename": payload.get("filename", "ceasefire_updated_layout.dxf"),
            "layer_prefix": "CEASEFIRE_",
            "layers": payload.get("layers", []),
            "note": payload.get("note", ""),
        }
        result["dxf_export_status"] = {
            "available": True,
            "message": "Updated DXF generated successfully",
            "filename": payload.get("filename", "ceasefire_updated_layout.dxf"),
            "layers": payload.get("layers", []),
        }
    except Exception as exc:
        reason = _safe_error_text(exc)
        warnings.append(f"Updated DXF export failed: {reason}")
        result["cad_export"] = {
            "status": "failed",
            "reason": reason,
        }
        result["dxf_export_status"] = {
            "available": False,
            "reason": reason,
        }


def _add_research_brief_metadata(
    result: dict[str, Any],
    suffix: str,
    sprinkler_standard_profile: str,
    occupancy_type: str,
) -> None:
    result.setdefault(
        "research_brief_alignment",
        {
            "project": "AI for Automated Fire-Safety CAD Drawings + Bill of Materials",
            "use_case": "Ceasefire Design-Team feasibility POC",
            "benchmark_reference": "FireDesign.ai-style sprinkler workflow comparison",
            "input_route": (
                "DXF vector parsing"
                if suffix in DXF_SUFFIXES
                else "Raster image CV/OCR"
                if suffix in RASTER_SUFFIXES
                else "Conversion required"
            ),
            "sprinkler_standard_profile": sprinkler_standard_profile,
            "occupancy_type": occupancy_type,
            "outputs_created": [
                "annotated PNG/SVG preview",
                "standard generated BOM",
                "Ceasefire-style mapped BOM CSV",
                "updated DXF export when dxf_writer is available",
                "export package ZIP",
                "engineering/review warnings",
            ],
            "limitations": [
                "Dummy product codes/prices are used until Ceasefire provides a product master.",
                "DXF device blocks are POC symbols until official Ceasefire CAD blocks are provided.",
                "Hydraulic values are preliminary seeds unless a proper solver is connected.",
                "Output requires engineer/AHJ review before client submission or construction use.",
            ],
        },
    )


def _add_firedesign_style_metadata(result: dict[str, Any]) -> None:
    result.setdefault(
        "firedesign_style_features",
        {
            "feature_matrix": FIREDESIGN_STYLE_FEATURES,
            "review_gate_count": len(REVIEW_GATE_DEFINITIONS),
            "review_gate_definitions": REVIEW_GATE_DEFINITIONS,
            "sprinkler_standard_profiles": SPRINKLER_STANDARD_PROFILES,
            "implemented_now": [
                "floor-plan upload",
                "room/zone detection",
                "device placement",
                "route seed generation",
                "BOM generation",
                "Ceasefire-style mapped BOM",
                "updated DXF export",
                "export package ZIP",
                "review warnings",
            ],
            "future_production_items": [
                "native DWG processing through APS/ODA workflow",
                "true hydraulic solver",
                "official Ceasefire CAD block library",
                "official product master/pricing",
                "project history and team workspace",
                "HydroCAD/AutoSPRINK export",
            ],
        },
    )


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/api/status")
def api_status(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    return {
        "status": "connected",
        "service": settings.app_name,
        "environment": settings.environment,
        "accepted_files": sorted(settings.allowed_extensions),
        "max_upload_mb": settings.max_upload_mb,
        "modules": MODULES,
        "hazard_profiles": HAZARD_PROFILES,
        "sprinkler_standard_profiles": SPRINKLER_STANDARD_PROFILES,
        "review_gate_count": len(REVIEW_GATE_DEFINITIONS),
        "review_gate_definitions": REVIEW_GATE_DEFINITIONS,
        "firedesign_style_features": FIREDESIGN_STYLE_FEATURES,
        "capabilities": {
            "ceasefire_bom_mapping": BOM_MAPPING_AVAILABLE,
            "ceasefire_bom_mapping_error": BOM_MAPPING_IMPORT_ERROR,
            "updated_dxf_export": DXF_EXPORT_AVAILABLE,
            "updated_dxf_export_error": DXF_EXPORT_IMPORT_ERROR,
            "export_package_zip": True,
            "dwg_direct_parsing": False,
            "dxf_overlay_export": DXF_EXPORT_AVAILABLE,
            "raster_to_poc_dxf_export": DXF_EXPORT_AVAILABLE,
            "native_hydraulic_solver": False,
        },
        "poc_outputs": {
            "drawing_preview": ["annotated_png", "downloads.svg"],
            "cad_export": ["downloads.updated_dxf"],
            "bom": ["report.bom", "report.ceasefire_bom", "downloads.ceasefire_bom_csv"],
            "package": ["downloads.export_package_zip"],
            "review": ["report.compliance_checks", "report.warnings"],
        },
        "endpoints": {
            "analyze": f"{settings.api_prefix}/analyze",
            "health": "/health",
            "docs": "/docs",
        },
        "recommended_test_settings": _recommended_settings(),
    }


@router.post("/api/analyze")
async def analyze(
    request: Request,
    file: UploadFile = File(...),
    metres_per_pixel: float = Form(0.01),
    min_room_area: float = Form(4.0),
    socket_spacing: float = Form(4.0),
    discipline: str = Form("full_package"),
    standard: str = Form("NFPA 13 + NFPA 72 + ASHRAE/NEC/IPC-style workflow"),
    hazard_class: str = Form("light"),
    system_type: str = Form("wet_pipe"),
    sprinkler_standard_profile: str = Form("nfpa_13"),
    occupancy_type: str = Form("office_commercial"),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    content_length = _safe_content_length(request)

    if content_length is not None:
        _validate_upload_size(content_length, settings)

    filename = file.filename or ""
    suffix = _normalise_suffix(filename)

    if suffix not in settings.allowed_extensions:
        allowed = ", ".join(sorted(settings.allowed_extensions)).upper()
        raise HTTPException(
            status_code=400,
            detail=f"Upload one of these file types: {allowed}",
        )

    if suffix in DWG_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "DWG cannot be parsed directly in this POC.",
                "reason": "DWG is a proprietary AutoCAD format. This backend currently works best with DXF or clear raster images.",
                "recommended_fix": [
                    "Convert the DWG to DXF using ODA File Converter, AutoCAD, LibreCAD/QCAD where possible, or Autodesk Platform Services.",
                    "Upload the converted DXF file.",
                    "For a quick demo, upload a high-resolution PNG/JPG floor plan instead.",
                ],
            },
        )

    _validate_positive_number("metres_per_pixel", metres_per_pixel)
    _validate_positive_number("min_room_area", min_room_area)
    _validate_positive_number("socket_spacing", socket_spacing)

    discipline = _normalise_discipline(discipline)
    hazard_class = _normalise_hazard(hazard_class)
    system_type = _normalise_system_type(system_type)
    sprinkler_standard_profile = _normalise_sprinkler_standard_profile(sprinkler_standard_profile)
    occupancy_type = (occupancy_type or "office_commercial").strip().lower()

    if discipline not in MODULES:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Unsupported discipline '{discipline}'",
                "allowed_values": list(MODULES.keys()),
            },
        )

    if hazard_class not in HAZARD_PROFILES:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Unsupported hazard_class '{hazard_class}'",
                "allowed_values": list(HAZARD_PROFILES.keys()),
            },
        )

    if system_type not in SUPPORTED_SYSTEM_TYPES:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Unsupported system_type '{system_type}'",
                "allowed_values": sorted(SUPPORTED_SYSTEM_TYPES),
            },
        )

    if SPRINKLER_STANDARD_PROFILES and sprinkler_standard_profile not in SPRINKLER_STANDARD_PROFILES:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Unsupported sprinkler_standard_profile '{sprinkler_standard_profile}'",
                "allowed_values": list(SPRINKLER_STANDARD_PROFILES.keys()),
            },
        )

    standard = (standard or "").strip()

    if not standard:
        standard = "NFPA 13 + NFPA 72 + ASHRAE/NEC/IPC-style workflow"

    if sprinkler_standard_profile in SPRINKLER_STANDARD_PROFILES:
        standard_label = SPRINKLER_STANDARD_PROFILES[sprinkler_standard_profile].get("short_label")
        if standard_label and standard_label not in standard:
            standard = f"{standard_label} + {standard}"

    data = await file.read()

    if not data:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty",
        )

    _validate_upload_size(len(data), settings)

    try:
        result = _call_analyze_floor_plan(
    data=data,
    suffix=suffix,
    metres_per_pixel=metres_per_pixel,
    min_room_area=min_room_area,
    socket_spacing=socket_spacing,
    discipline=discipline,
    standard=standard,
    hazard_class=hazard_class,
    system_type=system_type,
    sprinkler_standard_profile=sprinkler_standard_profile,
    occupancy_type=occupancy_type,
)

        if not isinstance(result, dict):
            raise TypeError("analyze_floor_plan must return a dictionary")

        result.setdefault(
            "request_settings",
            {
                "filename": filename,
                "file_type": suffix,
                "metres_per_pixel": metres_per_pixel,
                "min_room_area": min_room_area,
                "socket_spacing": socket_spacing,
                "discipline": discipline,
                "standard": standard,
                "hazard_class": hazard_class,
                "system_type": system_type,
                "sprinkler_standard_profile": sprinkler_standard_profile,
                "occupancy_type": occupancy_type,
            },
        )
        result.setdefault("accuracy_guidance", _accuracy_guidance(suffix))

        # Make sure required containers exist before enrichment.
        _ensure_report(result)
        _ensure_downloads(result)

        # Enrich response for Ceasefire / FireDesign-style deliverables.
        _add_ceasefire_bom_outputs(result)
        _add_updated_dxf_output(
            result=result,
            original_file_bytes=data,
            original_filename=filename,
        )
        _add_research_brief_metadata(
            result=result,
            suffix=suffix,
            sprinkler_standard_profile=sprinkler_standard_profile,
            occupancy_type=occupancy_type,
        )
        _add_firedesign_style_metadata(result)

        # Build the ZIP after BOM and DXF payloads are present.
        _add_export_package_output(result)

        return result

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Could not analyze drawing",
                "reason": str(exc),
                "file_type": suffix,
                "discipline": discipline,
                "recommended_fix": [
                    "Check that backend/analysis.py has the latest FireDesign-style upgrade code.",
                    "Check that backend/constants.py includes SPRINKLER_STANDARD_PROFILES and REVIEW_GATE_DEFINITIONS.",
                    "Check that backend/bom_mapper.py exists if Ceasefire BOM mapping is required.",
                    "Check that backend/dxf_writer.py exists if updated DXF export is required.",
                    "Check that OpenCV, NumPy, Pandas, Pillow, ezdxf and pytesseract are installed.",
                    "Install the Tesseract OCR executable if OCR is enabled.",
                    "Try a high-resolution PNG or DXF with clear room labels.",
                ],
            },
        ) from exc