"""
Project constants for FireDesign Automation.

This file is intentionally data-heavy and dependency-free.

It supports:
- multi-discipline MEP/fire feasibility output
- FireDesign.ai-style sprinkler workflow metadata
- Ceasefire-style BOM mapping
- NFPA 13 / 13R / 13D rule-profile labels
- 35+ fail-closed review gates
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Device / route names
# ---------------------------------------------------------------------------

DEVICE_NAMES = {
    # Fire alarm
    "FACP": "Fire Alarm Control Panel",
    "PANEL": "Fire Alarm Control Panel",
    "ANNUNCIATOR": "Remote Annunciator Panel",
    "SD": "Smoke Detector",
    "HD": "Heat Detector",
    "MCP": "Manual Call Point",
    "HS": "Horn/Strobe",
    "SOUNDER": "Sounder",
    "STROBE": "Visual Alarm Strobe",
    "NAC": "Notification Appliance Circuit",
    "SLC": "Signalling Line Circuit",
    "FIRE_ALARM_CABLE_M": "Fire Alarm Cable",
    "FIRE_ALARM_BACKBOX": "Fire Alarm Back Box",
    "FIRE_ALARM_BATTERY": "Fire Alarm Standby Battery",

    # Fire safety / egress
    "SIGN": "Fire Safety / Exit Signage",
    "EXT": "Portable Fire Extinguisher",
    "EM_LIGHT": "Emergency Light",
    "EXIT_SIGN": "Exit Sign",
    "FIRE_DOOR_SIGN": "Fire Door Sign",
    "EVAC_MAP": "Evacuation Route Map",

    # Sprinklers
    "SP": "Automatic Sprinkler Head",
    "SPRINKLER_HEAD": "Automatic Sprinkler Head",
    "RISER": "Sprinkler Riser",
    "BRANCH": "Branch Line Pipe",
    "MAIN": "Cross Main Pipe",
    "DROP": "Sprinkler Drop Pipe",
    "PIPE": "Total Sprinkler Pipe Allowance",
    "VALVE": "Control Valve Assembly",
    "CONTROL_VALVE": "Control Valve Assembly",
    "FITTING": "Pipe Fittings Allowance",
    "ELBOW": "Pipe Elbow",
    "TEE": "Pipe Tee",
    "COUPLING": "Pipe Coupling",
    "HANGER": "Pipe Hanger / Support",
    "FLOW_SWITCH": "Water Flow Switch",
    "TAMPER_SWITCH": "Valve Tamper Switch",
    "CHECK_VALVE": "Check Valve",
    "DRAIN_VALVE": "Drain Valve",
    "TEST_VALVE": "Inspector Test Valve",
    "PRESSURE_GAUGE": "Pressure Gauge",
    "ALARM_VALVE": "Alarm Valve",
    "FDC": "Fire Department Connection",

    # Hydrant / hose reel
    "HYDRANT": "Fire Hydrant Point",
    "HOSE_REEL": "Fire Hose Reel",
    "LANDING_VALVE": "Landing Valve",
    "FIRE_PUMP": "Fire Pump Placeholder",

    # HVAC
    "AHU": "Air Handling Unit / Indoor Unit",
    "SDIFF": "Supply Air Diffuser",
    "RGRILLE": "Return Air Grille",
    "EXH": "Exhaust Fan / Extract Grille",
    "EXFAN": "Exhaust Fan / Extract Grille",
    "DUCT_MAIN": "Main Duct Route",
    "DUCT_BRANCH": "Branch Duct Route",
    "RETURN_DUCT": "Return Air Duct Route",
    "EXHAUST_DUCT": "Exhaust Duct Route",
    "FIRE_DAMPER": "Fire Damper",
    "SMOKE_DAMPER": "Smoke Damper",

    # Electrical
    "EDB": "Electrical Distribution Board",
    "MDB": "Main Distribution Board",
    "LIGHT": "LED Light Fixture",
    "SWITCH": "Light Switch",
    "SO": "Socket Outlet",
    "POWER_SOCKET": "Power Socket Outlet",
    "LIGHTING_CIRCUIT": "Lighting Circuit Route",
    "POWER_CIRCUIT": "Power Circuit Route",
    "EMERGENCY_CIRCUIT": "Emergency Lighting Circuit",
    "CABLE_TRAY": "Cable Tray Route",
    "CONDUIT": "Electrical Conduit Route",

    # Plumbing
    "PLUMBING_RISER": "Plumbing Riser / Shaft",
    "LAV": "Wash Basin / Lavatory Point",
    "SINK": "Sink Point",
    "WC": "WC Fixture Point",
    "URINAL": "Urinal Fixture Point",
    "FD": "Floor Drain",
    "WATER_PIPE": "Domestic Water Pipe Route",
    "DRAIN_PIPE": "Drainage Pipe Route",
    "VENT_PIPE": "Plumbing Vent Pipe Route",

    # Review / annotation
    "REVIEW": "Manual Review Note",
    "WARNING": "Review Warning",
}


# ---------------------------------------------------------------------------
# Drawing colors in RGB order for OpenCV drawing
# ---------------------------------------------------------------------------

COLORS = {
    # Fire alarm
    "FACP": (190, 0, 0),
    "PANEL": (190, 0, 0),
    "ANNUNCIATOR": (180, 20, 20),
    "SD": (0, 90, 220),
    "HD": (220, 120, 0),
    "MCP": (220, 0, 0),
    "HS": (130, 0, 180),
    "SOUNDER": (130, 0, 180),
    "STROBE": (160, 0, 190),
    "NAC": (160, 0, 190),
    "SLC": (190, 0, 0),
    "FIRE_ALARM_CABLE_M": (190, 0, 0),
    "FIRE_ALARM_BACKBOX": (190, 0, 0),
    "FIRE_ALARM_BATTERY": (190, 0, 0),

    # Fire safety / egress
    "SIGN": (34, 197, 94),
    "EXT": (220, 38, 38),
    "EM_LIGHT": (234, 179, 8),
    "EXIT_SIGN": (34, 197, 94),
    "FIRE_DOOR_SIGN": (34, 197, 94),
    "EVAC_MAP": (34, 197, 94),

    # Sprinklers
    "SP": (0, 108, 255),
    "SPRINKLER_HEAD": (0, 108, 255),
    "RISER": (220, 20, 60),
    "BRANCH": (42, 111, 151),
    "MAIN": (13, 71, 161),
    "DROP": (38, 166, 154),
    "PIPE": (42, 111, 151),
    "VALVE": (220, 20, 60),
    "CONTROL_VALVE": (220, 20, 60),
    "FITTING": (38, 166, 154),
    "ELBOW": (38, 166, 154),
    "TEE": (38, 166, 154),
    "COUPLING": (38, 166, 154),
    "HANGER": (100, 116, 139),
    "FLOW_SWITCH": (220, 20, 60),
    "TAMPER_SWITCH": (220, 20, 60),
    "CHECK_VALVE": (220, 20, 60),
    "DRAIN_VALVE": (220, 20, 60),
    "TEST_VALVE": (220, 20, 60),
    "PRESSURE_GAUGE": (220, 20, 60),
    "ALARM_VALVE": (220, 20, 60),
    "FDC": (220, 20, 60),

    # Hydrant / hose reel
    "HYDRANT": (220, 38, 38),
    "HOSE_REEL": (220, 38, 38),
    "LANDING_VALVE": (220, 38, 38),
    "FIRE_PUMP": (220, 38, 38),

    # HVAC
    "AHU": (10, 132, 120),
    "SDIFF": (22, 163, 74),
    "RGRILLE": (34, 197, 94),
    "EXH": (2, 132, 199),
    "EXFAN": (2, 132, 199),
    "DUCT_MAIN": (13, 148, 136),
    "DUCT_BRANCH": (45, 212, 191),
    "RETURN_DUCT": (34, 197, 94),
    "EXHAUST_DUCT": (2, 132, 199),
    "FIRE_DAMPER": (239, 68, 68),
    "SMOKE_DAMPER": (239, 68, 68),

    # Electrical
    "EDB": (245, 158, 11),
    "MDB": (245, 120, 11),
    "LIGHT": (234, 179, 8),
    "SWITCH": (202, 138, 4),
    "SO": (217, 119, 6),
    "POWER_SOCKET": (217, 119, 6),
    "LIGHTING_CIRCUIT": (202, 138, 4),
    "POWER_CIRCUIT": (194, 65, 12),
    "EMERGENCY_CIRCUIT": (234, 179, 8),
    "CABLE_TRAY": (194, 65, 12),
    "CONDUIT": (194, 65, 12),

    # Plumbing
    "PLUMBING_RISER": (2, 132, 199),
    "LAV": (14, 165, 233),
    "SINK": (14, 165, 233),
    "WC": (3, 105, 161),
    "URINAL": (3, 105, 161),
    "FD": (56, 189, 248),
    "WATER_PIPE": (2, 132, 199),
    "DRAIN_PIPE": (71, 85, 105),
    "VENT_PIPE": (100, 116, 139),

    # Review / annotation
    "REVIEW": (220, 38, 38),
    "WARNING": (220, 38, 38),
}


# ---------------------------------------------------------------------------
# Module metadata shown on frontend and /api/status
# ---------------------------------------------------------------------------

MODULES = {
    "full_package": {
        "label": "Full MEP + Fire Package",
        "status": "Sprinklers, fire alarm, HVAC, electrical and plumbing generated together for feasibility review",
        "standards": [
            "NFPA 13",
            "NFPA 72",
            "ASHRAE-style workflow",
            "NEC-style workflow",
            "IPC/NPC-style workflow",
        ],
        "output_scope": [
            "Area classification",
            "Sprinkler layout and pipe route seed",
            "Fire alarm device and circuit route seed",
            "HVAC diffuser, return and exhaust seed",
            "Electrical lighting, socket, switch and DB route seed",
            "Plumbing wet-area fixture and riser route seed",
            "BOM and review notes",
            "Updated DXF export with CEASEFIRE layers",
            "Ceasefire-style mapped BOM",
        ],
        "accuracy_priority": "Highest when room labels, areas and service-room names are visible.",
    },
    "sprinklers": {
        "label": "Sprinklers",
        "status": "Automated sprinkler layout, routing, hydraulic seed schedule, material takeoff and compliance review",
        "standards": ["NFPA 13", "NFPA 13R", "NFPA 13D"],
        "output_scope": [
            "Sprinkler head count",
            "Coverage per head",
            "Preliminary demand",
            "Branch/main/drop routes",
            "Pipe diameter grouping",
            "Pipe/fitting/accessory takeoff",
            "Hazard-class review notes",
            "Updated DXF with sprinkler blocks",
        ],
        "accuracy_priority": "Depends on correct room area, hazard classification, ceiling, obstructions and standard profile.",
    },
    "fire_alarm": {
        "label": "Fire Alarms",
        "status": "Device placement, detector type selection, circuit routing and review package",
        "standards": ["NFPA 72"],
        "output_scope": [
            "Smoke/heat detector seed",
            "Manual call point seed",
            "Horn/strobe seed",
            "FACP seed",
            "SLC/NAC circuit routes",
            "Egress review notes",
            "Fire alarm cable allowance",
        ],
        "accuracy_priority": "Depends on correct room type: office, pantry, toilet, electrical, server, store and corridor.",
    },
    "hvac": {
        "label": "HVAC",
        "status": "Supply, return, exhaust, duct seed layout and preliminary load schedule",
        "standards": ["ASHRAE-style workflow", "SMACNA-style workflow"],
        "output_scope": [
            "AHU/source seed",
            "Supply diffuser seed",
            "Return grille seed",
            "Toilet/pantry exhaust seed",
            "Duct route allowance",
            "Cooling load placeholder",
        ],
        "accuracy_priority": "Depends on correct separation of occupied rooms, wet rooms, stair, server and electrical rooms.",
    },
    "electrical": {
        "label": "Electrical",
        "status": "Lighting, sockets, switching, panel seed and circuit route schedule",
        "standards": ["NEC-style workflow", "IEC-style workflow"],
        "output_scope": [
            "Electrical DB seed",
            "Lighting points",
            "Switch points",
            "Socket points",
            "Lighting and power routes",
            "Connected load placeholder",
        ],
        "accuracy_priority": "Depends on room perimeter, door side and DB room detection.",
    },
    "plumbing": {
        "label": "Plumbing",
        "status": "Wet-area fixture points, water/drain routes and plumbing BOM seed",
        "standards": ["IPC/NPC-style workflow", "NBC-style workflow"],
        "output_scope": [
            "Toilet fixture seed",
            "Pantry/janitor sink seed",
            "Floor drain seed",
            "Plumbing riser seed",
            "Water pipe route",
            "Drain pipe route",
        ],
        "accuracy_priority": "Depends on correct wet-area detection. Plumbing should not be generated in dry rooms.",
    },
}


# ---------------------------------------------------------------------------
# Sprinkler hazard profiles
# Kept compatible with existing analysis.py keys:
# max_spacing_m, max_coverage_m2, design_density_mm_min, k_factor_lpm_sqrtbar
# ---------------------------------------------------------------------------

HAZARD_PROFILES = {
    "light": {
        "label": "Light Hazard",
        "description": "Typical offices, reception, meeting rooms, training rooms, corridors and similar low-combustibility occupancies.",
        "max_spacing_m": 4.57,
        "max_coverage_m2": 20.9,
        "design_density_mm_min": 4.1,
        "k_factor_lpm_sqrtbar": 80,
        "remote_area_heads": 12,
        "recommended_standard": "NFPA 13",
        "recommended_rooms": [
            "office",
            "meeting",
            "training",
            "reception",
            "lobby",
            "corridor",
            "workstations",
        ],
        "review_note": "Confirm local code, ceiling type, obstruction and actual occupancy before final design.",
    },
    "ordinary_1": {
        "label": "Ordinary Hazard Group 1",
        "description": "Moderate combustibility or moderate heat-release areas such as light storage, service rooms and some retail/back-of-house spaces.",
        "max_spacing_m": 4.57,
        "max_coverage_m2": 12.1,
        "design_density_mm_min": 6.1,
        "k_factor_lpm_sqrtbar": 80,
        "remote_area_heads": 15,
        "recommended_standard": "NFPA 13",
        "recommended_rooms": [
            "store",
            "storage",
            "packing",
            "dispatch",
            "retail service area",
            "ordinary hazard area",
        ],
        "review_note": "Use stricter coverage than light hazard where storage or combustible loading is detected.",
    },
    "ordinary_2": {
        "label": "Ordinary Hazard Group 2",
        "description": "Higher ordinary hazard areas such as bulk storage, dense racks, loading/service areas or higher heat-release spaces.",
        "max_spacing_m": 4.57,
        "max_coverage_m2": 9.3,
        "design_density_mm_min": 8.1,
        "k_factor_lpm_sqrtbar": 115,
        "remote_area_heads": 18,
        "recommended_standard": "NFPA 13",
        "recommended_rooms": [
            "bulk storage",
            "rack storage",
            "loading dock",
            "warehouse",
            "high combustible service area",
        ],
        "review_note": "Storage commodity, rack height, ceiling height and obstruction rules must be confirmed by fire engineer.",
    },
}


# ---------------------------------------------------------------------------
# NFPA / sprinkler standard workflow profiles
# These are workflow profiles, not legal/code certification.
# ---------------------------------------------------------------------------

SPRINKLER_STANDARD_PROFILES = {
    "nfpa_13": {
        "label": "NFPA 13 Commercial Sprinkler Workflow",
        "short_label": "NFPA 13",
        "description": "Commercial/industrial workflow with hazard-based coverage, pipe route and review checks.",
        "allowed_hazard_classes": ["light", "ordinary_1", "ordinary_2"],
        "default_hazard_class": "light",
        "default_remote_area_heads": 12,
        "requires_hydraulic_review": True,
        "requires_engineer_review": True,
        "notes": [
            "Use for offices, commercial buildings, retail, warehouses and mixed-use projects.",
            "Commodity, ceiling height, obstructions and storage height must be engineer-confirmed.",
        ],
    },
    "nfpa_13r": {
        "label": "NFPA 13R Residential Low-Rise Workflow",
        "short_label": "NFPA 13R",
        "description": "Residential low-rise workflow with simplified residential review assumptions.",
        "allowed_hazard_classes": ["light"],
        "default_hazard_class": "light",
        "default_remote_area_heads": 4,
        "requires_hydraulic_review": True,
        "requires_engineer_review": True,
        "notes": [
            "Use only where residential low-rise scope is confirmed.",
            "Common spaces, parking, storage and service rooms may require NFPA 13 treatment.",
        ],
    },
    "nfpa_13d": {
        "label": "NFPA 13D One-/Two-Family Residential Workflow",
        "short_label": "NFPA 13D",
        "description": "Small residential workflow for feasibility study only.",
        "allowed_hazard_classes": ["light"],
        "default_hazard_class": "light",
        "default_remote_area_heads": 2,
        "requires_hydraulic_review": True,
        "requires_engineer_review": True,
        "notes": [
            "Use only for one-/two-family dwelling proof-of-concept cases.",
            "Water supply, domestic integration and local amendments require review.",
        ],
    },
}


# ---------------------------------------------------------------------------
# Room classification by text/OCR/CAD labels
# ---------------------------------------------------------------------------

ROOM_CLASSIFICATION_RULES = {
    "corridor": {
        "keywords": ["CORRIDOR", "EGRESS", "EXIT ACCESS", "PASSAGE", "HALLWAY"],
        "area_class": "egress_corridor",
        "semantic_type": "corridor",
        "fire_alarm_detector": "SD",
        "hvac_strategy": "linear supply/transfer-air review",
        "plumbing_allowed": False,
        "sprinkler_review": "Corridor coverage and spacing required.",
    },
    "toilet": {
        "keywords": ["TOILET", "WC", "RESTROOM", "WASHROOM", "BATH"],
        "area_class": "toilet_wet_area",
        "semantic_type": "wet_area",
        "fire_alarm_detector": "HD",
        "hvac_strategy": "exhaust required; avoid normal return-air route",
        "plumbing_allowed": True,
        "sprinkler_review": "Confirm small-room sprinkler exception or coverage requirement with local code.",
    },
    "pantry": {
        "keywords": ["PANTRY", "KITCHEN", "CAFETERIA", "JANITOR", "SINK", "WET AREA"],
        "area_class": "pantry_wet_area",
        "semantic_type": "wet_area",
        "fire_alarm_detector": "HD",
        "hvac_strategy": "exhaust/ventilation review",
        "plumbing_allowed": True,
        "sprinkler_review": "Heat detector and sprinkler coverage should be manually reviewed.",
    },
    "electrical": {
        "keywords": ["ELECTRICAL", "DB", "MDB", "PANEL", "SWITCHGEAR", "ELECT ROOM"],
        "area_class": "electrical_room_review",
        "semantic_type": "electrical_room",
        "fire_alarm_detector": "HD",
        "hvac_strategy": "dedicated ventilation/cooling review",
        "plumbing_allowed": False,
        "sprinkler_review": "Special fire protection and water exposure review required.",
    },
    "server": {
        "keywords": ["SERVER", "IT", "CONTROL", "DATA", "UPS"],
        "area_class": "server_it_review",
        "semantic_type": "server_it_room",
        "fire_alarm_detector": "HD",
        "hvac_strategy": "dedicated cooling / suppression review",
        "plumbing_allowed": False,
        "sprinkler_review": "Server/IT room needs special suppression and business-continuity review.",
    },
    "stair": {
        "keywords": ["STAIR", "STAIRCASE", "EXIT STAIR"],
        "area_class": "stair_exit",
        "semantic_type": "stair",
        "fire_alarm_detector": "SD",
        "hvac_strategy": "stair pressurization/transfer-air review",
        "plumbing_allowed": False,
        "sprinkler_review": "Stair coverage depends on local code and enclosure design.",
    },
    "storage": {
        "keywords": ["STORE", "STORAGE", "RACK", "WAREHOUSE", "LOADING", "BULK"],
        "area_class": "store_room",
        "semantic_type": "storage_or_obstructed_area",
        "fire_alarm_detector": "HD",
        "hvac_strategy": "storage ventilation review",
        "plumbing_allowed": False,
        "sprinkler_review": "Escalate to Ordinary Hazard where storage is detected.",
    },
    "lobby": {
        "keywords": ["LOBBY", "RECEPTION", "WAITING"],
        "area_class": "lobby_reception",
        "semantic_type": "standard_room",
        "fire_alarm_detector": "SD",
        "hvac_strategy": "supply diffuser + return path",
        "plumbing_allowed": False,
        "sprinkler_review": "Treat as light hazard unless occupancy or finishes require stricter review.",
    },
    "office": {
        "keywords": ["OFFICE", "MEETING", "TRAINING", "WORKSTATION", "OPEN OFFICE"],
        "area_class": "office_work_area",
        "semantic_type": "standard_room",
        "fire_alarm_detector": "SD",
        "hvac_strategy": "supply diffuser + return path",
        "plumbing_allowed": False,
        "sprinkler_review": "Treat as light hazard unless storage/combustible load is present.",
    },
}


# ---------------------------------------------------------------------------
# Accuracy target metadata
# ---------------------------------------------------------------------------

ACCURACY_TARGETS = {
    "demo_png_jpg": {
        "target_score": "7.0 to 8.0 / 10",
        "requirements": [
            "clear wall lines",
            "good resolution",
            "visible room labels",
            "visible room areas",
            "known scale or readable dimensions",
        ],
    },
    "controlled_test_plan": {
        "target_score": "8.5 to 9.2 / 10",
        "requirements": [
            "known room template",
            "visible area labels",
            "consistent CAD-style layout",
            "correct OCR/template mapping",
        ],
    },
    "layered_dxf": {
        "target_score": "9.0+ / 10",
        "requirements": [
            "clean WALL layer",
            "ROOM labels or closed room polygons",
            "DOOR layer",
            "MEP source layers",
            "known scale/unit data",
        ],
    },
}


# ---------------------------------------------------------------------------
# FireDesign-style named review gates
# These are fail-closed feasibility checks for UI/reporting.
# ---------------------------------------------------------------------------

REVIEW_GATE_DEFINITIONS = [
    {"id": "FD-001", "name": "Input file accepted", "discipline": "general"},
    {"id": "FD-002", "name": "Drawing scale identified or user supplied", "discipline": "general"},
    {"id": "FD-003", "name": "Main plan region detected", "discipline": "general"},
    {"id": "FD-004", "name": "Room/zones detected", "discipline": "general"},
    {"id": "FD-005", "name": "Room areas calculated", "discipline": "general"},
    {"id": "FD-006", "name": "Room labels/OCR reviewed", "discipline": "general"},
    {"id": "FD-007", "name": "Wet rooms classified", "discipline": "general"},
    {"id": "FD-008", "name": "Electrical/service rooms classified", "discipline": "general"},
    {"id": "FD-009", "name": "Storage hazard escalation reviewed", "discipline": "general"},
    {"id": "FD-010", "name": "Sprinkler heads placed in protected rooms", "discipline": "sprinklers"},
    {"id": "FD-011", "name": "Sprinkler max coverage per head checked", "discipline": "sprinklers"},
    {"id": "FD-012", "name": "Sprinkler spacing checked", "discipline": "sprinklers"},
    {"id": "FD-013", "name": "Sprinkler hazard profile assigned", "discipline": "sprinklers"},
    {"id": "FD-014", "name": "Riser/control valve seed provided", "discipline": "sprinklers"},
    {"id": "FD-015", "name": "Flow switch and tamper switch included", "discipline": "sprinklers"},
    {"id": "FD-016", "name": "Branch/main/drop routes generated", "discipline": "sprinklers"},
    {"id": "FD-017", "name": "Pipe diameter grouping created", "discipline": "sprinklers"},
    {"id": "FD-018", "name": "Remote-area demand seed calculated", "discipline": "sprinklers"},
    {"id": "FD-019", "name": "Obstruction/ceiling review flagged", "discipline": "sprinklers"},
    {"id": "FD-020", "name": "Smoke/heat detectors placed", "discipline": "fire_alarm"},
    {"id": "FD-021", "name": "Detector type selected by room use", "discipline": "fire_alarm"},
    {"id": "FD-022", "name": "Manual call points placed on egress routes", "discipline": "fire_alarm"},
    {"id": "FD-023", "name": "Notification devices placed", "discipline": "fire_alarm"},
    {"id": "FD-024", "name": "FACP seed provided", "discipline": "fire_alarm"},
    {"id": "FD-025", "name": "SLC/NAC route seed created", "discipline": "fire_alarm"},
    {"id": "FD-026", "name": "Fire extinguisher/signage seed created", "discipline": "fire_alarm"},
    {"id": "FD-027", "name": "AHU/source seed created", "discipline": "hvac"},
    {"id": "FD-028", "name": "Supply/return points created", "discipline": "hvac"},
    {"id": "FD-029", "name": "Wet-room exhaust review created", "discipline": "hvac"},
    {"id": "FD-030", "name": "Electrical DB seed created", "discipline": "electrical"},
    {"id": "FD-031", "name": "Lighting/socket layout seed created", "discipline": "electrical"},
    {"id": "FD-032", "name": "Wet-room electrical protection review flagged", "discipline": "electrical"},
    {"id": "FD-033", "name": "Plumbing riser seed created", "discipline": "plumbing"},
    {"id": "FD-034", "name": "Wet-area fixtures created", "discipline": "plumbing"},
    {"id": "FD-035", "name": "BOM generated", "discipline": "general"},
    {"id": "FD-036", "name": "Ceasefire product mapping generated", "discipline": "general"},
    {"id": "FD-037", "name": "Updated DXF export generated", "discipline": "general"},
    {"id": "FD-038", "name": "Engineering review warnings included", "discipline": "general"},
]


# ---------------------------------------------------------------------------
# Research / product comparison metadata
# ---------------------------------------------------------------------------

FIREDESIGN_STYLE_FEATURES = {
    "upload_and_validate": {
        "label": "Upload and validate DXF/DWG/floor plan",
        "implemented": "partial",
        "notes": "DXF and raster supported. DWG requires conversion.",
    },
    "room_detection": {
        "label": "Detect rooms, walls, plan region and labels",
        "implemented": "partial",
        "notes": "Raster CV/OCR and DXF rendering supported. True CAD polygon room extraction should be added later.",
    },
    "sprinkler_layout": {
        "label": "Generate sprinkler layout",
        "implemented": "yes",
        "notes": "Rule-based head placement and pipe-route seed implemented.",
    },
    "hydraulic_analysis": {
        "label": "Hydraulic analysis",
        "implemented": "partial",
        "notes": "Preliminary node demand only. Production requires a real hydraulic solver.",
    },
    "compliance_report": {
        "label": "Compliance / review report",
        "implemented": "partial",
        "notes": "Fail-closed review gates and warnings implemented. Needs jurisdiction-specific rules later.",
    },
    "material_list": {
        "label": "Material list / BOM",
        "implemented": "yes",
        "notes": "Standard BOM and Ceasefire-style mapped BOM supported.",
    },
    "cad_export": {
        "label": "Annotated CAD export",
        "implemented": "partial",
        "notes": "Updated DXF with CEASEFIRE layers supported. Official client blocks required for production.",
    },
    "project_history": {
        "label": "Project history / saved jobs",
        "implemented": "future",
        "notes": "Not needed for POC.",
    },
}


# ---------------------------------------------------------------------------
# Baseline warnings
# ---------------------------------------------------------------------------

BASE_WARNINGS = [
    "Engineering-assistance output only; a competent fire protection / MEP professional and the local AHJ must review it before use.",
    "DWG parsing is not available in the open-source runtime. Convert DWG to DXF or connect an approved DWG conversion service before production use.",
    "Compliance checks are fail-closed feasibility review gates, not stamped design approval.",
    "Raster PNG/JPG plans depend on OCR, wall clarity and image resolution. Use DXF with clean layers for highest accuracy.",
    "Room areas, occupancy, hazard class, ceiling height, obstructions and commodity classification must be verified before final design.",
    "Sprinkler hydraulic values are preliminary seeds only. Replace with a proper hydraulic solver before construction or authority submission.",
    "Fire alarm detector type and spacing must be checked against final ceiling, air movement, obstruction and code requirements.",
    "HVAC output is a zoning and route seed only. Replace with heat-load calculation, ventilation-rate calculation and duct sizing.",
    "Electrical output is a circuit seed only. Replace with load calculation, voltage drop, protection coordination and local code checks.",
    "Plumbing output is a fixture and route seed only. Replace with fixture schedule, pipe sizing, slopes, invert levels and shaft confirmation.",
    "Ceasefire BOM product codes and prices are dummy POC mappings unless replaced with the official product master.",
    "Updated DXF symbols are POC blocks unless replaced with the official Ceasefire CAD block library.",
]