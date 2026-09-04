"""Material database for DoorBench.

Every entry is grounded in published engineering data (densities, friction
coefficients) with a provenance note.  Effective slab densities are
calibrated against manufacturer door-weight tables (see SOURCES).

Units: SI throughout (kg, m, N, Pa).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

SOURCES = {
    "kv_door_weight": "Knape & Vogt calculated door weight table (lb/ft^2 by core & species), absupply.net/pdf/KV_Door-Weight-Table.pdf",
    "vt_door_weights": "VT Industries Technical Bulletin: Architectural wood door weights by core type",
    "sdi_hollow_metal": "Steel Door Institute (SDI) hollow metal door weights ~ 18ga 3'0x7'0 ≈ 100-110 lb; 16ga ≈ 120-135 lb",
    "glass_density": "Soda-lime glass 2,500 kg/m^3 (ASTM C1036); tempered same density",
    "steel_density": "Carbon steel 7,850 kg/m^3; stainless 304 7,900-8,000 kg/m^3",
    "aluminum_density": "Aluminum 6063-T5 storefront extrusions 2,700 kg/m^3",
    "wood_species": "USDA Forest Products Lab Wood Handbook, specific gravity at 12% MC",
    "friction_coeffs": "Engineering ToolBox / Marks' Handbook static & sliding friction coefficients",
    "ada_404": "2010 ADA Standards §404.2.9: interior hinged door opening force ≤ 5 lbf (22.2 N); hardware operable force ≤ 5 lbf",
    "ibc_1010": "IBC §1010.1.3 door opening force: 30 lbf (133 N) to set in motion, 15 lbf (67 N) to swing to full open; §1010.1.10 panic hardware unlatching force ≤ 15 lbf (67 N)",
    "en1154": "EN 1154:1996 Table 1 controlled door closing devices, power sizes 1-7",
    "bhma_a156_2": "ANSI/BHMA A156.2 bored locks: latch bolt throw ≥ 1/2 in (12.7 mm); Grade 1 deadlatch 3/4 in (19 mm)",
    "bhma_a156_5": "ANSI/BHMA A156.5 auxiliary locks: deadbolt throw ≥ 1 in (25.4 mm)",
    "ul305": "UL 305 panic hardware; touch bar/crossbar must span ≥ 1/2 door width",
}


@dataclass
class Material:
    id: str
    name: str
    family: str                 # wood | metal | glass | composite | plastic | paper | fabric | stone | mesh
    density: float              # kg/m^3 of the solid material (or effective density for a slab construction)
    is_slab_effective: bool     # True if density is an *effective* through-thickness density for a slab build-up
    friction_static: float      # mu_s against a generic steel/wood contact (used for sliding doors, gaskets)
    friction_kinetic: float
    youngs_modulus: float       # Pa (for damage model / compliance)
    yield_or_rupture_stress: float  # Pa (bending/impact damage threshold)
    dent_force_N: float         # force at fist-sized contact that leaves visible damage
    puncture_force_N: float     # force to breach/puncture a 25 mm probe
    transparent: bool = False
    base_color: tuple = (0.6, 0.6, 0.6, 1.0)
    roughness: float = 0.6
    metallic: float = 0.0
    texture: Optional[str] = None   # Poly Haven CC0 texture id (see textures.json)
    source: str = ""
    notes: str = ""

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Solid materials (true densities)
# ---------------------------------------------------------------------------
MATERIALS: dict[str, Material] = {}


def _add(m: Material):
    MATERIALS[m.id] = m
    return m


# Woods (specific gravity @12% MC from USDA Wood Handbook)
_add(Material("pine", "Eastern white pine", "wood", 400, False, 0.45, 0.35, 8.5e9, 60e6, 900, 2500,
              base_color=(0.82, 0.70, 0.50, 1), roughness=0.55, texture="kitchen_wood", source=SOURCES["wood_species"]))
_add(Material("douglas_fir", "Douglas fir", "wood", 510, False, 0.45, 0.35, 12e9, 85e6, 1100, 3000,
              base_color=(0.75, 0.55, 0.38, 1), roughness=0.55, texture="fine_grained_wood", source=SOURCES["wood_species"]))
_add(Material("oak_red", "Red oak", "wood", 700, False, 0.5, 0.4, 12.5e9, 99e6, 1800, 5000,
              base_color=(0.62, 0.45, 0.30, 1), roughness=0.5, texture="red_oak_veneer", source=SOURCES["wood_species"]))
_add(Material("oak_white", "White oak", "wood", 750, False, 0.5, 0.4, 12.3e9, 105e6, 1900, 5200,
              base_color=(0.70, 0.58, 0.42, 1), roughness=0.5, texture="white_oak_veneer", source=SOURCES["wood_species"]))
_add(Material("mahogany", "Mahogany", "wood", 590, False, 0.45, 0.35, 10e9, 79e6, 1500, 4200,
              base_color=(0.45, 0.24, 0.15, 1), roughness=0.45, texture="dark_wood", source=SOURCES["wood_species"]))
_add(Material("walnut", "Black walnut", "wood", 610, False, 0.45, 0.35, 11.6e9, 101e6, 1600, 4500,
              base_color=(0.36, 0.25, 0.18, 1), roughness=0.45, texture="rosewood_veneer_02", source=SOURCES["wood_species"]))
_add(Material("maple", "Hard maple", "wood", 700, False, 0.5, 0.4, 12.6e9, 109e6, 1900, 5200,
              base_color=(0.85, 0.74, 0.58, 1), roughness=0.5, texture="silver_oak_veneer_01", source=SOURCES["wood_species"]))
_add(Material("cherry", "Black cherry (lacquered)", "wood", 560, False, 0.4, 0.3, 10.3e9, 85e6, 1400, 4000,
              base_color=(0.55, 0.30, 0.20, 1), roughness=0.3, texture="lacquered_cherry_wood", source=SOURCES["wood_species"]))
_add(Material("cedar", "Western red cedar", "wood", 350, False, 0.45, 0.35, 7.7e9, 52e6, 700, 2200,
              base_color=(0.70, 0.45, 0.32, 1), roughness=0.6, texture="japanese_cedar_planks", source=SOURCES["wood_species"]))
_add(Material("hinoki", "Hinoki cypress", "wood", 410, False, 0.45, 0.35, 9e9, 60e6, 800, 2400,
              base_color=(0.88, 0.78, 0.60, 1), roughness=0.6, texture="hinoki_planks", source=SOURCES["wood_species"]))
_add(Material("teak", "Teak", "wood", 650, False, 0.4, 0.3, 10.7e9, 100e6, 1700, 4800,
              base_color=(0.60, 0.42, 0.25, 1), roughness=0.4, texture="oak_veneer_03", source=SOURCES["wood_species"]))
_add(Material("mdf", "Medium density fiberboard", "composite", 750, False, 0.5, 0.4, 3.5e9, 30e6, 1200, 3500,
              base_color=(0.80, 0.72, 0.60, 1), roughness=0.6, texture="plywood", source="ANSI A208.2 MDF 700-800 kg/m^3"))
_add(Material("particleboard", "Particleboard", "composite", 650, False, 0.5, 0.4, 2.5e9, 15e6, 900, 2800,
              base_color=(0.78, 0.70, 0.55, 1), roughness=0.7, texture="plywood", source="ANSI A208.1 M-2 640-800 kg/m^3"))
_add(Material("plywood", "Plywood (birch)", "wood", 620, False, 0.5, 0.4, 9e9, 40e6, 1300, 3800,
              base_color=(0.85, 0.75, 0.58, 1), roughness=0.6, texture="plywood", source=SOURCES["wood_species"]))
_add(Material("hardboard", "Hardboard (HDF skin)", "composite", 950, False, 0.45, 0.35, 4.5e9, 35e6, 700, 1800,
              base_color=(0.90, 0.90, 0.88, 1), roughness=0.5, texture="white_planks_clean", source="ANSI A135.4 hardboard 900-1050 kg/m^3"))
_add(Material("bamboo", "Bamboo strand", "wood", 700, False, 0.45, 0.35, 12e9, 90e6, 1700, 4700,
              base_color=(0.80, 0.68, 0.45, 1), roughness=0.5, texture="oak_veneer_05", source=SOURCES["wood_species"]))
_add(Material("reclaimed_barnwood", "Reclaimed weathered barn wood", "wood", 480, False, 0.55, 0.45, 8e9, 55e6, 800, 2400,
              base_color=(0.50, 0.42, 0.35, 1), roughness=0.85, texture="weathered_planks", source=SOURCES["wood_species"]))

# Metals
_add(Material("steel", "Carbon steel", "metal", 7850, False, 0.6, 0.45, 200e9, 250e6, 3500, 25000,
              base_color=(0.55, 0.56, 0.58, 1), roughness=0.45, metallic=1.0, texture="metal_plate_02", source=SOURCES["steel_density"]))
_add(Material("steel_painted", "Painted steel", "metal", 7850, False, 0.5, 0.4, 200e9, 250e6, 3500, 25000,
              base_color=(0.45, 0.48, 0.52, 1), roughness=0.4, metallic=0.2, texture="blue_metal_plate", source=SOURCES["steel_density"]))
_add(Material("steel_galvanized", "Galvanized steel", "metal", 7850, False, 0.55, 0.42, 200e9, 250e6, 3500, 25000,
              base_color=(0.65, 0.66, 0.68, 1), roughness=0.5, metallic=0.9, texture="metal_plate", source=SOURCES["steel_density"]))
_add(Material("steel_rusty", "Weathered rusty steel", "metal", 7800, False, 0.75, 0.6, 190e9, 200e6, 3000, 20000,
              base_color=(0.45, 0.30, 0.22, 1), roughness=0.9, metallic=0.5, texture="rusty_metal_04", source=SOURCES["steel_density"]))
_add(Material("stainless", "Stainless steel 304 brushed", "metal", 7900, False, 0.45, 0.35, 193e9, 215e6, 3800, 26000,
              base_color=(0.75, 0.76, 0.77, 1), roughness=0.35, metallic=1.0, source=SOURCES["steel_density"]))
_add(Material("aluminum", "Aluminum 6063-T5 (clear anodized)", "metal", 2700, False, 0.45, 0.35, 69e9, 145e6, 1500, 9000,
              base_color=(0.80, 0.81, 0.83, 1), roughness=0.4, metallic=1.0, source=SOURCES["aluminum_density"]))
_add(Material("aluminum_dark", "Aluminum (dark bronze anodized)", "metal", 2700, False, 0.45, 0.35, 69e9, 145e6, 1500, 9000,
              base_color=(0.22, 0.17, 0.13, 1), roughness=0.4, metallic=1.0, source=SOURCES["aluminum_density"]))
_add(Material("brass", "Brass (polished)", "metal", 8500, False, 0.35, 0.3, 100e9, 200e6, 3000, 20000,
              base_color=(0.85, 0.66, 0.30, 1), roughness=0.25, metallic=1.0, source="CDA C260 cartridge brass 8,530 kg/m^3"))
_add(Material("brass_antique", "Antique brass", "metal", 8500, False, 0.35, 0.3, 100e9, 200e6, 3000, 20000,
              base_color=(0.55, 0.42, 0.22, 1), roughness=0.55, metallic=0.9, source="CDA C260"))
_add(Material("bronze", "Bronze (oil rubbed)", "metal", 8800, False, 0.35, 0.3, 105e9, 220e6, 3200, 22000,
              base_color=(0.30, 0.22, 0.15, 1), roughness=0.5, metallic=0.9, source="CDA C90500 bronze 8,720 kg/m^3"))
_add(Material("chrome", "Polished chrome (plated zinc/brass)", "metal", 7100, False, 0.3, 0.25, 90e9, 150e6, 2500, 15000,
              base_color=(0.90, 0.90, 0.92, 1), roughness=0.1, metallic=1.0, source="Zamak 3 zinc alloy 6,600-7,100 kg/m^3"))
_add(Material("nickel_satin", "Satin nickel (plated)", "metal", 7100, False, 0.35, 0.3, 90e9, 150e6, 2500, 15000,
              base_color=(0.72, 0.70, 0.66, 1), roughness=0.35, metallic=1.0, source="Zamak 3"))
_add(Material("black_matte_metal", "Matte black powder-coated steel", "metal", 7850, False, 0.5, 0.4, 200e9, 250e6, 3500, 25000,
              base_color=(0.05, 0.05, 0.05, 1), roughness=0.7, metallic=0.3, source=SOURCES["steel_density"]))
_add(Material("wrought_iron", "Wrought iron", "metal", 7700, False, 0.6, 0.45, 190e9, 200e6, 4000, 30000,
              base_color=(0.08, 0.08, 0.09, 1), roughness=0.75, metallic=0.6, source="Wrought iron 7,700 kg/m^3"))
_add(Material("cast_iron", "Cast iron", "metal", 7200, False, 0.6, 0.45, 110e9, 150e6, 3000, 20000,
              base_color=(0.15, 0.15, 0.16, 1), roughness=0.8, metallic=0.5, source="Grey cast iron 7,200 kg/m^3"))
_add(Material("lead", "Lead lining", "metal", 11340, False, 0.9, 0.7, 16e9, 12e6, 200, 1500,
              base_color=(0.35, 0.36, 0.40, 1), roughness=0.6, metallic=0.8, source="Lead 11,340 kg/m^3"))

# Glass & plastics
_add(Material("glass_clear", "Clear tempered glass", "glass", 2500, False, 0.4, 0.4, 70e9, 100e6, 1500, 1500,
              transparent=True, base_color=(0.85, 0.92, 0.95, 0.25), roughness=0.05, source=SOURCES["glass_density"],
              notes="Tempered glass shatters into small fragments; impact breach ~1.5 kN at 25 mm probe"))
_add(Material("glass_frosted", "Frosted/acid-etched glass", "glass", 2500, False, 0.45, 0.4, 70e9, 100e6, 1500, 1500,
              transparent=True, base_color=(0.90, 0.93, 0.95, 0.6), roughness=0.5, source=SOURCES["glass_density"]))
_add(Material("glass_tinted", "Bronze tinted glass", "glass", 2500, False, 0.4, 0.4, 70e9, 100e6, 1500, 1500,
              transparent=True, base_color=(0.45, 0.35, 0.28, 0.45), roughness=0.05, source=SOURCES["glass_density"]))
_add(Material("glass_wired", "Wired safety glass", "glass", 2600, False, 0.4, 0.4, 70e9, 60e6, 1200, 2500,
              transparent=True, base_color=(0.80, 0.85, 0.85, 0.35), roughness=0.1, source=SOURCES["glass_density"]))
_add(Material("glass_laminated_security", "Laminated security glass (28 mm)", "glass", 2450, False, 0.4, 0.4, 70e9, 100e6, 5000, 20000,
              transparent=True, base_color=(0.85, 0.90, 0.92, 0.3), roughness=0.05, source="Laminated glass w/ PVB interlayers"))
_add(Material("mirror", "Mirror glass (silvered)", "glass", 2500, False, 0.4, 0.4, 70e9, 100e6, 1200, 1200,
              transparent=False, base_color=(0.95, 0.95, 0.97, 1), roughness=0.0, metallic=1.0, source=SOURCES["glass_density"]))
_add(Material("acrylic", "Acrylic (PMMA) sheet", "plastic", 1180, False, 0.5, 0.45, 3.2e9, 70e6, 1200, 2500,
              transparent=True, base_color=(0.90, 0.92, 0.95, 0.3), roughness=0.1, source="PMMA 1,180 kg/m^3"))
_add(Material("polycarbonate", "Polycarbonate sheet", "plastic", 1200, False, 0.5, 0.45, 2.4e9, 65e6, 4000, 12000,
              transparent=True, base_color=(0.90, 0.92, 0.94, 0.35), roughness=0.15, source="PC 1,200 kg/m^3"))
_add(Material("pvc", "Rigid PVC", "plastic", 1400, False, 0.45, 0.4, 3e9, 50e6, 800, 2500,
              base_color=(0.92, 0.92, 0.90, 1), roughness=0.4, source="uPVC 1,380-1,450 kg/m^3"))
_add(Material("pvc_flexible", "Flexible PVC strip", "plastic", 1250, False, 0.6, 0.5, 0.01e9, 15e6, 1e9, 400,
              transparent=True, base_color=(0.85, 0.90, 0.95, 0.45), roughness=0.3, source="Plasticized PVC ~1,250 kg/m^3"))
_add(Material("fiberglass_frp", "Fiberglass (FRP) skin", "composite", 1800, False, 0.45, 0.4, 17e9, 150e6, 1500, 5000,
              base_color=(0.60, 0.35, 0.22, 1), roughness=0.5, texture="mocha_oak_veneer", source="GFRP 1,700-1,900 kg/m^3"))
_add(Material("hpl", "High-pressure laminate (phenolic)", "composite", 1400, False, 0.4, 0.35, 10e9, 100e6, 2000, 7000,
              base_color=(0.75, 0.75, 0.72, 1), roughness=0.35, source="HPL compact 1,350-1,450 kg/m^3"))
_add(Material("rubber", "EPDM rubber gasket", "plastic", 1150, False, 0.9, 0.8, 0.005e9, 10e6, 1e9, 500,
              base_color=(0.08, 0.08, 0.08, 1), roughness=0.9, source="EPDM 1,100-1,200 kg/m^3"))
_add(Material("foam_pu", "Polyurethane foam core", "plastic", 40, False, 0.5, 0.4, 0.01e9, 0.3e6, 150, 400,
              base_color=(0.9, 0.85, 0.6, 1), roughness=0.9, source="Rigid PU foam 32-48 kg/m^3"))
_add(Material("foam_eps", "Polystyrene foam core", "plastic", 24, False, 0.5, 0.4, 0.005e9, 0.2e6, 100, 300,
              base_color=(0.95, 0.95, 0.95, 1), roughness=0.9, source="EPS Type II 22-25 kg/m^3"))
_add(Material("honeycomb_kraft", "Kraft paper honeycomb core", "paper", 30, False, 0.4, 0.3, 0.05e9, 0.5e6, 120, 300,
              base_color=(0.75, 0.65, 0.45, 1), roughness=0.9, source="Kraft honeycomb 25-40 kg/m^3"))
_add(Material("mineral_core", "Mineral (gypsum/perlite) fire core", "stone", 400, False, 0.5, 0.4, 2e9, 3e6, 900, 2000,
              base_color=(0.85, 0.85, 0.80, 1), roughness=0.9, source="Mineral core fire doors ~ 380-450 kg/m^3"))
_add(Material("washi_paper", "Washi shoji paper", "paper", 800, False, 0.35, 0.3, 2e9, 20e6, 15, 30,
              transparent=True, base_color=(0.98, 0.96, 0.90, 0.75), roughness=0.9, source="Shoji paper ~ 0.1 mm, 80 g/m^2",
              notes="Tears at ~15-30 N point load; translucent"))
_add(Material("canvas", "Canvas fabric", "fabric", 600, False, 0.5, 0.4, 0.5e9, 30e6, 1e9, 300,
              base_color=(0.80, 0.75, 0.60, 1), roughness=0.95, texture="fabric_pattern_05", source="Cotton duck ~ 600 kg/m^3"))
_add(Material("leather", "Leather upholstery", "fabric", 900, False, 0.6, 0.5, 0.3e9, 20e6, 1e9, 800,
              base_color=(0.35, 0.18, 0.10, 1), roughness=0.6, texture="brown_leather", source="Leather 860-1,000 kg/m^3"))
_add(Material("concrete", "Reinforced concrete", "stone", 2400, False, 0.65, 0.55, 30e9, 4e6, 1e5, 5e5,
              base_color=(0.60, 0.60, 0.58, 1), roughness=0.9, texture="concrete_panels", source="RC 2,400 kg/m^3"))
_add(Material("chain_link", "Chain-link mesh (galvanized 9 ga)", "mesh", 57.1, False, 0.6, 0.45, 200e9, 300e6, 2000, 6000,
              base_color=(0.70, 0.71, 0.72, 1), roughness=0.5, metallic=0.9, source="9 ga wire 2 in mesh ≈ 2.4 kg/m^2"))
_add(Material("expanded_metal", "Expanded metal mesh", "mesh", 7850, False, 0.6, 0.45, 200e9, 250e6, 3000, 10000,
              base_color=(0.35, 0.36, 0.38, 1), roughness=0.6, metallic=0.8, texture="rusty_metal_grid", source="3/4 in #9 expanded steel ≈ 8 kg/m^2"))
_add(Material("insect_screen", "Aluminum insect screen mesh", "mesh", 2700, False, 0.4, 0.35, 69e9, 100e6, 20, 40,
              transparent=True, base_color=(0.2, 0.2, 0.2, 0.5), roughness=0.8, source="18x16 alu mesh ≈ 0.25 kg/m^2"))

# Paints (used as visual finish layers; density irrelevant, set to 0)
PAINT_COLORS = {
    "white": (0.93, 0.93, 0.91, 1), "off_white": (0.90, 0.88, 0.82, 1), "cream": (0.93, 0.88, 0.74, 1),
    "light_grey": (0.72, 0.73, 0.74, 1), "grey": (0.50, 0.51, 0.53, 1), "charcoal": (0.20, 0.21, 0.23, 1),
    "black": (0.04, 0.04, 0.04, 1), "navy": (0.09, 0.13, 0.28, 1), "royal_blue": (0.10, 0.25, 0.60, 1),
    "sky_blue": (0.55, 0.70, 0.85, 1), "teal": (0.10, 0.45, 0.45, 1), "forest_green": (0.10, 0.28, 0.16, 1),
    "sage": (0.55, 0.62, 0.50, 1), "olive": (0.40, 0.42, 0.22, 1), "yellow": (0.90, 0.75, 0.15, 1),
    "orange": (0.90, 0.45, 0.10, 1), "red": (0.65, 0.10, 0.10, 1), "burgundy": (0.40, 0.08, 0.12, 1),
    "brown": (0.35, 0.22, 0.14, 1), "tan": (0.72, 0.60, 0.45, 1), "beige": (0.82, 0.75, 0.62, 1),
    "hospital_green": (0.65, 0.78, 0.68, 1), "safety_red": (0.80, 0.05, 0.05, 1), "fire_red": (0.70, 0.08, 0.05, 1),
    "safety_yellow": (0.95, 0.80, 0.05, 1), "school_blue": (0.30, 0.45, 0.70, 1), "hotel_walnut": (0.30, 0.20, 0.14, 1),
    "pink": (0.90, 0.65, 0.70, 1), "lavender": (0.65, 0.58, 0.80, 1), "mint": (0.70, 0.88, 0.78, 1),
}


# ---------------------------------------------------------------------------
# Slab constructions: how a door leaf is built up (skins + core + edge frame).
# Effective area density is computed here and calibrated against catalogs.
# ---------------------------------------------------------------------------
@dataclass
class SlabConstruction:
    id: str
    name: str
    skin_material: str
    skin_thickness: float          # per skin, m (0 for monolithic)
    core_material: str
    core_fill_fraction: float      # 1.0 = solid core; <1 for stiles/rails + hollow
    stile_rail_material: str       # perimeter frame material for hollow/ core constructions
    stile_width: float             # m, perimeter frame width (0 for monolithic)
    monolithic: bool               # True: whole slab is core_material
    typical_thickness: tuple       # m options
    extra_area_density: float      # kg/m^2 for internal stiffeners, glue, edge channels
    calib_area_density_1_75in: Optional[float]  # kg/m^2 reference from catalogs @ 44 mm (None if n/a)
    source: str
    fire_rating_min: int = 0       # minutes (0 none)
    visual_material: Optional[str] = None  # override for face appearance

    def area_density(self, thickness: float) -> float:
        """kg/m^2 for a given slab thickness (physics-based build-up)."""
        if self.monolithic:
            return MATERIALS[self.core_material].density * thickness
        skins = 2 * self.skin_thickness * MATERIALS[self.skin_material].density
        core_t = max(thickness - 2 * self.skin_thickness, 0.0)
        core = core_t * MATERIALS[self.core_material].density * self.core_fill_fraction
        # perimeter stiles/rails approximated as fraction of face area (~ for 0.9x2.1 door)
        if self.stile_width > 0:
            frame_frac = min(1.0, (2 * self.stile_width * (0.9 + 2.1) - 4 * self.stile_width ** 2) / (0.9 * 2.1))
            frame = frame_frac * core_t * MATERIALS[self.stile_rail_material].density * (1 - self.core_fill_fraction)
        else:
            frame = 0.0
        return skins + core + frame + self.extra_area_density


SLABS: dict[str, SlabConstruction] = {}


def _slab(s: SlabConstruction):
    SLABS[s.id] = s
    return s


# Residential interior
_slab(SlabConstruction("hollow_core", "Hollow-core (HDF skins, kraft honeycomb)", "hardboard", 0.003, "honeycomb_kraft", 0.85,
                       "pine", 0.03, False, (0.035, 0.044), 0.6, 10.0,
                       SOURCES["kv_door_weight"] + " hollow core 1-3/8in ≈ 1.7-2.2 lb/ft^2"))
_slab(SlabConstruction("hollow_core_molded", "Molded panel hollow-core (HDF skins)", "hardboard", 0.0035, "honeycomb_kraft", 0.85,
                       "pine", 0.03, False, (0.035,), 0.8, 11.0, SOURCES["kv_door_weight"]))
_slab(SlabConstruction("solid_core_pb", "Solid-core (particleboard core, wood veneer)", "plywood", 0.003, "particleboard", 1.0,
                       "pine", 0.0, False, (0.035, 0.044), 0.5, 27.0,
                       SOURCES["vt_door_weights"] + " PC-5 core 1-3/4in ≈ 5.5 lb/ft^2"))
_slab(SlabConstruction("solid_core_scl", "Solid-core (structural composite lumber core)", "plywood", 0.003, "plywood", 1.0,
                       "pine", 0.0, False, (0.044,), 0.5, 29.0, SOURCES["vt_door_weights"]))
_slab(SlabConstruction("solid_wood_pine", "Solid stile & rail pine", "pine", 0.0, "pine", 1.0, "pine", 0.0, True,
                       (0.035, 0.044), 0.0, 17.6, SOURCES["kv_door_weight"] + " white pine 1.5 lb/ft^2 @1-3/8in"))
_slab(SlabConstruction("solid_wood_fir", "Solid stile & rail Douglas fir", "douglas_fir", 0.0, "douglas_fir", 1.0, "douglas_fir", 0.0, True,
                       (0.035, 0.044), 0.0, 22.4, SOURCES["kv_door_weight"]))
_slab(SlabConstruction("solid_wood_oak", "Solid oak", "oak_red", 0.0, "oak_red", 1.0, "oak_red", 0.0, True,
                       (0.044, 0.050), 0.0, 30.8, SOURCES["kv_door_weight"] + " oak ≈ 2.9 lb/ft^2 @1-3/8in"))
_slab(SlabConstruction("solid_wood_mahogany", "Solid mahogany", "mahogany", 0.0, "mahogany", 1.0, "mahogany", 0.0, True,
                       (0.044, 0.057), 0.0, 26.0, SOURCES["kv_door_weight"]))
_slab(SlabConstruction("solid_wood_walnut", "Solid walnut", "walnut", 0.0, "walnut", 1.0, "walnut", 0.0, True,
                       (0.044,), 0.0, 26.8, SOURCES["kv_door_weight"]))
_slab(SlabConstruction("solid_wood_maple", "Solid maple", "maple", 0.0, "maple", 1.0, "maple", 0.0, True,
                       (0.044,), 0.0, 30.8, SOURCES["kv_door_weight"]))
_slab(SlabConstruction("solid_wood_cherry", "Solid cherry (lacquered)", "cherry", 0.0, "cherry", 1.0, "cherry", 0.0, True,
                       (0.044,), 0.0, 24.6, SOURCES["kv_door_weight"]))
_slab(SlabConstruction("solid_wood_teak", "Solid teak", "teak", 0.0, "teak", 1.0, "teak", 0.0, True,
                       (0.044, 0.050), 0.0, 28.6, SOURCES["kv_door_weight"]))
_slab(SlabConstruction("mdf_solid", "Solid MDF (paint grade)", "mdf", 0.0, "mdf", 1.0, "mdf", 0.0, True,
                       (0.035, 0.044), 0.0, 33.0, "MDF 750 kg/m^3 x 44 mm"))
_slab(SlabConstruction("barn_plank", "Plank & Z-brace barn door (reclaimed)", "reclaimed_barnwood", 0.0, "reclaimed_barnwood", 1.0,
                       "reclaimed_barnwood", 0.0, True, (0.035, 0.044), 3.0, 24.0, "1x6 planks + 1x4 Z-brace"))
_slab(SlabConstruction("cedar_plank", "Cedar plank (garden gate)", "cedar", 0.0, "cedar", 1.0, "cedar", 0.0, True,
                       (0.025, 0.035), 1.5, 13.8, SOURCES["kv_door_weight"]))

# Fire-rated wood
_slab(SlabConstruction("mineral_core_20", "20-min fire door (particleboard core w/ intumescent)", "plywood", 0.003, "particleboard", 1.0,
                       "pine", 0.0, False, (0.044,), 1.0, 28.0, SOURCES["vt_door_weights"], fire_rating_min=20))
_slab(SlabConstruction("mineral_core_45", "45-min fire door (mineral core)", "plywood", 0.003, "mineral_core", 1.0,
                       "pine", 0.0, False, (0.044,), 2.0, 22.0, SOURCES["vt_door_weights"] + " mineral core ≈ 4.5-6 lb/ft^2", fire_rating_min=45))
_slab(SlabConstruction("mineral_core_90", "90-min fire door (mineral core, steel edges)", "plywood", 0.003, "mineral_core", 1.0,
                       "steel", 0.0, False, (0.044,), 5.0, 30.0, SOURCES["vt_door_weights"], fire_rating_min=90))

# Hollow metal (Steel Door Institute)
_slab(SlabConstruction("hollow_metal_20ga", "Hollow metal 20 ga (polystyrene core)", "steel_painted", 0.00091, "foam_eps", 1.0,
                       "steel", 0.0, False, (0.044,), 3.5, 18.5, SOURCES["sdi_hollow_metal"]))
_slab(SlabConstruction("hollow_metal_18ga", "Hollow metal 18 ga (honeycomb core)", "steel_painted", 0.00121, "honeycomb_kraft", 1.0,
                       "steel", 0.0, False, (0.044,), 4.0, 24.0, SOURCES["sdi_hollow_metal"], fire_rating_min=90))
_slab(SlabConstruction("hollow_metal_18ga_pu", "Hollow metal 18 ga (polyurethane core, exterior)", "steel_galvanized", 0.00121, "foam_pu", 1.0,
                       "steel", 0.0, False, (0.044,), 4.0, 25.0, SOURCES["sdi_hollow_metal"]))
_slab(SlabConstruction("hollow_metal_16ga", "Hollow metal 16 ga (steel-stiffened)", "steel_painted", 0.00152, "foam_eps", 1.0,
                       "steel", 0.0, False, (0.044,), 6.5, 31.0, SOURCES["sdi_hollow_metal"], fire_rating_min=180))
_slab(SlabConstruction("hollow_metal_14ga", "Hollow metal 14 ga (detention/security)", "steel", 0.0019, "foam_eps", 1.0,
                       "steel", 0.0, False, (0.044, 0.050), 12.0, 44.0, "Detention hollow metal 14 ga ≈ 9 lb/ft^2"))
_slab(SlabConstruction("stainless_hollow", "Stainless steel hollow metal (18 ga, PU core)", "stainless", 0.00121, "foam_pu", 1.0,
                       "stainless", 0.0, False, (0.044,), 4.0, 25.0, SOURCES["sdi_hollow_metal"]))
_slab(SlabConstruction("steel_plate_security", "Solid steel plate (security/safe room)", "steel", 0.0, "steel", 1.0, "steel", 0.0, True,
                       (0.006, 0.010, 0.012), 8.0, None, "Steel plate 7,850 kg/m^3"))
_slab(SlabConstruction("vault_composite", "Vault door (steel plates + concrete fill)", "steel", 0.012, "concrete", 1.0, "steel", 0.0, False,
                       (0.10, 0.15, 0.25), 30.0, None, "UL 608 class vault doors 500-2500 kg"))
_slab(SlabConstruction("blast_steel", "Blast door (steel plates + stiffeners)", "steel", 0.010, "steel", 0.35, "steel", 0.0, False,
                       (0.08, 0.12), 40.0, None, "Blast doors 300-1500 kg typical"))
_slab(SlabConstruction("lead_lined", "Lead-lined wood door (radiology)", "plywood", 0.003, "particleboard", 1.0, "pine", 0.0, False,
                       (0.044,), 18.2, 45.0, "1/16 in lead sheet ≈ 3.7 lb/ft^2 added"))

# Glass & storefront
_slab(SlabConstruction("glass_frameless_10", "Frameless tempered glass 10 mm", "glass_clear", 0.0, "glass_clear", 1.0, "glass_clear", 0.0, True,
                       (0.010,), 0.0, None, SOURCES["glass_density"]))
_slab(SlabConstruction("glass_frameless_12", "Frameless tempered glass 12 mm", "glass_clear", 0.0, "glass_clear", 1.0, "glass_clear", 0.0, True,
                       (0.012,), 0.0, None, SOURCES["glass_density"]))
_slab(SlabConstruction("glass_frameless_19", "Frameless tempered glass 19 mm (heavy)", "glass_clear", 0.0, "glass_clear", 1.0, "glass_clear", 0.0, True,
                       (0.019,), 0.0, None, SOURCES["glass_density"]))
_slab(SlabConstruction("storefront_alu", "Aluminum storefront (medium stile) w/ 6 mm glass", "aluminum", 0.0, "glass_clear", 1.0, "aluminum", 0.0, True,
                       (0.006,), 12.0, None, "Kawneer 350 medium stile; 6 mm glass 15 kg/m^2 + frame ~12 kg/m^2"))
_slab(SlabConstruction("storefront_alu_igu", "Aluminum storefront (wide stile) w/ 25 mm IGU", "aluminum", 0.0, "glass_clear", 1.0, "aluminum", 0.0, True,
                       (0.012,), 14.0, None, "Wide stile w/ 1 in insulating glass unit"))
_slab(SlabConstruction("patio_slider_glass", "Sliding patio door (vinyl frame, 19 mm IGU)", "pvc", 0.0, "glass_clear", 1.0, "pvc", 0.0, True,
                       (0.012,), 6.0, None, "Vinyl sliding patio panels ≈ 25-35 kg per 3ft panel"))
_slab(SlabConstruction("mirror_bypass", "Mirrored bypass closet door (6 mm mirror, steel frame)", "mirror", 0.0, "mirror", 1.0, "steel", 0.0, True,
                       (0.006,), 3.0, None, "6 mm mirror 15 kg/m^2 + steel frame"))

# Composite / exterior
_slab(SlabConstruction("fiberglass_entry", "Fiberglass entry door (PU foam core)", "fiberglass_frp", 0.002, "foam_pu", 1.0,
                       "pine", 0.10, False, (0.044,), 2.5, 13.5, "Therma-Tru / Masonite fiberglass 3070 ≈ 55-65 lb"))
_slab(SlabConstruction("steel_entry_24ga", "Steel entry door 24 ga (PU foam core, residential)", "steel_painted", 0.0006, "foam_pu", 1.0,
                       "pine", 0.10, False, (0.044,), 2.0, 14.0, "Residential steel entry 3070 ≈ 55-70 lb"))
_slab(SlabConstruction("upvc_panel", "uPVC panel door (foam core)", "pvc", 0.003, "foam_pu", 1.0, "steel", 0.05, False,
                       (0.044, 0.070), 4.0, 16.0, "uPVC composite doors 30-40 kg"))
_slab(SlabConstruction("hpl_partition", "HPL toilet partition door (13 mm compact)", "hpl", 0.0, "hpl", 1.0, "hpl", 0.0, True,
                       (0.013,), 0.0, None, "Compact laminate 13 mm ≈ 18 kg/m^2"))
_slab(SlabConstruction("phenolic_partition", "Powder-coated steel partition door (honeycomb)", "steel_painted", 0.0006, "honeycomb_kraft", 1.0,
                       "steel", 0.0, False, (0.025,), 1.5, None, "Steel toilet partitions ≈ 12 kg/m^2"))
_slab(SlabConstruction("cold_storage_100", "Cold storage door (100 mm PU core, stainless skins)", "stainless", 0.0008, "foam_pu", 1.0,
                       "stainless", 0.0, False, (0.100, 0.120), 6.0, None, "Walk-in cooler doors ≈ 20-25 kg/m^2"))
_slab(SlabConstruction("freezer_150", "Freezer door (150 mm PU core, heated gasket)", "stainless", 0.0008, "foam_pu", 1.0,
                       "stainless", 0.0, False, (0.150,), 8.0, None, "Freezer doors ≈ 28 kg/m^2"))
_slab(SlabConstruction("screen_alu", "Aluminum screen door (insect mesh)", "aluminum", 0.0, "insect_screen", 1.0, "aluminum", 0.0, True,
                       (0.025,), 1.6, None, "Aluminum screen door ≈ 4-6 kg total"))
_slab(SlabConstruction("screen_wood", "Wood-frame screen door", "pine", 0.0, "insect_screen", 1.0, "pine", 0.0, True,
                       (0.028,), 4.0, None, "Wood screen door ≈ 8-10 kg"))
_slab(SlabConstruction("storm_alu_glass", "Aluminum storm door (full glass in 25 mm frame)", "glass_clear", 0.0015, "insect_screen", 0.0, "aluminum", 0.04, False,
                       (0.025,), 5.0, None, "Storm door ≈ 15-20 kg", visual_material="glass_clear"))
_slab(SlabConstruction("shoji", "Shoji (hinoki lattice + washi paper)", "hinoki", 0.0, "washi_paper", 1.0, "hinoki", 0.0, True,
                       (0.0001,), 3.2, None, "Shoji screen 0.9x1.8 m ≈ 5-7 kg"))
_slab(SlabConstruction("fusuma", "Fusuma (paper on wood lattice, opaque)", "hinoki", 0.0, "washi_paper", 1.0, "hinoki", 0.0, True,
                       (0.0002,), 4.0, None, "Fusuma ≈ 6-9 kg"))
_slab(SlabConstruction("louver_wood", "Full louver pine door", "pine", 0.0, "pine", 0.55, "pine", 0.0, False,
                       (0.035,), 0.0, 10.5, SOURCES["kv_door_weight"]))
_slab(SlabConstruction("garage_steel_single", "Garage sectional 25 ga steel (non-insulated)", "steel_painted", 0.0005, "foam_eps", 0.0,
                       "steel", 0.0, False, (0.045,), 3.0, None, "Clopay/Amarr single-layer ≈ 7 kg/m^2"))
_slab(SlabConstruction("garage_steel_insulated", "Garage sectional (steel/PU/steel 2 in)", "steel_painted", 0.0005, "foam_pu", 1.0,
                       "steel", 0.0, False, (0.050,), 3.5, None, "Insulated 3-layer ≈ 12-14 kg/m^2"))
_slab(SlabConstruction("garage_wood_carriage", "Wood carriage-house garage door (cedar overlay)", "cedar", 0.0, "cedar", 1.0, "cedar", 0.0, True,
                       (0.045,), 6.0, None, "Wood garage doors ≈ 20-25 kg/m^2"))
_slab(SlabConstruction("rollup_steel", "Roll-up door 22 ga corrugated slats", "steel_galvanized", 0.0, "steel_galvanized", 1.0, "steel", 0.0, True,
                       (0.0008,), 4.0, None, "Coiling doors ≈ 10 kg/m^2"))
_slab(SlabConstruction("rollup_alu_grille", "Roll-up aluminum grille (security)", "aluminum", 0.0, "aluminum", 0.35, "aluminum", 0.0, False,
                       (0.010,), 2.0, None, "Rolling grilles ≈ 8 kg/m^2"))
_slab(SlabConstruction("chain_link_gate", "Chain-link gate (1-5/8 in frame)", "chain_link", 0.0, "chain_link", 1.0, "steel_galvanized", 0.0, True,
                       (0.042,), 6.5, None, "Frame ≈ 4 kg/m + 9 ga mesh 2.4 kg/m^2 (leaf depth = the 1-5/8 in frame pipe; the fabric's 2.4 kg/m^2 is kept as an areal-equivalent density 57 kg/m^3 x 42 mm)"))
_slab(SlabConstruction("wrought_iron_gate", "Wrought iron gate", "wrought_iron", 0.0, "wrought_iron", 0.12, "wrought_iron", 0.0, False,
                       (0.025,), 5.0, None, "Ornamental iron gates ≈ 25-35 kg/m^2"))
_slab(SlabConstruction("steel_bar_grille", "Steel bar door (detention/cell)", "steel", 0.0, "steel", 0.18, "steel", 0.0, False,
                       (0.025,), 8.0, None, "Cell door w/ 3/4 in bars @ 5 in ≈ 40-60 kg/m^2"))
_slab(SlabConstruction("tube_gate", "Tube gate (1-3/4 in 16 ga steel tube frame + 5 rails)", "steel_galvanized", 0.0, "steel_galvanized", 0.03, "steel_galvanized", 0.0, False,
                       (0.045,), 2.5, None, "16 ft tube gate ≈ 70-90 kg (~7 kg/m^2)"))
_slab(SlabConstruction("expanded_metal_gate", "Expanded metal security gate", "expanded_metal", 0.0, "expanded_metal", 0.5, "steel", 0.0, False,
                       (0.006,), 4.0, None, "Expanded metal 8 kg/m^2 + frame"))
_slab(SlabConstruction("pet_flap_pvc", "Pet door flap (flexible PVC)", "pvc_flexible", 0.0, "pvc_flexible", 1.0, "pvc", 0.0, True,
                       (0.004, 0.006), 0.0, None, "PetSafe flap 4-6 mm PVC"))
_slab(SlabConstruction("pet_flap_acrylic", "Pet door flap (rigid acrylic)", "acrylic", 0.0, "acrylic", 1.0, "pvc", 0.0, True,
                       (0.005,), 0.0, None, "Rigid acrylic flap"))
_slab(SlabConstruction("baby_gate_steel", "Baby gate (steel tube)", "steel_painted", 0.0, "steel_painted", 0.08, "steel", 0.0, False,
                       (0.020,), 0.5, None, "Pressure-mounted gate ≈ 5-8 kg"))
_slab(SlabConstruction("strip_curtain", "PVC strip curtain (200x2 mm strips)", "pvc_flexible", 0.0, "pvc_flexible", 1.0, "pvc", 0.0, True,
                       (0.002,), 0.0, None, "PVC strip 200 mm x 2 mm ≈ 0.5 kg/m"))
_slab(SlabConstruction("hospital_solid", "Hospital door (solid core, HPL faces, stainless kick)", "hpl", 0.001, "particleboard", 1.0,
                       "pine", 0.0, False, (0.044,), 3.0, 30.0, SOURCES["vt_door_weights"]))
_slab(SlabConstruction("acoustic_wood", "Acoustic (STC 45) wood door (dense core)", "plywood", 0.003, "mdf", 1.0, "pine", 0.0, False,
                       (0.057,), 4.0, 45.0, "Acoustic doors 2-1/4 in ≈ 9-10 lb/ft^2"))
_slab(SlabConstruction("elevator_landing", "Elevator landing door (16 ga steel, stiffened)", "stainless", 0.0015, "foam_eps", 0.0,
                       "steel", 0.0, False, (0.030,), 6.0, None, "Elevator hoistway doors ≈ 30 kg/m^2"))
_slab(SlabConstruction("ship_watertight", "Ship watertight door (8 mm steel, stiffened)", "steel_painted", 0.0, "steel_painted", 1.0, "steel", 0.0, True,
                       (0.008,), 25.0, None, "Quick-acting WT doors ≈ 90-120 kg/m^2"))
_slab(SlabConstruction("submarine_hatch", "Submarine/marine hatch (dished steel)", "steel_painted", 0.0, "steel_painted", 1.0, "steel", 0.0, True,
                       (0.012,), 20.0, None, "Marine hatch covers ≈ 110-130 kg/m^2"))
_slab(SlabConstruction("attic_hatch", "Attic hatch (plywood on pine frame)", "plywood", 0.0, "plywood", 1.0, "pine", 0.0, True,
                       (0.012,), 2.0, None, "Attic access panel ≈ 8-10 kg"))
_slab(SlabConstruction("cellar_trapdoor", "Cellar trapdoor (oak planks)", "oak_white", 0.0, "oak_white", 1.0, "oak_white", 0.0, True,
                       (0.032,), 3.0, None, "Oak trapdoor"))
_slab(SlabConstruction("revolving_wing", "Revolving door wing (aluminum frame + 10 mm glass)", "aluminum", 0.0, "glass_clear", 1.0, "aluminum", 0.0, True,
                       (0.010,), 9.0, None, "Boon Edam wings ≈ 45-60 kg"))
_slab(SlabConstruction("turnstile_arm", "Turnstile arm (stainless tube)", "stainless", 0.0, "stainless", 1.0, "stainless", 0.0, True,
                       (0.038,), 0.0, None, "Tripod turnstile arms 38 mm dia stainless tube"))
_slab(SlabConstruction("cardboard", "Corrugated cardboard (stage prop / play door)", "honeycomb_kraft", 0.0, "honeycomb_kraft", 1.0, "honeycomb_kraft", 0.0, True,
                       (0.010,), 0.6, None, "Double-wall corrugated ≈ 0.9 kg/m^2"))
_slab(SlabConstruction("canvas_tent", "Canvas tent door (rigid pole frame)", "canvas", 0.0, "canvas", 1.0, "aluminum", 0.0, True,
                       (0.001,), 1.0, None, "Canvas + aluminum pole frame"))
_slab(SlabConstruction("leather_padded", "Leather-padded studio door (solid core)", "leather", 0.003, "particleboard", 1.0, "pine", 0.0, False,
                       (0.050,), 3.0, None, "Upholstered acoustic door"))
_slab(SlabConstruction("bamboo_solid", "Solid bamboo strand door", "bamboo", 0.0, "bamboo", 1.0, "bamboo", 0.0, True,
                       (0.040,), 0.0, None, "Strand bamboo 700 kg/m^3"))
_slab(SlabConstruction("polycarbonate_panel", "Polycarbonate panel door (greenhouse)", "polycarbonate", 0.0, "polycarbonate", 1.0, "aluminum", 0.0, True,
                       (0.008,), 3.0, None, "Twin-wall PC ≈ 1.5 kg/m^2 + alu frame"))


def slab_face_material(slab: SlabConstruction) -> Material:
    if slab.visual_material:
        return MATERIALS[slab.visual_material]
    return MATERIALS[slab.skin_material if not slab.monolithic else slab.core_material]


# Rolling / sliding friction coefficients for tracks & rollers
ROLLER_FRICTION = {
    # (mu_rolling effective, notes)
    "ball_bearing_nylon": (0.012, "Sealed ball-bearing nylon rollers on aluminum track (new)"),
    "ball_bearing_steel": (0.015, "Steel ball-bearing rollers on steel track"),
    "plain_nylon": (0.03, "Plain-bore nylon rollers (closet bypass)"),
    "plain_steel_worn": (0.06, "Worn steel rollers, dusty track"),
    "dirty_track": (0.12, "Debris-filled track, flat spots on rollers"),
    "barn_hanger": (0.02, "Flat-track barn door hangers, steel wheels"),
    "wood_on_wood": (0.30, "Shoji/fusuma wood runner in wood groove (kinetic mu 0.25-0.4)"),
    "wood_on_wood_waxed": (0.12, "Waxed wood runner"),
    "glide_teflon": (0.05, "PTFE glide pads"),
    "bottom_rolling_heavy": (0.02, "Bottom-rolling heavy leaf on steel rail (cantilever gate)"),
    "cantilever_gate": (0.025, "Cantilever gate carriages (4 sealed bearing trolleys)"),
    "elevator_hanger": (0.01, "Elevator hanger rollers"),
    "garage_nylon": (0.02, "Garage nylon rollers in steel track"),
    "garage_steel_dry": (0.05, "Garage steel rollers, dry track"),
    "rollup_curtain": (0.08, "Roll-up curtain in guides (slat friction)"),
    "accordion_glides": (0.04, "Accordion door top glides"),
    "bifold_pivot_guide": (0.03, "Bifold pivot + top guide"),
}

# Hinge pin bearing friction coefficients (steel pin in knuckle)
HINGE_BEARING_MU = {
    "ball_bearing": (0.04, "Ball-bearing butt hinge (Hager BB1279 class)"),
    "plain_bearing_new": (0.12, "Plain bearing butt hinge, lubricated"),
    "plain_bearing_worn": (0.25, "Plain bearing, dry, worn"),
    "rusty": (0.55, "Corroded pin, seized/creaky"),
    "bronze_bushing": (0.08, "Oil-impregnated bronze bushing"),
    "nylon_bushing": (0.10, "Nylon bushing (residential)"),
    "pivot_thrust": (0.05, "Floor pivot w/ thrust bearing"),
    "continuous_geared": (0.07, "Continuous geared aluminum hinge"),
    "strap_pintle": (0.30, "Strap hinge on pintle (gate)"),
    "spring_hinge": (0.15, "Spring hinge internal friction"),
    "piano": (0.15, "Continuous piano hinge"),
    "lift_off": (0.14, "Lift-off (flag) hinge"),
    "concealed_soss": (0.10, "Concealed SOSS hinge multi-link"),
    "rising_butt": (0.15, "Rising butt hinge (helical)"),
    "cam_lift": (0.12, "Cam-lift hinge (cold storage)"),
    "double_action_spring": (0.18, "Double-acting spring hinge (saloon)"),
    "gravity_pivot": (0.06, "Gravity pivot (toilet partition, self-closing)"),
    "pet_flap_pin": (0.20, "Plastic flap pin in bushing"),
    "rotor_bearing": (0.02, "Revolving door/turnstile rotor bearing"),
}
