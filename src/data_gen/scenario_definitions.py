"""Scenario definitions, consumer archetypes, and seasonal profiles for BillGuard."""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class ConsumerArchetype:
    name: str
    code: str
    base_bimonthly_units_mean: float
    base_bimonthly_units_std: float
    has_ac: bool
    summer_multiplier: float
    monsoon_multiplier: float
    description: str


CONSUMER_ARCHETYPES: Dict[str, ConsumerArchetype] = {
    "LIFELINE_LOW": ConsumerArchetype(
        name="Lifeline / Basic Domestic",
        code="LIFELINE",
        base_bimonthly_units_mean=85.0,
        base_bimonthly_units_std=15.0,
        has_ac=False,
        summer_multiplier=1.15,
        monsoon_multiplier=0.90,
        description="1-2 room home, basic lighting, fans, small TV. Eligible for 100-unit full subsidy."
    ),
    "MODERATE": ConsumerArchetype(
        name="Moderate Middle Class",
        code="MODERATE",
        base_bimonthly_units_mean=260.0,
        base_bimonthly_units_std=40.0,
        has_ac=False,
        summer_multiplier=1.30,
        monsoon_multiplier=0.85,
        description="Fans, refrigerator, TV, washing machine, grinder, mixer. Typically in 200-400 slab."
    ),
    "HIGH_AC": ConsumerArchetype(
        name="Urban High Consumption (With AC)",
        code="HIGH_AC",
        base_bimonthly_units_mean=550.0,
        base_bimonthly_units_std=75.0,
        has_ac=True,
        summer_multiplier=1.75,
        monsoon_multiplier=0.75,
        description="1-2 Inverter ACs, refrigerator, microwave, water heaters. Sharp summer spike (May-June)."
    ),
    "LUXURY_VILLA": ConsumerArchetype(
        name="Luxury Domestic / Villa",
        code="LUXURY_VILLA",
        base_bimonthly_units_mean=1350.0,
        base_bimonthly_units_std=180.0,
        has_ac=True,
        summer_multiplier=1.85,
        monsoon_multiplier=0.80,
        description="Multiple centralized/split ACs, water pumping, EV charging, smart appliances."
    ),
}

# Bi-monthly cycle seasonal weights in Tamil Nadu (6 billing cycles per year)
# Cycle 1: Jan-Feb (Mild winter / Margazhi-Thai)
# Cycle 2: Mar-Apr (Early Summer / Panguni-Chithirai)
# Cycle 3: May-Jun (Peak Summer / Agni Nakshatram / Vaikasi-Aani)
# Cycle 4: Jul-Aug (Monsoon onset / Aadi-Avani)
# Cycle 5: Sep-Oct (Post-monsoon / Purattasi-Aippasi)
# Cycle 6: Nov-Dec (Northeast Monsoon & cool season / Karthigai-Margazhi)
BIMONTHLY_SEASONAL_FACTORS: Dict[int, float] = {
    1: 0.88,  # Jan-Feb
    2: 1.25,  # Mar-Apr
    3: 1.65,  # May-Jun (Peak Summer)
    4: 1.10,  # Jul-Aug
    5: 0.95,  # Sep-Oct
    6: 0.82,  # Nov-Dec (Coolest / Heavy Rain)
}

ANOMALY_SCENARIOS = [
    "NORMAL",
    "ARITHMETIC_OVERCHARGE",
    "METER_FAULT_SURGE",
    "DELAYED_READING_SLAB_JUMP",
    "METER_ROLLOVER_UNHANDLED",
    "DEAD_METER_ZERO_USAGE",
    "PHANTOM_ARREARS",
]
