import re
from typing import Any


class Parser:
    """A collection of parsing methods for extracting structured data from text."""

    @staticmethod
    def parse_direction(text: str) -> tuple[float, str] | None:
        """
        Extracts a signed bearing angle from *text*.

        Examples:
        -------
        • “… at a 79-degree angle.”
        • “… at an angle of -24 degrees.”
        • “… at 79 degrees.”
        • “… at a 79-degree direction.”
        • “… at 79° direction.”
        • “… bearing angle is -18 ° …”

        Returns
        -------
        tuple[float, str] | None
            Tuple of (angle, unit) where angle is in degrees (-180 <= θ <= 180)
            and unit is the matched unit string. Returns None if not found.
        """

        # Number (integer/decimal, ± sign option)
        num = r"([-+]?\d+(?:\.\d+)?)"

        # All variations of "79°", "79 degrees", "79-degree"
        unit = r"((?:°|deg(?:ree)?s?))"

        pattern = re.compile(
            rf"{num}\s*-?\s*{unit}",  # 79°  / 79 degrees / 79-degree
            flags=re.I,
        )

        m = pattern.search(text)
        if not m:
            return None

        angle = float(m.group(1))
        unit_str = m.group(2)

        # Just in case, check the range (specification: [-180, 180])
        if angle < -180 or angle > 180:
            # Wrap 0-360° to [-180,180]
            angle = ((angle + 180) % 360) - 180
        return angle, unit_str

    @staticmethod
    def parse_clock_position(text: str) -> tuple[int, str] | None:
        """
        Parses a clock direction (1-12) from a string.
        Args:
            text (str): The input string.
        Returns:
            tuple[int, str] | None: Tuple of (hour, unit) where unit is the matched clock phrase. Returns None if not found.
        """
        # Apostrophe variants between o and clock: ' ' ` ´ or nothing.
        apos = "['\u2019`´]?"  # optional ASCII (') or Unicode (') apostrophe
        patterns = [
            # 3 o'clock / 3 o clock / 3 oclock
            rf"\b(\d{{1,2}})\s*o{apos}\s*clock",
            # clock hour 3 / clock position 3
            r"\bclock\s*(?:hour|position)?\s*(\d{1,2})",
            # 3 o'clock position
            rf"\b(\d{{1,2}})\s*o{apos}\s*clock\s*position",
            # Region [3] -> treat 3 as 3 o'clock when within 1-12
            r"\bregion\s*\[\s*(\d{1,2})\s*\]",
        ]

        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                val = int(match.group(1))
                if 1 <= val <= 12:
                    unit_str = match.group(0).lstrip(match.group(1)).strip()
                    return val, unit_str
        return None

    @staticmethod
    def parse_distance(text: str) -> tuple[float, str] | None:
        """
        Parses a distance from a string and converts it to meters.
        Supports inches, feet, cm, and meters.
        Args:
            text (str): The input string.
        Returns:
            tuple[float, str] | None: Tuple of (distance_meters, unit) or None if not found.
        """
        # Conversion factors (canonical unit -> meters)
        conversions = {
            "inch": 0.0254,
            "foot": 0.3048,
            "cm": 0.01,
            "m": 1.0,
            "km": 1000.0,
        }

        # Synonym list for each canonical unit
        synonyms = {
            "inch": ["inch", "inches", "in"],
            "foot": ["foot", "feet", "ft"],
            "cm": ["cm", "centimeters", "centimetres"],
            "m": ["m", "meter", "meters", "metre", "metres"],
            "km": ["km", "kilometer", "kilometers", "kilometre", "kilometres"],
        }

        for canonical, terms in synonyms.items():
            # Build a pattern ensuring we don't accidentally capture speed units like "m/s".
            # The negative lookahead `(?![\s*/])` makes sure the unit is not immediately
            # followed by a slash (e.g., the "m" in "m/s") or another letter that would
            # form a larger token (e.g., "mph" inside "kmph").
            unit_pattern = r"|".join([re.escape(t) for t in terms])

            # Exclude cases like "10 m/s" where the unit is immediately followed by a slash.
            match = re.search(
                rf"(\d+\.?\d*)\s*({unit_pattern})\b(?!/)",
                text,
                re.IGNORECASE,
            )
            if match:
                value = float(match.group(1))
                unit_str = match.group(2)
                return value * conversions[canonical], unit_str

        return None

    @staticmethod
    def parse_velocity(text: str) -> tuple[float, str] | None:
        """
        Parses a speed from a string and converts it to meters per second.
        Supports m/s, km/h, and mph.
        Args:
            text (str): The input string.
        Returns:
            tuple[float, str] | None: Tuple of (speed_m_s, unit) or None if not found.
        """
        # Canonical unit -> meters/second factor
        conversions = {
            "ms": 1.0,
            "kmh": 1 / 3.6,
            "mph": 1609.34 / 3600,
        }

        synonyms = {
            "ms": [
                "m/s",
                "m s-1",
                r"m s\^1",
                "m s⁻1",
                "mps",
                "meter per second",
                "metre per second",
                "meters per second",
                "metres per second",
            ],
            "kmh": [
                "km/h",
                "km h-1",
                "kmh",
                "kph",
                "kilometer per hour",
                "kilometre per hour",
                "kilometers per hour",
                "kilometres per hour",
            ],
            "mph": [
                "mph",
                "mile per hour",
                "miles per hour",
            ],
        }

        for canonical, terms in synonyms.items():
            for term in terms:
                term_regex = re.escape(term).replace("\\ ", "\\s+")
                match = re.search(
                    rf"(\d+\.?\d*)\s*\b({term_regex})\b",
                    text,
                    re.IGNORECASE,
                )
                if match:
                    value = float(match.group(1))
                    unit_str = match.group(2)
                    return value * conversions[canonical], unit_str

        return None


def parse_pred_value(qa_type: str, pred_text: str) -> Any | None:
    if qa_type == "direction":
        return Parser.parse_direction(pred_text)
    elif qa_type == "distance":
        return Parser.parse_distance(pred_text)
    elif qa_type == "velocity":
        return Parser.parse_velocity(pred_text)
    else:
        raise ValueError(f"Unsupported qa_type: {qa_type}")
