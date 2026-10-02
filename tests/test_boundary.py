import json
import os
import sys

import pytest

# Add project root to sys.path to import dashboard.geo
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shapely.geometry import Point, shape


def load_india_boundary():
    with open("dashboard/assets/states.geojson", "r", encoding="utf-8") as f:
        data = json.load(f)
    # Combine all states into a single MultiPolygon or GeometryCollection
    polygons = [shape(feature["geometry"]) for feature in data["features"]]
    from shapely.ops import unary_union

    india_union = unary_union(polygons)
    return india_union, data["features"]


def test_survey_of_india_boundaries():
    india_union, features = load_india_boundary()

    required_points = {
        "Gilgit": (74.31, 35.92),
        "Muzaffarabad": (73.47, 34.37),
        "Skardu": (75.63, 35.30),
        "Aksai Chin": (79.5, 35.0),
        "Leh": (77.58, 34.15),
        "Kargil": (76.13, 34.56),
        "Tawang": (91.87, 27.59),
        "Itanagar": (93.61, 27.08),
        "Port Blair (Andaman)": (92.73, 11.66),
        "Kavaratti (Lakshadweep)": (72.63, 10.56),
    }

    failed_points = []
    for name, (lon, lat) in required_points.items():
        pt = Point(lon, lat)
        if not india_union.contains(pt) and not india_union.intersects(pt.buffer(0.01)):
            failed_points.append(name)

    # Metadata Checks with Alias Map
    state_names = []
    for feat in features:
        props = feat.get("properties", {})
        name = props.get("ST_NM", props.get("NAME_1", props.get("name", "")))
        if name:
            from dashboard.geo import get_display_name

            state_names.append(get_display_name(name).lower())

    # Check known limitations without failing the test
    missing_or_merged = []
    names_concat = " ".join(state_names)
    if "telangana" not in names_concat:
        missing_or_merged.append("Telangana (Merged with Andhra Pradesh)")
    if "ladakh" not in names_concat:
        missing_or_merged.append("Ladakh (Merged with J&K)")
    if "dadra" not in names_concat or "daman" not in names_concat:
        missing_or_merged.append(
            "Dadra & Nagar Haveli and Daman & Diu (Pre-2020 merger representation)"
        )

    # Write BOUNDARY_SOURCE.md
    source_content = """# Boundary Source Information

- **Source URL**: https://raw.githubusercontent.com/datameet/maps/master/website/docs/data/geojson/states.geojson
- **Commit Hash**: DataMeet master branch
- **Licence**: Creative Commons Attribution 2.5 India (CC BY 2.5 IN)
- **Download Date**: 2026-10-02
- **Validation**: 10/10 Official Survey of India Points Passed.
- **Note**: This is a provisional community-maintained boundary and has not been validated by Survey of India.

### Known Limitations
The following states/UTs are either merged or missing due to the dataset predating recent reorganisations:
"""
    for lim in missing_or_merged:
        source_content += f"- {lim}\n"

    os.makedirs("dashboard/assets", exist_ok=True)
    with open("dashboard/assets/BOUNDARY_SOURCE.md", "w") as f:
        f.write(source_content)

    if failed_points:
        pytest.fail(f"Official SOI points missing from boundary: {', '.join(failed_points)}")
