import json

import plotly.graph_objects as go

STATE_ALIASES = {
    "Arunanchal Pradesh": "Arunachal Pradesh",
    "Jammu & Kashmir": "Jammu & Kashmir (incl. Ladakh)",
}


def get_display_name(raw_name: str) -> str:
    return STATE_ALIASES.get(raw_name, raw_name)


def get_india_borders_trace():
    with open("dashboard/assets/states.geojson", "r", encoding="utf-8") as f:
        data = json.load(f)

    lons = []
    lats = []
    texts = []

    for feature in data["features"]:
        geom = feature["geometry"]
        state_name = get_display_name(feature["properties"].get("ST_NM", "Unknown"))
        if geom["type"] == "Polygon":
            for ring in geom["coordinates"]:
                for coord in ring:
                    lons.append(coord[0])
                    lats.append(coord[1])
                    texts.append(state_name)
                lons.append(None)
                lats.append(None)
                texts.append(None)
        elif geom["type"] == "MultiPolygon":
            for poly in geom["coordinates"]:
                for ring in poly:
                    for coord in ring:
                        lons.append(coord[0])
                        lats.append(coord[1])
                        texts.append(state_name)
                    lons.append(None)
                    lats.append(None)
                    texts.append(None)

    return go.Scatter(
        x=lons,
        y=lats,
        text=texts,
        mode="lines",
        line={"color": "white", "width": 1},
        hoverinfo="text",
        showlegend=False,
    )


def get_cities_trace():
    cities = {
        "Delhi": (28.70, 77.10),
        "Mumbai": (19.07, 72.87),
        "Kolkata": (22.57, 88.36),
        "Chennai": (13.08, 80.27),
        "Bengaluru": (12.97, 77.59),
        "Hyderabad": (17.38, 78.48),
        "Ahmedabad": (23.02, 72.57),
    }
    return go.Scatter(
        x=[c[1] for c in cities.values()],
        y=[c[0] for c in cities.values()],
        mode="markers+text",
        marker={"size": 6, "color": "white", "line": {"color": "black", "width": 1}},
        text=list(cities.keys()),
        textposition="top center",
        textfont={"color": "white", "size": 10},
        hoverinfo="text",
        showlegend=False,
    )
