# Simple i18n dictionary for the Streamlit dashboard

TRANSLATIONS = {
    "en": {
        "heavy_rain_alert": "Heavy Rain Alert",
        "heatwave_alert": "Heatwave Alert",
        "high_wind_alert": "High Wind Alert",
        "level_red": "RED: Take Action",
        "level_orange": "ORANGE: Be Prepared",
        "level_yellow": "YELLOW: Be Updated",
        "level_green": "GREEN: No Warning",
    },
    "hi": {
        "heavy_rain_alert": "भारी बारिश की चेतावनी (Heavy Rain)",
        "heatwave_alert": "हीटवेव की चेतावनी (Heatwave)",
        "high_wind_alert": "तेज हवा की चेतावनी (High Wind)",
        "level_red": "लाल (RED): कार्रवाई करें",
        "level_orange": "नारंगी (ORANGE): तैयार रहें",
        "level_yellow": "पीला (YELLOW): अपडेट रहें",
        "level_green": "हरा (GREEN): कोई चेतावनी नहीं",
    },
}


def get_text(lang: str, key: str) -> str:
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)
