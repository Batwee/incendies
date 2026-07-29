import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import datetime
import os
import time
from streamlit_autorefresh import st_autorefresh

# ==========================================
# CONFIGURATION
# ==========================================
st.set_page_config(page_title="Suivi Satellite des Incendies (France & Espagne)", page_icon="🔥", layout="wide")

REGION_BBOX = {"lat_min": 35.0, "lat_max": 51.5, "lon_min": -10.0, "lon_max": 10.0}

CACHE_DIR = "cache_data"
os.makedirs(CACHE_DIR, exist_ok=True)
FIRES_JSON_PATH = os.path.join(CACHE_DIR, "fires_data_fr_es.json")

def is_cache_valid(filepath, max_age_seconds):
    if os.path.exists(filepath):
        return (time.time() - os.path.getmtime(filepath)) < max_age_seconds
    return False

def get_recency_info(dt, now):
    hours_ago = (now - dt).total_seconds() / 3600.0

    if hours_ago <= 12:
        return "#8B0000", "< 12h", "Moins de 12 heures"
    elif hours_ago <= 24:
        return "#FF0000", "< 24h", "12h à 24h"
    elif hours_ago <= 48:
        return "#FF7F00", "< 48h", "24h à 48h"
    elif hours_ago <= 72:
        return "#FFB84D", "< 72h", "48h à 72h"
    else:
        return "#FFE082", "> 72h", "Plus de 72h"

# ==========================================
# RÉCUPÉRATION DES DONNÉES (NASA FIRMS)
# ==========================================
@st.cache_data(ttl=1800)
def fetch_all_fires_data():
    if is_cache_valid(FIRES_JSON_PATH, 1800):
        try:
            df = pd.read_json(FIRES_JSON_PATH)
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['datetime_str'])
                return df
        except Exception:
            pass

    urls = [
        "https://firms.modaps.eosdis.nasa.gov/data/active_fire/suomi-npp-viirs-c2/csv/SUOMI_NPP_VIIRS_C2_Europe_7d.csv",
        "https://firms.modaps.eosdis.nasa.gov/data/active_fire/noaa-20-viirs-c2/csv/J1_VIIRS_C2_Europe_7d.csv",
        "https://firms.modaps.eosdis.nasa.gov/data/active_fire/noaa-21-viirs-c2/csv/J2_VIIRS_C2_Europe_7d.csv",
        "https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/MODIS_C6_1_Europe_7d.csv"
    ]
    
    dfs = []
    for url in urls:
        try:
            df_raw = pd.read_csv(url)
            df_sub = df_raw[
                (df_raw['latitude'] >= REGION_BBOX["lat_min"]) & 
                (df_raw['latitude'] <= REGION_BBOX["lat_max"]) &
                (df_raw['longitude'] >= REGION_BBOX["lon_min"]) & 
                (df_raw['longitude'] <= REGION_BBOX["lon_max"])
            ].copy()
            if not df_sub.empty:
                dfs.append(df_sub)
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    data = pd.concat(dfs, ignore_index=True)
    data['acq_time'] = data['acq_time'].astype(str).str.zfill(4)
    data['datetime_str'] = data['acq_date'] + ' ' + data['acq_time'].str[:2] + ':' + data['acq_time'].str[2:] + ':00'
    data['datetime'] = pd.to_datetime(data['datetime_str'])

    columns_to_keep = ['latitude', 'longitude', 'datetime_str', 'confidence']
    data_clean = data[columns_to_keep].drop_duplicates(subset=['latitude', 'longitude', 'datetime_str']).copy()

    if not data_clean.empty:
        try:
            data_clean.to_json(FIRES_JSON_PATH, orient="records")
        except Exception:
            pass

    data_clean['datetime'] = pd.to_datetime(data_clean['datetime_str'])
    return data_clean

# ==========================================
# APPLICATION PRINCIPALE
# ==========================================
def main():
    st.title("🔥 Détection des Foyers par Satellite (France & Espagne)")
    
    st_autorefresh(interval=5 * 60 * 1000, key="datarefresh")
    
    df_fires = fetch_all_fires_data()
    now_utc = datetime.datetime.utcnow()

    col1, col2 = st.columns(2)
    col1.metric("🔥 Total foyers détectés (7 jours)", len(df_fires) if not df_fires.empty else 0)
    col2.metric("⏱️ Dernière actualisation", datetime.datetime.now().strftime("%H:%M:%S"))

    # LÉGENDE DE RÉCENCE
    st.markdown("""
    <div style="background-color: #262730; padding: 12px; border-radius: 8px; margin-bottom: 12px; border: 1px solid #444; color: #FFFFFF;">
        <span style="font-weight: bold; margin-right: 15px; color: #FFFFFF;">🎨 Récence des détections :</span>
        <span style="margin-right: 15px; font-size: 14px; color: #FFFFFF;"><span style="color:#8B0000;">🟤</span> <b style="color: #FFFFFF;">&lt; 12h</b></span>
        <span style="margin-right: 15px; font-size: 14px; color: #FFFFFF;"><span style="color:#FF0000;">🔴</span> <b style="color: #FFFFFF;">12h à 24h</b></span>
        <span style="margin-right: 15px; font-size: 14px; color: #FFFFFF;"><span style="color:#FF7F00;">🟠</span> <b style="color: #FFFFFF;">24h à 48h</b></span>
        <span style="margin-right: 15px; font-size: 14px; color: #FFFFFF;"><span style="color:#FFB84D;">🟡</span> <b style="color: #FFFFFF;">48h à 72h</b></span>
        <span style="margin-right: 15px; font-size: 14px; color: #FFFFFF;"><span style="color:#FFE082;">⚪</span> <b style="color: #FFFFFF;">&gt; 72h (Zone jaune diffuse)</b></span>
    </div>
    """, unsafe_allow_html=True)

    m = folium.Map(location=[43.0, 1.5], zoom_start=5, prefer_canvas=True, tiles=None)

    # Fonds de carte
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", attr="Esri", name="🛰️ Satellite HD", overlay=False).add_to(m)
    folium.TileLayer("https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", attr="OpenTopoMap", name="🌲 Relief & Végétation", overlay=False).add_to(m)
    folium.TileLayer("OpenStreetMap", name="🗺️ Carte Routière", overlay=False).add_to(m)

    if not df_fires.empty:
        # 1. FILTRE HEATMAP (Léger et optimisé)
        heat_group = folium.FeatureGroup(name="🔥 Heatmap (Densité)", show=True)
        HeatMap(
            df_fires[['latitude', 'longitude']].values.tolist(),
            radius=10,
            blur=8,
            min_opacity=0.35,
            gradient={0.2: '#FFE082', 0.5: '#FF7F00', 0.8: '#FF0000', 1.0: '#8B0000'}
        ).add_to(heat_group)
        heat_group.add_to(m)

        # 2. FILTRE RÉCENCE (Sécurité RAM : limitation stricte à 3000 points max pour éviter le plantage navigateur)
        recency_group = folium.FeatureGroup(name="🎨 Récence (Points & zones par âge)", show=False)

        hours_ago_series = (now_utc - df_fires['datetime']).dt.total_seconds() / 3600.0
        df_recent = df_fires[hours_ago_series <= 72]
        df_old = df_fires[hours_ago_series > 72]

        # Voile jaune pour les anciens foyers
        if not df_old.empty:
            HeatMap(
                df_old[['latitude', 'longitude']].values.tolist(),
                radius=14,
                blur=10,
                min_opacity=0.25,
                gradient={0.4: '#FFE082', 1.0: '#FFD54F'}
            ).add_to(recency_group)

        # Protection anti-saturation : si trop de points récents, on garde les plus récents en priorité (max 3000)
        if len(df_recent) > 3000:
            df_recent = df_recent.sort_values(by='datetime', ascending=False).head(3000)

        for _, row in df_recent.iterrows():
            color, label_age, age_desc = get_recency_info(row['datetime'], now_utc)
            
            popup_html = f"""
            <div style="font-family: Arial; font-size: 12px; min-width: 150px;">
                <b style="color:{color};">🔥 Foyer ({label_age})</b><br><br>
                <b>Ancienneté :</b> {age_desc}<br>
                <b>Date UTC :</b> {row['datetime_str']}<br>
                <b>Coordonnées :</b> {row['latitude']:.4f}, {row['longitude']:.4f}
            </div>
            """
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=4,
                color="#000000",
                weight=0.5,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                popup=folium.Popup(popup_html, max_width=220),
                tooltip=f"Feu {label_age}"
            ).add_to(recency_group)

        recency_group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    st_folium(m, width="100%", height=720, key="main_fire_map")

if __name__ == "__main__":
    main()
