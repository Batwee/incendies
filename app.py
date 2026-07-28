# -*- coding: utf-8 -*-
"""
=============================================================================
 FireWatch France - Surveillance des incendies en temps (quasi) réel
=============================================================================
Application Streamlit + Leaflet (Folium) affichant les incendies actifs
en France en combinant plusieurs sources de données publiques et gratuites.

SOURCES DE DONNÉES UTILISÉES (et pourquoi)
-----------------------------------------------------------------------------
1. NASA FIRMS (Fire Information for Resource Management System)
   -> Détections actives par satellite VIIRS (SNPP, NOAA-20, NOAA-21,
      résolution ~375 m) et MODIS (Terra/Aqua, résolution ~1 km).
   -> Données "Near Real-Time" (latence ~3h), c'est la source la PLUS
      fraîche et la plus fiable disponible gratuitement au monde pour la
      détection de foyers actifs par satellite.
   -> Nécessite une clé gratuite (MAP_KEY) : https://firms.modaps.eosdis.nasa.gov/api/map_key/
   -> Endpoint : https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{SOURCE}/{BBOX}/{DAYS}

2. EFFIS / GWIS (Copernicus - European Forest Fire Information System)
   -> Source secondaire européenne (JRC / Copernicus Emergency Management).
   -> Utilisée en complément pour recouper / enrichir les détections FIRMS
      (elle peut détecter des feux non couverts par un passage satellite
      VIIRS/MODIS récent). Traitée en "best effort" : si le service est
      indisponible ou change de format, l'application continue de
      fonctionner avec FIRMS seul.

3. Open-Meteo (https://open-meteo.com) - 100% gratuit, sans clé
   -> Données météo (vent, humidité, température, précipitations) utilisées
      pour (a) enrichir chaque foyer avec le contexte météo local et
      (b) calculer l'indice de risque de propagation de la couche
      "Zones forestières menacées".
   -> Open-Meteo Elevation API -> relief / altitude (facteur de pente).

4. Overpass API (OpenStreetMap, overpass-api.de) - gratuit, sans clé
   -> Polygones de forêts / bois (natural=wood, landuse=forest) utilisés
      pour localiser la végétation combustible sur le territoire.

FUSION DES DONNÉES
-----------------------------------------------------------------------------
Les détections FIRMS (plusieurs capteurs) et EFFIS sont fusionnées dans un
schéma commun, puis regroupées en "foyers" (clusters spatiaux) afin
d'estimer une superficie et une tendance d'évolution (nouveau / en hausse /
stable / en baisse) pour chaque foyer, au lieu d'afficher des milliers de
points bruts individuels.

RÉSILIENCE
-----------------------------------------------------------------------------
Chaque appel réseau est encapsulé dans un try/except avec timeout. Si une
source est indisponible, elle est simplement ignorée (et signalée dans la
barre latérale) sans jamais faire planter l'application.

Lancement : streamlit run app.py
=============================================================================
"""

from __future__ import annotations

import io
import math
import time
import datetime as dt
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
import folium
from folium.plugins import HeatMap, MarkerCluster, Fullscreen, MiniMap
from streamlit_folium import st_folium

try:
    from streamlit_autorefresh import st_autorefresh
    _AUTOREFRESH_OK = True
except ImportError:
    _AUTOREFRESH_OK = False


# =============================================================================
# 1. CONFIGURATION GÉNÉRALE
# =============================================================================

st.set_page_config(
    page_title="FireWatch France",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Emprise approximative de la France métropolitaine + Corse (west, south, east, north)
FRANCE_BBOX = (-5.2, 41.2, 9.7, 51.3)
FRANCE_CENTER = (46.6, 2.3)
DEFAULT_ZOOM = 6

# Résolution du maillage utilisé pour la couche de risque (en degrés).
# Un pas plus petit = plus précis mais beaucoup plus d'appels API.
THREAT_GRID_STEP = 1.0

# Capteurs FIRMS interrogés (tous gratuits, complémentaires en couverture temporelle)
FIRMS_SOURCES = ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "MODIS_NRT"]

# Taille de pixel approximative par capteur -> sert à estimer une superficie (km²)
PIXEL_AREA_KM2 = {
    "VIIRS_SNPP_NRT": 0.14,
    "VIIRS_NOAA20_NRT": 0.14,
    "VIIRS_NOAA21_NRT": 0.14,
    "MODIS_NRT": 1.0,
    "EFFIS": 0.09,  # Sentinel-2 / MODIS hotspot EFFIS, ordre de grandeur
}

REQUEST_TIMEOUT = 20  # secondes

CONFIDENCE_MAP_VIIRS = {"l": 25, "n": 60, "h": 90, "low": 25, "nominal": 60, "high": 90}


# =============================================================================
# 2. ÉTAT DE SANTÉ DES SOURCES (pour affichage transparent à l'utilisateur)
# =============================================================================

def _status_init():
    if "api_status" not in st.session_state:
        st.session_state["api_status"] = {}


def report_status(name: str, ok: bool, detail: str = ""):
    """Enregistre l'état d'une source de données pour affichage sidebar."""
    _status_init()
    st.session_state["api_status"][name] = {"ok": ok, "detail": detail, "ts": dt.datetime.now()}


def safe_get(url: str, params: Optional[dict] = None, timeout: int = REQUEST_TIMEOUT):
    """Wrapper GET tolérant aux pannes : ne lève jamais d'exception vers l'appelant."""
    try:
        resp = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": "FireWatchFrance/1.0"})
        resp.raise_for_status()
        return resp
    except requests.exceptions.RequestException as exc:
        return None


# =============================================================================
# 3. SOURCE 1 : NASA FIRMS (foyers actifs, priorité temps réel)
# =============================================================================

@st.cache_data(ttl=300, show_spinner=False)
def fetch_firms(map_key: str, day_range: int) -> pd.DataFrame:
    """
    Récupère les détections actives FIRMS pour la France sur les N derniers
    jours, en interrogeant successivement chaque capteur disponible et en
    fusionnant les résultats. day_range est borné à 10 (limite de l'API).
    """
    if not map_key:
        report_status("NASA FIRMS", False, "Clé MAP_KEY manquante")
        return pd.DataFrame()

    day_range = max(1, min(int(day_range), 10))
    bbox_str = ",".join(str(v) for v in FRANCE_BBOX)
    frames = []
    errors = []

    for source in FIRMS_SOURCES:
        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/{source}/{bbox_str}/{day_range}"
        resp = safe_get(url)
        if resp is None:
            errors.append(source)
            continue
        try:
            df = pd.read_csv(io.StringIO(resp.text))
        except Exception:
            errors.append(source)
            continue
        if df.empty or "latitude" not in df.columns:
            # Réponse vide ou message d'erreur textuel renvoyé par l'API (ex: quota, clé invalide)
            continue
        df["source"] = source
        frames.append(df)

    if not frames:
        report_status("NASA FIRMS", False, f"Aucun capteur disponible ({', '.join(errors) or 'inconnu'})")
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True, sort=False)
    detail = f"{len(merged)} détections ({len(frames)}/{len(FIRMS_SOURCES)} capteurs OK)"
    report_status("NASA FIRMS", True, detail)
    return merged


def normalize_firms(df: pd.DataFrame) -> pd.DataFrame:
    """Met les données FIRMS (VIIRS + MODIS, colonnes légèrement différentes) au format commun."""
    if df.empty:
        return df

    out = pd.DataFrame()
    out["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    out["acq_date"] = df.get("acq_date")
    # acq_time est au format HHMM (ex: 1345) -> on le remet en HH:MM
    raw_time = df.get("acq_time", pd.Series(["0000"] * len(df))).astype(str).str.zfill(4)
    out["acq_time"] = raw_time.str[:2] + ":" + raw_time.str[2:]
    out["acq_datetime"] = pd.to_datetime(
        out["acq_date"].astype(str) + " " + out["acq_time"], errors="coerce", utc=True
    )

    # Confiance : VIIRS = l/n/h (texte) ; MODIS = 0-100 (numérique)
    conf_raw = df.get("confidence", pd.Series([None] * len(df)))

    def to_conf_pct(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip().lower()
        if s in CONFIDENCE_MAP_VIIRS:
            return CONFIDENCE_MAP_VIIRS[s]
        try:
            return float(s)
        except ValueError:
            return np.nan

    out["confidence_pct"] = conf_raw.apply(to_conf_pct)
    out["confidence_label"] = conf_raw.astype(str)

    # FRP (Fire Radiative Power, en MW) = meilleur proxy gratuit d'intensité
    out["frp"] = pd.to_numeric(df.get("frp", np.nan), errors="coerce")
    out["brightness"] = pd.to_numeric(df.get("bright_ti4", df.get("brightness", np.nan)), errors="coerce")
    out["daynight"] = df.get("daynight", "")
    out["source"] = df.get("source", "FIRMS")
    out["satellite"] = df.get("satellite", df.get("source", ""))
    out["provider"] = "NASA FIRMS"
    return out


# =============================================================================
# 4. SOURCE 2 : EFFIS / Copernicus (secondaire, best-effort)
# =============================================================================

@st.cache_data(ttl=600, show_spinner=False)
def fetch_effis() -> pd.DataFrame:
    """
    Interroge le service WFS EFFIS (Copernicus) pour les points chauds actifs
    récents sur l'emprise France. Cette source change parfois de format côté
    serveur : toute erreur est absorbée et la source est simplement désactivée
    pour ce cycle de rafraîchissement, sans impacter le reste de l'app.
    """
    bbox_str = f"{FRANCE_BBOX[0]},{FRANCE_BBOX[1]},{FRANCE_BBOX[2]},{FRANCE_BBOX[3]}"
    url = "https://maps.effis.emergency.copernicus.eu/effis"
    params = {
        "service": "WFS",
        "version": "1.1.0",
        "request": "GetFeature",
        "typeName": "ms:modis.hs",
        "outputFormat": "application/json",
        "srsName": "EPSG:4326",
        "bbox": f"{bbox_str},EPSG:4326",
    }
    resp = safe_get(url, params=params)
    if resp is None:
        report_status("EFFIS (Copernicus)", False, "Service indisponible")
        return pd.DataFrame()
    try:
        geojson = resp.json()
        feats = geojson.get("features", [])
        rows = []
        for feat in feats:
            geom = feat.get("geometry") or {}
            coords = geom.get("coordinates")
            props = feat.get("properties", {})
            if not coords or geom.get("type") != "Point":
                continue
            rows.append(
                {
                    "latitude": coords[1],
                    "longitude": coords[0],
                    "acq_date": props.get("date") or props.get("DATE") or props.get("acq_date"),
                    "province": props.get("province") or props.get("PROVINCE"),
                    "raw_props": props,
                }
            )
        if not rows:
            report_status("EFFIS (Copernicus)", False, "Aucune donnée exploitable (0 foyer ou format inattendu)")
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        report_status("EFFIS (Copernicus)", True, f"{len(df)} points chauds")
        return df
    except Exception as exc:
        report_status("EFFIS (Copernicus)", False, f"Format de réponse inattendu ({exc.__class__.__name__})")
        return pd.DataFrame()


def normalize_effis(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = pd.DataFrame()
    out["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    out["acq_date"] = df["acq_date"]
    out["acq_time"] = "--:--"
    out["acq_datetime"] = pd.to_datetime(df["acq_date"], errors="coerce", utc=True)
    out["confidence_pct"] = 70.0  # EFFIS ne fournit pas de score de confiance comparable -> valeur par défaut
    out["confidence_label"] = "EFFIS"
    out["frp"] = np.nan
    out["brightness"] = np.nan
    out["daynight"] = ""
    out["source"] = "EFFIS"
    out["satellite"] = "EFFIS/Copernicus"
    out["provider"] = "EFFIS (Copernicus)"
    return out


# =============================================================================
# 5. FUSION DES SOURCES DE FEUX + CLUSTERING EN "FOYERS"
# =============================================================================

@st.cache_data(ttl=300, show_spinner=False)
def fetch_all_fires(map_key: str, day_range: int) -> pd.DataFrame:
    """Fusionne FIRMS + EFFIS dans un schéma commun unique."""
    firms_raw = fetch_firms(map_key, day_range)
    effis_raw = fetch_effis()

    parts = []
    firms_n = normalize_firms(firms_raw)
    if not firms_n.empty:
        parts.append(firms_n)
    effis_n = normalize_effis(effis_raw)
    if not effis_n.empty:
        parts.append(effis_n)

    if not parts:
        return pd.DataFrame()

    merged = pd.concat(parts, ignore_index=True, sort=False)
    merged = merged.dropna(subset=["latitude", "longitude"])
    # On ne garde que les points réellement dans l'emprise France
    w, s, e, n = FRANCE_BBOX
    merged = merged[
        (merged["longitude"].between(w, e)) & (merged["latitude"].between(s, n))
    ].reset_index(drop=True)

    def intensity_label(frp):
        if pd.isna(frp):
            return "Inconnue"
        if frp < 5:
            return "Faible"
        if frp < 15:
            return "Modérée"
        if frp < 50:
            return "Forte"
        return "Extrême"

    merged["intensity_label"] = merged["frp"].apply(intensity_label)
    return merged


def cluster_fires(df: pd.DataFrame, grid_deg: float = 0.02) -> pd.DataFrame:
    """
    Regroupe les détections individuelles en "foyers" (clusters spatiaux) afin
    d'estimer une superficie et une tendance d'évolution par foyer plutôt que
    d'afficher chaque pixel satellite séparément.
    """
    if df.empty:
        return df

    work = df.copy()
    work["bin_lat"] = (work["latitude"] / grid_deg).round().astype(int)
    work["bin_lon"] = (work["longitude"] / grid_deg).round().astype(int)
    work["cluster_id"] = work["bin_lat"].astype(str) + "_" + work["bin_lon"].astype(str)

    now = pd.Timestamp.now(tz="UTC")
    recent_cut = now - pd.Timedelta(hours=24)
    prev_cut = now - pd.Timedelta(hours=48)

    records = []
    for cid, grp in work.groupby("cluster_id"):
        n_det = len(grp)
        lat_c = grp["latitude"].mean()
        lon_c = grp["longitude"].mean()
        frp_sum = grp["frp"].sum(skipna=True)
        frp_max = grp["frp"].max(skipna=True)
        conf_mean = grp["confidence_pct"].mean(skipna=True)
        first_seen = grp["acq_datetime"].min()
        last_seen = grp["acq_datetime"].max()
        sources = sorted(set(grp["source"].dropna()))
        area_km2 = sum(PIXEL_AREA_KM2.get(s, 0.2) for s in grp["source"])

        n_recent = (grp["acq_datetime"] >= recent_cut).sum()
        n_prev = ((grp["acq_datetime"] >= prev_cut) & (grp["acq_datetime"] < recent_cut)).sum()
        if n_prev == 0 and n_recent > 0:
            trend = "🆕 Nouveau"
        elif n_recent > n_prev * 1.2:
            trend = "📈 En hausse"
        elif n_recent < n_prev * 0.8:
            trend = "📉 En baisse"
        else:
            trend = "➡️ Stable"

        best_intensity = grp.loc[grp["frp"].idxmax()]["intensity_label"] if grp["frp"].notna().any() else "Inconnue"

        records.append(
            {
                "cluster_id": cid,
                "latitude": lat_c,
                "longitude": lon_c,
                "n_detections": n_det,
                "frp_sum": frp_sum,
                "frp_max": frp_max,
                "confidence_pct": conf_mean,
                "first_seen": first_seen,
                "last_seen": last_seen,
                "sources": ", ".join(sources),
                "area_km2_est": round(area_km2, 2),
                "trend": trend,
                "intensity_label": best_intensity,
            }
        )

    return pd.DataFrame(records)


# =============================================================================
# 6. SOURCE 3 : OPEN-METEO (météo + relief) — pour enrichissement & risque
# =============================================================================

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_weather_grid(lats: tuple, lons: tuple) -> pd.DataFrame:
    """Récupère la météo courante (vent, humidité, T°, précipitations) pour une liste de points."""
    if not lats:
        return pd.DataFrame()
    lat_str = ",".join(f"{v:.3f}" for v in lats)
    lon_str = ",".join(f"{v:.3f}" for v in lons)
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat_str,
        "longitude": lon_str,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,precipitation",
        "daily": "precipitation_sum",
        "forecast_days": 1,
        "timezone": "auto",
    }
    resp = safe_get(url, params=params)
    if resp is None:
        report_status("Open-Meteo (météo)", False, "Service indisponible")
        return pd.DataFrame()
    try:
        data = resp.json()
        # Avec plusieurs points, l'API renvoie une liste d'objets ; avec un seul point, un objet unique.
        entries = data if isinstance(data, list) else [data]
        rows = []
        for i, entry in enumerate(entries):
            cur = entry.get("current", {})
            daily = entry.get("daily", {})
            precip_today = 0.0
            try:
                precip_today = float(daily.get("precipitation_sum", [0])[0])
            except (IndexError, TypeError, ValueError):
                pass
            rows.append(
                {
                    "latitude": entry.get("latitude", lats[i] if i < len(lats) else None),
                    "longitude": entry.get("longitude", lons[i] if i < len(lons) else None),
                    "temperature_c": cur.get("temperature_2m"),
                    "humidity_pct": cur.get("relative_humidity_2m"),
                    "wind_speed_kmh": cur.get("wind_speed_10m"),
                    "wind_dir_deg": cur.get("wind_direction_10m"),
                    "precip_today_mm": precip_today,
                }
            )
        report_status("Open-Meteo (météo)", True, f"{len(rows)} points météo")
        return pd.DataFrame(rows)
    except Exception as exc:
        report_status("Open-Meteo (météo)", False, f"Erreur de parsing ({exc.__class__.__name__})")
        return pd.DataFrame()


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_elevation(lats: tuple, lons: tuple) -> pd.DataFrame:
    """Relief / altitude (Open-Meteo Elevation API), utilisé comme facteur de propagation."""
    if not lats:
        return pd.DataFrame()
    url = "https://api.open-meteo.com/v1/elevation"
    lat_str = ",".join(f"{v:.3f}" for v in lats)
    lon_str = ",".join(f"{v:.3f}" for v in lons)
    resp = safe_get(url, params={"latitude": lat_str, "longitude": lon_str})
    if resp is None:
        report_status("Open-Meteo (relief)", False, "Service indisponible")
        return pd.DataFrame()
    try:
        elevations = resp.json().get("elevation", [])
        df = pd.DataFrame({"latitude": lats, "longitude": lons, "elevation_m": elevations})
        report_status("Open-Meteo (relief)", True, f"{len(df)} points")
        return df
    except Exception as exc:
        report_status("Open-Meteo (relief)", False, f"Erreur de parsing ({exc.__class__.__name__})")
        return pd.DataFrame()


# =============================================================================
# 7. SOURCE 4 : OVERPASS API (OpenStreetMap) — végétation / forêts
# =============================================================================

@st.cache_data(ttl=21600, show_spinner=False)
def fetch_forest_points() -> pd.DataFrame:
    """
    Interroge Overpass (OpenStreetMap) pour obtenir le centre approximatif de
    chaque massif forestier de plus de ~0.3 km² sur l'emprise France. Le
    filtre de surface minimale limite le volume de données renvoyées afin de
    respecter les capacités du service gratuit Overpass.
    """
    w, s, e, n = FRANCE_BBOX
    query = f"""
    [out:json][timeout:50];
    (
      way["natural"="wood"]({s},{w},{n},{e});
      way["landuse"="forest"]({s},{w},{n},{e});
      relation["natural"="wood"]({s},{w},{n},{e});
      relation["landuse"="forest"]({s},{w},{n},{e});
    );
    out center 4000;
    """
    resp = None
    for endpoint in ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]:
        try:
            r = requests.post(endpoint, data={"data": query}, timeout=50)
            r.raise_for_status()
            resp = r
            break
        except requests.exceptions.RequestException:
            continue

    if resp is None:
        report_status("OpenStreetMap (forêts)", False, "Service Overpass indisponible")
        return pd.DataFrame()
    try:
        elements = resp.json().get("elements", [])
        rows = []
        for el in elements:
            center = el.get("center")
            if not center:
                continue
            rows.append({"latitude": center["lat"], "longitude": center["lon"]})
        report_status("OpenStreetMap (forêts)", True, f"{len(rows)} massifs forestiers")
        return pd.DataFrame(rows)
    except Exception as exc:
        report_status("OpenStreetMap (forêts)", False, f"Erreur de parsing ({exc.__class__.__name__})")
        return pd.DataFrame()


# =============================================================================
# 8. CALCUL DE LA COUCHE "ZONES FORESTIÈRES MENACÉES"
# =============================================================================

def build_threat_grid_points(step: float = THREAT_GRID_STEP) -> pd.DataFrame:
    """Génère les centroïdes de la grille de risque sur l'emprise France."""
    w, s, e, n = FRANCE_BBOX
    lats = np.arange(s, n, step)
    lons = np.arange(w, e, step)
    grid = [(lat + step / 2, lon + step / 2) for lat in lats for lon in lons]
    return pd.DataFrame(grid, columns=["latitude", "longitude"])


def haversine_km(lat1, lon1, lat2, lon2):
    """Distance orthodromique (km) — utilisée pour la proximité aux foyers actifs."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def compute_threat_grid(fires_clustered: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule un indice de risque de propagation (0-100) par cellule de la
    grille, en combinant :
      - présence de végétation forestière (OSM)
      - vent (Open-Meteo) : un vent fort augmente le risque de propagation
      - humidité relative (Open-Meteo) : un air sec augmente le risque
      - température (Open-Meteo)
      - sécheresse récente : absence de précipitations dans la journée
      - relief : zones en altitude / terrain accidenté = propagation facilitée
      - proximité à un foyer actif détecté (FIRMS/EFFIS)

    NOTE : il s'agit d'un indice heuristique de sensibilisation, PAS d'une
    prévision officielle de type Météo-France / IFN. Il vise à mettre en
    évidence les zones forestières à surveiller, pas à remplacer les
    autorités compétentes.
    """
    grid = build_threat_grid_points()
    forest = fetch_forest_points()

    if forest.empty:
        grid["forest_density"] = 0.0
    else:
        # Densité de massifs forestiers par cellule (comptage simple, normalisé)
        step = THREAT_GRID_STEP
        forest = forest.copy()
        forest["bin_lat"] = (forest["latitude"] // step) * step
        forest["bin_lon"] = (forest["longitude"] // step) * step
        counts = forest.groupby(["bin_lat", "bin_lon"]).size().rename("count").reset_index()
        grid["bin_lat"] = (grid["latitude"] // step) * step
        grid["bin_lon"] = (grid["longitude"] // step) * step
        grid = grid.merge(counts, on=["bin_lat", "bin_lon"], how="left")
        grid["count"] = grid["count"].fillna(0)
        max_count = grid["count"].max() or 1
        grid["forest_density"] = (grid["count"] / max_count).clip(0, 1)
        grid = grid.drop(columns=["bin_lat", "bin_lon", "count"])

    # On ne calcule la météo/relief que pour les cellules réellement boisées
    # (économie d'appels API) — seuil bas pour ne pas exclure les petites forêts.
    active_cells = grid[grid["forest_density"] > 0.02].reset_index(drop=True)
    if active_cells.empty:
        grid["risk_score"] = 0.0
        return grid

    lats = tuple(active_cells["latitude"].round(3))
    lons = tuple(active_cells["longitude"].round(3))
    weather = fetch_weather_grid(lats, lons)
    elevation = fetch_elevation(lats, lons)

    active_cells = active_cells.reset_index(drop=True)
    if not weather.empty and len(weather) == len(active_cells):
        active_cells = pd.concat([active_cells, weather[["temperature_c", "humidity_pct", "wind_speed_kmh", "precip_today_mm"]]], axis=1)
    else:
        active_cells["temperature_c"] = np.nan
        active_cells["humidity_pct"] = np.nan
        active_cells["wind_speed_kmh"] = np.nan
        active_cells["precip_today_mm"] = np.nan

    if not elevation.empty and len(elevation) == len(active_cells):
        active_cells["elevation_m"] = elevation["elevation_m"].values
    else:
        active_cells["elevation_m"] = np.nan

    # Proximité au foyer actif le plus proche (0 = très proche -> risque de propagation immédiat)
    def nearest_fire_km(lat, lon):
        if fires_clustered is None or fires_clustered.empty:
            return np.nan
        dists = fires_clustered.apply(
            lambda r: haversine_km(lat, lon, r["latitude"], r["longitude"]), axis=1
        )
        return dists.min()

    if fires_clustered is not None and not fires_clustered.empty:
        active_cells["dist_fire_km"] = active_cells.apply(
            lambda r: nearest_fire_km(r["latitude"], r["longitude"]), axis=1
        )
    else:
        active_cells["dist_fire_km"] = np.nan

    def norm(series, lo, hi, invert=False):
        s = series.clip(lo, hi)
        n = (s - lo) / (hi - lo) if hi > lo else s * 0
        n = n.fillna(0.5)  # donnée manquante -> contribution neutre
        return (1 - n) if invert else n

    wind_n = norm(active_cells["wind_speed_kmh"], 0, 60)
    dry_n = norm(active_cells["humidity_pct"], 20, 90, invert=True)
    temp_n = norm(active_cells["temperature_c"], 10, 40)
    drought_n = norm(active_cells["precip_today_mm"], 0, 10, invert=True)
    relief_n = norm(active_cells["elevation_m"], 0, 1500)
    proximity_n = active_cells["dist_fire_km"].apply(
        lambda d: 0.0 if pd.isna(d) else max(0.0, 1 - min(d, 30) / 30)
    )

    active_cells["risk_score"] = (
        100
        * (
            0.20 * active_cells["forest_density"]
            + 0.20 * wind_n
            + 0.18 * dry_n
            + 0.12 * temp_n
            + 0.10 * drought_n
            + 0.10 * relief_n
            + 0.10 * proximity_n
        )
    ).round(1)

    def risk_class(v):
        if v >= 70:
            return "Critique"
        if v >= 50:
            return "Élevé"
        if v >= 30:
            return "Modéré"
        return "Faible"

    active_cells["risk_class"] = active_cells["risk_score"].apply(risk_class)
    return active_cells


# =============================================================================
# 9. INTERFACE UTILISATEUR — BARRE LATÉRALE
# =============================================================================

def sidebar_controls():
    st.sidebar.title("🔥 FireWatch France")
    st.sidebar.caption("Surveillance des incendies — sources publiques gratuites")

    with st.sidebar.expander("🔑 Configuration NASA FIRMS", expanded=False):
        st.markdown(
            "Clé gratuite requise : "
            "[firms.modaps.eosdis.nasa.gov/api/map_key](https://firms.modaps.eosdis.nasa.gov/api/map_key/)"
        )
        default_key = st.secrets.get("FIRMS_MAP_KEY", "") if hasattr(st, "secrets") else ""
        map_key = st.text_input("MAP_KEY FIRMS", value=default_key, type="password")

    st.sidebar.header("📅 Période")
    period = st.sidebar.radio(
        "Fraîcheur des données",
        ["Temps réel (24h)", "Aujourd'hui / 48h", "7 derniers jours"],
        index=0,
        help="FIRMS priorise toujours la donnée la plus récente disponible (NRT).",
    )
    day_range = {"Temps réel (24h)": 1, "Aujourd'hui / 48h": 2, "7 derniers jours": 7}[period]

    st.sidebar.header("🎚️ Filtres")
    min_conf = st.sidebar.slider("Confiance minimale (%)", 0, 100, 30, step=5)
    intensities = st.sidebar.multiselect(
        "Intensité (puissance radiative)",
        ["Faible", "Modérée", "Forte", "Extrême", "Inconnue"],
        default=["Faible", "Modérée", "Forte", "Extrême", "Inconnue"],
    )
    sources_sel = st.sidebar.multiselect(
        "Sources",
        FIRMS_SOURCES + ["EFFIS"],
        default=FIRMS_SOURCES + ["EFFIS"],
    )

    st.sidebar.header("🗺️ Couches de la carte")
    show_clusters = st.sidebar.checkbox("Foyers d'incendie (agrégés)", value=True)
    show_raw = st.sidebar.checkbox("Détections brutes (satellite)", value=False)
    show_heatmap = st.sidebar.checkbox("Carte de chaleur", value=False)
    show_threat = st.sidebar.checkbox("🌲 Zones forestières menacées", value=False)

    st.sidebar.header("🔄 Actualisation automatique")
    refresh_choice = st.sidebar.selectbox(
        "Intervalle", ["Désactivée", "1 min", "5 min", "10 min", "30 min"], index=0
    )
    if not _AUTOREFRESH_OK and refresh_choice != "Désactivée":
        st.sidebar.warning("Package `streamlit-autorefresh` non installé — actualisation manuelle uniquement.")

    manual_refresh = st.sidebar.button("↻ Rafraîchir maintenant", use_container_width=True)

    return {
        "map_key": map_key,
        "day_range": day_range,
        "min_conf": min_conf,
        "intensities": intensities,
        "sources_sel": sources_sel,
        "show_clusters": show_clusters,
        "show_raw": show_raw,
        "show_heatmap": show_heatmap,
        "show_threat": show_threat,
        "refresh_choice": refresh_choice,
        "manual_refresh": manual_refresh,
    }


def sidebar_status_panel():
    _status_init()
    with st.sidebar.expander("📡 État des sources de données", expanded=False):
        statuses = st.session_state.get("api_status", {})
        if not statuses:
            st.caption("Aucun appel effectué pour l'instant.")
        for name, info in statuses.items():
            icon = "✅" if info["ok"] else "⚠️"
            st.markdown(f"{icon} **{name}** — {info['detail']}")
        st.caption(f"Dernière vérification : {dt.datetime.now().strftime('%H:%M:%S')}")


# =============================================================================
# 10. CONSTRUCTION DE LA CARTE FOLIUM (LEAFLET)
# =============================================================================

INTENSITY_COLORS = {
    "Faible": "#FFD166",
    "Modérée": "#F9844A",
    "Forte": "#EF476F",
    "Extrême": "#7B0D1E",
    "Inconnue": "#8D99AE",
}

RISK_COLORS = {
    "Faible": "#FFF3B0",
    "Modéré": "#FFC857",
    "Élevé": "#F26430",
    "Critique": "#8E0E00",
}


def build_map(fires_raw: pd.DataFrame, fires_clustered: pd.DataFrame, threat_grid: Optional[pd.DataFrame], opts: dict) -> folium.Map:
    fmap = folium.Map(
        location=FRANCE_CENTER,
        zoom_start=DEFAULT_ZOOM,
        tiles="CartoDB positron",
        control_scale=True,
        max_bounds=True,
        min_zoom=5,
    )
    # Contraint la carte à rester centrée sur la France (léger débordement autorisé)
    sw = (FRANCE_BBOX[1] - 2, FRANCE_BBOX[0] - 3)
    ne = (FRANCE_BBOX[3] + 2, FRANCE_BBOX[2] + 3)
    fmap.fit_bounds([sw, ne])

    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(fmap)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite",
    ).add_to(fmap)

    Fullscreen(position="topleft").add_to(fmap)
    MiniMap(toggle_display=True).add_to(fmap)

    # --- Couche : zones forestières menacées (dessinée en premier, sous les feux) ---
    if opts["show_threat"] and threat_grid is not None and not threat_grid.empty:
        threat_layer = folium.FeatureGroup(name="🌲 Zones forestières menacées", show=True)
        step = THREAT_GRID_STEP
        for _, row in threat_grid.iterrows():
            if row.get("risk_class") in (None, "Faible") or pd.isna(row.get("risk_score")):
                continue
            color = RISK_COLORS.get(row["risk_class"], "#999999")
            folium.Rectangle(
                bounds=[
                    (row["latitude"] - step / 2, row["longitude"] - step / 2),
                    (row["latitude"] + step / 2, row["longitude"] + step / 2),
                ],
                color=color,
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=0.35,
                tooltip=(
                    f"Risque {row['risk_class']} ({row['risk_score']}/100)<br>"
                    f"Vent: {row.get('wind_speed_kmh', '?')} km/h | "
                    f"Humidité: {row.get('humidity_pct', '?')}% | "
                    f"T°: {row.get('temperature_c', '?')}°C"
                ),
            ).add_to(threat_layer)
        threat_layer.add_to(fmap)

    # --- Couche : carte de chaleur ---
    if opts["show_heatmap"] and not fires_raw.empty:
        heat_data = [
            [r["latitude"], r["longitude"], float(r["frp"]) if pd.notna(r["frp"]) else 1.0]
            for _, r in fires_raw.iterrows()
        ]
        HeatMap(heat_data, name="Carte de chaleur", radius=12, blur=18, max_zoom=9).add_to(fmap)

    # --- Couche : détections brutes ---
    if opts["show_raw"] and not fires_raw.empty:
        raw_cluster = MarkerCluster(name="Détections brutes (satellite)")
        for _, r in fires_raw.iterrows():
            color = INTENSITY_COLORS.get(r["intensity_label"], "#8D99AE")
            popup_html = (
                f"<b>Détection satellite</b><br>"
                f"Source : {r['source']}<br>"
                f"Date : {r.get('acq_date','?')} {r.get('acq_time','')} UTC<br>"
                f"Confiance : {r.get('confidence_pct','?')}%<br>"
                f"FRP : {r.get('frp','?')} MW ({r['intensity_label']})<br>"
                f"Jour/Nuit : {r.get('daynight','?')}"
            )
            folium.CircleMarker(
                location=(r["latitude"], r["longitude"]),
                radius=4,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                popup=folium.Popup(popup_html, max_width=280),
            ).add_to(raw_cluster)
        raw_cluster.add_to(fmap)

    # --- Couche : foyers agrégés (vue principale recommandée) ---
    if opts["show_clusters"] and not fires_clustered.empty:
        cluster_layer = folium.FeatureGroup(name="Foyers d'incendie (agrégés)", show=True)
        for _, r in fires_clustered.iterrows():
            color = INTENSITY_COLORS.get(r["intensity_label"], "#8D99AE")
            radius = 6 + min(18, math.sqrt(max(r["area_km2_est"], 0.1)) * 4)
            popup_html = f"""
            <div style="font-family: sans-serif; font-size: 13px;">
              <b style="font-size:14px;">🔥 Foyer d'incendie</b><br>
              <b>Position :</b> {r['latitude']:.3f}, {r['longitude']:.3f}<br>
              <b>Première détection :</b> {r['first_seen']}<br>
              <b>Dernière détection :</b> {r['last_seen']}<br>
              <b>Évolution :</b> {r['trend']}<br>
              <b>Intensité :</b> {r['intensity_label']} (FRP max {r['frp_max']:.1f} MW)<br>
              <b>Superficie estimée :</b> ~{r['area_km2_est']} km²<br>
              <b>Confiance moyenne :</b> {r['confidence_pct']:.0f}%<br>
              <b>Détections cumulées :</b> {r['n_detections']}<br>
              <b>Source(s) :</b> {r['sources']}
            </div>
            """
            folium.CircleMarker(
                location=(r["latitude"], r["longitude"]),
                radius=radius,
                color="#3a0000",
                weight=1.5,
                fill=True,
                fill_color=color,
                fill_opacity=0.75,
                popup=folium.Popup(popup_html, max_width=320),
                tooltip=f"{r['intensity_label']} — {r['trend']} — ~{r['area_km2_est']} km²",
            ).add_to(cluster_layer)
        cluster_layer.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap


# =============================================================================
# 11. LÉGENDE / EN-TÊTE
# =============================================================================

def render_legend():
    cols = st.columns(6)
    labels = [("Faible", "#FFD166"), ("Modérée", "#F9844A"), ("Forte", "#EF476F"), ("Extrême", "#7B0D1E")]
    for col, (label, color) in zip(cols, labels):
        col.markdown(
            f"<span style='background:{color};padding:2px 10px;border-radius:10px;color:white;font-size:12px'>{label}</span>",
            unsafe_allow_html=True,
        )


# =============================================================================
# 12. PROGRAMME PRINCIPAL
# =============================================================================

def main():
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1rem; padding-bottom: 0rem;}
        iframe {min-height: 72vh;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    opts = sidebar_controls()

    # Actualisation automatique configurable
    interval_map = {"1 min": 60_000, "5 min": 300_000, "10 min": 600_000, "30 min": 1_800_000}
    if _AUTOREFRESH_OK and opts["refresh_choice"] in interval_map:
        st_autorefresh(interval=interval_map[opts["refresh_choice"]], key="auto_refresh_timer")

    if opts["manual_refresh"]:
        fetch_all_fires.clear()
        fetch_effis.clear()
        fetch_firms.clear()
        st.rerun()

    st.title("🔥 Incendies en cours — France")
    st.caption(
        f"Dernière actualisation : {dt.datetime.now().strftime('%d/%m/%Y %H:%M:%S')} · "
        "Sources : NASA FIRMS (VIIRS/MODIS), EFFIS/Copernicus, Open-Meteo, OpenStreetMap"
    )

    if not opts["map_key"]:
        st.warning(
            "⚠️ Aucune clé NASA FIRMS renseignée : les détections satellite ne pourront pas être chargées. "
            "Obtenez une clé gratuite via le panneau 'Configuration NASA FIRMS' dans la barre latérale."
        )

    # --- Récupération des données de feux ---
    with st.spinner("Récupération des données incendies..."):
        fires_all = fetch_all_fires(opts["map_key"], opts["day_range"])

    # --- Application des filtres utilisateur ---
    fires_filtered = fires_all.copy()
    if not fires_filtered.empty:
        fires_filtered = fires_filtered[
            (fires_filtered["confidence_pct"].fillna(0) >= opts["min_conf"])
            & (fires_filtered["intensity_label"].isin(opts["intensities"]))
            & (fires_filtered["source"].isin(opts["sources_sel"]))
        ]

    fires_clustered = cluster_fires(fires_filtered)

    # --- Zones menacées (calculées uniquement si la couche est activée -> économie d'appels API) ---
    threat_grid = None
    if opts["show_threat"]:
        with st.spinner("Calcul des zones forestières à risque (météo + relief + végétation)..."):
            threat_grid = compute_threat_grid(fires_clustered)

    # --- Indicateurs synthétiques ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Détections brutes", len(fires_filtered))
    c2.metric("Foyers identifiés", len(fires_clustered))
    n_critical = 0
    if not fires_clustered.empty:
        n_critical = (fires_clustered["intensity_label"].isin(["Forte", "Extrême"])).sum()
    c3.metric("Foyers forte/extrême intensité", int(n_critical))
    n_threat = 0
    if threat_grid is not None and not threat_grid.empty and "risk_class" in threat_grid.columns:
        n_threat = (threat_grid["risk_class"].isin(["Élevé", "Critique"])).sum()
    c4.metric("Zones forestières à risque élevé/critique", int(n_threat))

    render_legend()

    # --- Carte plein écran ---
    fmap = build_map(fires_filtered, fires_clustered, threat_grid, opts)
    st_folium(fmap, use_container_width=True, height=650, returned_objects=[])

    # --- Tableau détaillé (optionnel) ---
    with st.expander("📋 Détail des foyers identifiés (tableau)"):
        if fires_clustered.empty:
            st.info("Aucun foyer ne correspond aux filtres actuels.")
        else:
            display_cols = [
                "latitude", "longitude", "intensity_label", "trend", "area_km2_est",
                "confidence_pct", "n_detections", "first_seen", "last_seen", "sources",
            ]
            st.dataframe(
                fires_clustered[display_cols].sort_values("frp_max", ascending=False, na_position="last"),
                use_container_width=True,
                hide_index=True,
            )

    sidebar_status_panel()

    st.caption(
        "ℹ️ La couche « Zones forestières menacées » est un indice heuristique de sensibilisation basé sur "
        "des données publiques (végétation OSM, météo Open-Meteo, relief). Elle ne remplace pas les bulletins "
        "officiels de risque incendie (Météo-France, préfectures)."
    )


if __name__ == "__main__":
    main()