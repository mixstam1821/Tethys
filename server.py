"""
Tethys Greece — MCP Server
==========================
Exposes ERA5-based tools for heat events and water stress over Greece.
Uses the same CDS API stack as ERMES — cdsapi + xarray + numpy.

Tools:
  - get_heat_events          : heatwave detection, heat day count
  - get_water_stress         : WSI = evaporation / precipitation
  - get_temperature_anomaly  : 2m T vs 1991–2020 baseline
  - get_drought_index        : SPI-like standardised precipitation index
  - get_wind_speed_anomaly   : 10m wind speed vs 1991–2020 baseline
  - get_solar_radiation      : surface solar radiation (SSRD)
  - get_sea_surface_temp     : SST anomaly for Greek seas

━━━ DATA RESOLUTION STRATEGY (3-tier, fastest → slowest) ━━━━━━━━━━━━━━━━━━

  TIER 1 — Pre-loaded local files  (./era5_local/)    ← INSTANT
    Big consolidated .nc files you downloaded manually from the CDS web UI.
    One file per variable, covering ALL years + ALL months.
    The server slices the exact year/month it needs from these files.
    See LOCAL DATA SPEC at the bottom of this docstring for exactly what to
    download.

  TIER 2 — Query-level cache  (./era5_cache/)          ← INSTANT (after first run)
    Small .nc files saved automatically after every CDS download.
    Named by SHA-1 of the request params.

  TIER 3 — Live CDS API download  (~60 s per request)  ← FALLBACK
    Used only when neither tier 1 nor tier 2 has the data.

━━━ LOCAL DATA SPEC — what to download from the CDS web UI ━━━━━━━━━━━━━━━━

  Dataset : ERA5 monthly averaged reanalysis
  URL     : https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means

  For ALL files below use these settings:
    Product type : Monthly averaged reanalysis
    Year         : 1940 – 2024  (or at least 1991 – present for anomalies)
    Month        : 01 02 03 04 05 06 07 08 09 10 11 12
    Time         : 00:00
    Area (N/W/S/E): 42 / 19 / 34.5 / 30   (Greece)
    Format       : NetCDF

  Variables to download (one file each, save with the exact filename shown):

    Filename                          CDS variable name
    ────────────────────────────────────────────────────────────────
    era5_monthly_t2m.nc               2m_temperature
    era5_monthly_tp.nc                total_precipitation
    era5_monthly_evap.nc              evaporation
    era5_monthly_u10.nc               10m_u_component_of_wind
    era5_monthly_v10.nc               10m_v_component_of_wind
    era5_monthly_ssrd.nc              surface_solar_radiation_downwards
    era5_monthly_sst.nc               sea_surface_temperature
                                        (use area 42/18/33/30.5 for SST)

  For hourly heat events (get_heat_events tool):
    Dataset : ERA5 hourly reanalysis (reanalysis-era5-single-levels)
    Variable: 2m_temperature
    Year    : whichever years you want instant heat-event analysis for
              (each month is a separate file, named era5_hourly_t2m_YYYY_MM.nc)
    Time    : 00:00 01:00 … 23:00
    Area    : 42 / 19 / 34.5 / 30

  Place all files in:  ./era5_local/
"""

import asyncio
import hashlib
import json
import os
import shutil
import tempfile
import warnings
from pathlib import Path

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
CDS_KEY = os.environ.get("CDS_KEY")
if not CDS_KEY:
    raise RuntimeError("CDS_KEY environment variable not set")

CDS_URL = "https://cds.climate.copernicus.eu/api"

GREECE_BBOX     = {"north": 42.0, "west": 19.0, "south": 34.5, "east": 30.0}
GREEK_SEAS_BBOX = {"north": 42.0, "west": 18.0, "south": 33.0, "east": 30.5}

# Tier 1 — pre-loaded consolidated files (you download these manually)
LOCAL_DIR  = Path(os.environ.get("ERA5_LOCAL_DIR",  "./era5_local"))
# Tier 2 — auto-saved cache from live CDS downloads
CACHE_DIR  = Path(os.environ.get("ERA5_CACHE_DIR",  "./era5_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

# Maps CDS variable name → local consolidated filename
LOCAL_FILES: dict[str, str] = {
    "2m_temperature":                       "era5_monthly_t2m.nc",
    "total_precipitation":                  "era5_monthly_tp.nc",
    "evaporation":                          "era5_monthly_evap.nc",
    "10m_u_component_of_wind":              "era5_monthly_u10.nc",
    "10m_v_component_of_wind":              "era5_monthly_v10.nc",
    "surface_solar_radiation_downwards":    "era5_monthly_ssrd.nc",
    "sea_surface_temperature":              "era5_monthly_sst.nc",
}
# ERA5 short variable names in the .nc files (what xarray actually calls them)
NC_VARNAME: dict[str, str] = {
    "2m_temperature":                       "t2m",
    "total_precipitation":                  "tp",
    "evaporation":                          "e",
    "10m_u_component_of_wind":              "u10",
    "10m_v_component_of_wind":              "v10",
    "surface_solar_radiation_downwards":    "ssrd",
    "sea_surface_temperature":              "sst",
}

app = Server("Tethys-greece-climate")


# ── TIER 1 — LOCAL PRE-LOADED DATA ───────────────────────────────────────────

def _load_local_monthly(
    variable: str,
    years: list[int | str],
    months: list[str],
) -> "xr.Dataset | None":
    """
    Try to slice the required years+months from a pre-downloaded consolidated file.
    Returns an xr.Dataset on success, None if the file doesn't exist or doesn't
    cover the requested time range.
    """
    fname = LOCAL_FILES.get(variable)
    if not fname:
        return None
    path = LOCAL_DIR / fname
    if not path.exists():
        return None

    try:
        ds = xr.open_dataset(str(path))

        # Normalise the time coordinate name
        time_dim = next(
            (d for d in ds.dims if d in ("time", "valid_time", "expver")),
            None,
        )
        if time_dim is None:
            return None

        # Build a set of (year, month) tuples we need
        needed = {
            (int(y), int(m)) for y in years for m in months
        }

        # Filter time steps
        times = pd.to_datetime(ds[time_dim].values)
        mask  = [(t.year, t.month) in needed for t in times]
        if not any(mask):
            return None  # file doesn't cover requested period

        ds_sel = ds.isel({time_dim: mask})
        return ds_sel

    except Exception:
        return None


def _load_local_hourly(variable: str, year: str, month: str) -> "xr.Dataset | None":
    """
    Try to load pre-downloaded ERA5 hourly data for a specific year+month.
    Expected filename: era5_hourly_t2m_YYYY_MM.nc  (only t2m supported for now)
    """
    short = NC_VARNAME.get(variable, variable)
    fname = f"era5_hourly_{short}_{year}_{month}.nc"
    path  = LOCAL_DIR / fname
    if not path.exists():
        return None
    try:
        return xr.open_dataset(str(path))
    except Exception:
        return None


# ── TIER 2 — QUERY-LEVEL CACHE ───────────────────────────────────────────────

def _cache_path(dataset: str, params: dict) -> Path:
    blob = json.dumps({"dataset": dataset, **params}, sort_keys=True)
    h    = hashlib.sha1(blob.encode()).hexdigest()[:14]
    return CACHE_DIR / f"{h}.nc"


def _try_cache(dataset: str, params: dict) -> "xr.Dataset | None":
    p = _cache_path(dataset, params)
    if p.exists():
        try:
            return xr.open_dataset(str(p))
        except Exception:
            p.unlink(missing_ok=True)
    return None


def _save_cache(dataset: str, params: dict, tmp_path: str) -> None:
    try:
        shutil.copy2(tmp_path, _cache_path(dataset, params))
    except Exception:
        pass


# ── TIER 3 — LIVE CDS DOWNLOAD ───────────────────────────────────────────────

def _cds_client() -> cdsapi.Client:
    return cdsapi.Client(url=CDS_URL, key=CDS_KEY, verify=False, quiet=True)


# ── UNIFIED FETCH — monthly ───────────────────────────────────────────────────

def _fetch_era5_monthly(
    variable: str,
    years: list,
    months: list,
    bbox: dict | None = None,
) -> xr.Dataset:
    """
    3-tier fetch for ERA5 monthly data.
    1. Pre-loaded local file (era5_local/) — instant
    2. Query cache (era5_cache/)           — instant
    3. Live CDS download                   — ~60 s, then saved to cache
    """
    if bbox is None:
        bbox = GREECE_BBOX

    int_years  = [int(y) for y in years]
    str_months = [str(m).zfill(2) for m in months]

    # ── Tier 1: pre-loaded local data ────────────────────────────────────────
    ds = _load_local_monthly(variable, int_years, str_months)
    if ds is not None:
        return ds

    # ── Tier 2: query cache ───────────────────────────────────────────────────
    area = [bbox["north"], bbox["west"], bbox["south"], bbox["east"]]
    cds_params: dict = {
        "product_type": "monthly_averaged_reanalysis",
        "variable":     [variable],
        "year":         sorted(set(str(y) for y in int_years)),
        "month":        sorted(set(str_months)),
        "time":         "00:00",
        "area":         area,
        "format":       "netcdf",
    }
    cache_key = {"var": variable, **cds_params}

    ds = _try_cache("monthly", cache_key)
    if ds is not None:
        return ds

    # ── Tier 3: live CDS download ─────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = tmp.name
    _cds_client().retrieve(
        "reanalysis-era5-single-levels-monthly-means", cds_params, tmp_path
    )
    _save_cache("monthly", cache_key, tmp_path)
    return xr.open_dataset(tmp_path)


# ── UNIFIED FETCH — hourly ────────────────────────────────────────────────────

def _fetch_era5_hourly(
    variable: str,
    year: str,
    month: str,
    bbox: dict | None = None,
) -> xr.Dataset:
    """
    3-tier fetch for ERA5 hourly data (used by get_heat_events).
    """
    if bbox is None:
        bbox = GREECE_BBOX

    # ── Tier 1: pre-loaded local file ────────────────────────────────────────
    ds = _load_local_hourly(variable, year, month)
    if ds is not None:
        return ds

    # ── Tier 2: query cache ───────────────────────────────────────────────────
    area = [bbox["north"], bbox["west"], bbox["south"], bbox["east"]]
    cds_params: dict = {
        "product_type": "reanalysis",
        "variable":     [variable],
        "year":         year,
        "month":        month,
        "day":          [f"{d:02d}" for d in range(1, 32)],
        "time":         [f"{h:02d}:00" for h in range(24)],
        "area":         area,
        "format":       "netcdf",
    }
    cache_key = {"var": variable, **cds_params}

    ds = _try_cache("hourly", cache_key)
    if ds is not None:
        return ds

    # ── Tier 3: live CDS download ─────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = tmp.name
    _cds_client().retrieve("reanalysis-era5-single-levels", cds_params, tmp_path)
    _save_cache("hourly", cache_key, tmp_path)
    return xr.open_dataset(tmp_path)


# ── SPATIAL HELPERS ───────────────────────────────────────────────────────────

def _spatial_mean(da: xr.DataArray) -> xr.DataArray:
    lat_name = next((d for d in da.dims if "lat" in d), "latitude")
    lon_name = next((d for d in da.dims if "lon" in d), "longitude")
    weights  = np.cos(np.deg2rad(da[lat_name]))
    return da.weighted(weights).mean(dim=[lat_name, lon_name])


def _safe_float(x) -> "float | None":
    v = float(x)
    return round(v, 4) if np.isfinite(v) else None


def _ms_to_beaufort(ms: float) -> int:
    for b, t in enumerate([0.3,1.6,3.4,5.5,8.0,10.8,13.9,17.2,20.8,24.5,28.5,32.7]):
        if ms < t:
            return b
    return 12


SEASON_MONTHS = {
    "spring": ["03","04","05"],
    "summer": ["06","07","08"],
    "autumn": ["09","10","11"],
    "winter": ["12","01","02"],
}
SEASON_DAYS = {"spring": 92, "summer": 92, "autumn": 91, "winter": 90}


# ── TOOL IMPLEMENTATIONS ──────────────────────────────────────────────────────

def heat_events(year: int, month: int, threshold_c: float = 35.0) -> dict:
    ds  = _fetch_era5_hourly("2m_temperature", str(year), f"{month:02d}")
    t2m = ds["t2m"] - 273.15

    t_spatial = _spatial_mean(t2m)
    time_dim  = next(d for d in t_spatial.dims if "time" in d or "valid" in d)
    t_daily   = t_spatial.resample({time_dim: "1D"}).max()

    vals  = t_daily.values
    times = pd.to_datetime(t_daily[time_dim].values)

    heat_mask  = vals > threshold_c
    n_heat     = int(heat_mask.sum())
    heat_dates = [str(times[i].date()) for i in np.where(heat_mask)[0]]

    heatwaves, streak, start_i = [], 0, None
    for i, hot in enumerate(heat_mask):
        if hot:
            if streak == 0:
                start_i = i
            streak += 1
        else:
            if streak >= 3:
                heatwaves.append({"start": str(times[start_i].date()),
                                   "end":   str(times[i-1].date()),
                                   "days":  streak,
                                   "peak_c": round(float(vals[start_i:i].max()), 2)})
            streak = 0
    if streak >= 3:
        heatwaves.append({"start": str(times[start_i].date()),
                           "end":   str(times[-1].date()),
                           "days":  streak,
                           "peak_c": round(float(vals[start_i:].max()), 2)})

    return {
        "region": "Greece", "year": year, "month": month,
        "threshold_c": threshold_c,
        "tmax_mean_c": _safe_float(np.nanmean(vals)),
        "tmax_max_c":  _safe_float(np.nanmax(vals)),
        "n_heat_days": n_heat, "heat_dates": heat_dates,
        "n_heatwaves": len(heatwaves), "heatwaves": heatwaves,
        "source": "ERA5 hourly — Copernicus CDS",
    }


def water_stress(year: int, season: str) -> dict:
    months = SEASON_MONTHS.get(season.lower())
    if not months:
        raise ValueError(f"Unknown season '{season}'.")

    ds_tp = _fetch_era5_monthly("total_precipitation", [str(year)], months)
    ds_ev = _fetch_era5_monthly("evaporation",         [str(year)], months)

    tp_mm = float(_spatial_mean(ds_tp["tp"]).sum()) * 1000
    ev_mm = float(_spatial_mean(ds_ev["e"]).sum())  * -1000
    wsi   = round(ev_mm / tp_mm, 3) if tp_mm > 0 else None

    status = ("no data" if wsi is None else "severe stress" if wsi > 2.0
              else "moderate stress" if wsi > 1.0 else "mild stress" if wsi > 0.5
              else "adequate water")

    return {
        "region": "Greece", "year": year, "season": season,
        "total_precip_mm": round(tp_mm, 2), "total_evap_mm": round(ev_mm, 2),
        "water_stress_index": wsi, "status": status,
        "interpretation": "WSI = evaporation / precipitation. WSI > 1 = water deficit.",
        "source": "ERA5 monthly — Copernicus CDS",
    }


def temperature_anomaly(year: int, month: int,
                         baseline_start: int = 1991,
                         baseline_end:   int = 2020) -> dict:
    mon        = f"{month:02d}"
    ds_t       = _fetch_era5_monthly("2m_temperature", [str(year)], [mon])
    t_target   = float(_spatial_mean(ds_t["t2m"]).mean()) - 273.15

    base_years = [str(y) for y in range(baseline_start, baseline_end + 1)]
    ds_b       = _fetch_era5_monthly("2m_temperature", base_years, [mon])
    t_base     = float(_spatial_mean(ds_b["t2m"]).mean()) - 273.15
    anomaly    = round(t_target - t_base, 3)

    category = ("strongly above normal" if anomaly >  2.0 else
                "above normal"          if anomaly >  0.5 else
                "strongly below normal" if anomaly < -2.0 else
                "below normal"          if anomaly < -0.5 else "near normal")

    return {
        "region": "Greece", "year": year, "month": month,
        "t2m_c": round(t_target, 3), "baseline_mean_c": round(t_base, 3),
        "baseline_period": f"{baseline_start}–{baseline_end}",
        "anomaly_c": anomaly, "category": category,
        "source": "ERA5 monthly — Copernicus CDS",
    }


def drought_index(year: int, season: str) -> dict:
    months    = SEASON_MONTHS.get(season.lower())
    if not months:
        raise ValueError(f"Unknown season '{season}'.")

    ds_t      = _fetch_era5_monthly("total_precipitation", [str(year)], months)
    tp_target = float(_spatial_mean(ds_t["tp"]).sum()) * 1000

    base_years = [str(y) for y in range(1991, 2021)]
    ds_b       = _fetch_era5_monthly("total_precipitation", base_years, months)
    tp_series  = _spatial_mean(ds_b["tp"]) * 1000

    time_dim  = next(d for d in tp_series.dims if "time" in d or "valid" in d)
    tp_annual = tp_series.to_series().resample("YE").sum()
    base_mean = float(tp_annual.mean())
    base_std  = float(tp_annual.std())

    spi = round((tp_target - base_mean) / base_std, 3) if base_std > 0 else None
    category = ("no data" if spi is None else "extreme drought" if spi < -2.0
                else "severe drought" if spi < -1.5 else "moderate drought" if spi < -1.0
                else "mildly dry" if spi < 0 else "near normal" if spi < 1.0 else "wet")

    return {
        "region": "Greece", "year": year, "season": season,
        "precip_mm": round(tp_target, 2),
        "baseline_mean_mm": round(base_mean, 2), "baseline_std_mm": round(base_std, 2),
        "spi": spi, "category": category, "baseline_period": "1991–2020",
        "source": "ERA5 monthly — Copernicus CDS",
    }


def wind_speed_anomaly(year: int, month: int,
                        baseline_start: int = 1991,
                        baseline_end:   int = 2020) -> dict:
    mon        = f"{month:02d}"
    base_years = [str(y) for y in range(baseline_start, baseline_end + 1)]

    def _ws(ds_u, ds_v):
        # Compute per-timestep magnitude then average.
        # Averaging components first is wrong when wind direction varies across months.
        u = _spatial_mean(ds_u["u10"])   # DataArray (time,)
        v = _spatial_mean(ds_v["v10"])   # DataArray (time,)
        return float(np.sqrt(u**2 + v**2).mean())

    ws_target = _ws(
        _fetch_era5_monthly("10m_u_component_of_wind", [str(year)], [mon]),
        _fetch_era5_monthly("10m_v_component_of_wind", [str(year)], [mon]),
    )
    ws_base = _ws(
        _fetch_era5_monthly("10m_u_component_of_wind", base_years, [mon]),
        _fetch_era5_monthly("10m_v_component_of_wind", base_years, [mon]),
    )
    anomaly  = round(ws_target - ws_base, 3)
    category = ("strongly above normal" if anomaly >  1.5 else
                "above normal"          if anomaly >  0.3 else
                "strongly below normal" if anomaly < -1.5 else
                "below normal"          if anomaly < -0.3 else "near normal")

    return {
        "region": "Greece", "year": year, "month": month,
        "wind_speed_ms": round(ws_target, 3), "baseline_mean_ms": round(ws_base, 3),
        "baseline_period": f"{baseline_start}–{baseline_end}",
        "anomaly_ms": anomaly, "category": category,
        "beaufort": _ms_to_beaufort(ws_target),
        "source": "ERA5 monthly (u10, v10) — Copernicus CDS",
    }


def solar_radiation(year: int, season: str) -> dict:
    months = SEASON_MONTHS.get(season.lower())
    if not months:
        raise ValueError(f"Unknown season '{season}'.")

    def _jm2_per_day(ds) -> float:
        """
        ERA5 monthly ssrd with stepType=avgad stores the MEAN DAILY accumulation
        for each month in J/m²/day — NOT a monthly total.
        Physics check: June 2023 Greece raw value = 25,966,934 J/m²/day ≈ 300 W/m² ✓
        So: just average across the season months. No division by n_days needed.
        .mean() also correctly normalises across any number of years in the baseline.
        """
        return float(_spatial_mean(ds["ssrd"]).mean())

    ds_t     = _fetch_era5_monthly("surface_solar_radiation_downwards", [str(year)], months)
    ssrd_val = round(_jm2_per_day(ds_t), 3)

    base_years = [str(y) for y in range(1991, 2021)]
    ds_b       = _fetch_era5_monthly("surface_solar_radiation_downwards", base_years, months)
    ssrd_base  = round(_jm2_per_day(ds_b), 3)

    anomaly  = round(ssrd_val - ssrd_base, 3)

    # Thresholds in J/m²/day — typical Greek summer ~25 MJ/m²/day
    # ±500,000 J ≈ ±2%  →  above/below normal
    # ±1,500,000 J ≈ ±6% →  well above/below normal
    category = ("well above normal" if anomaly >  1_500_000 else
                "above normal"      if anomaly >    500_000 else
                "well below normal" if anomaly < -1_500_000 else
                "below normal"      if anomaly <   -500_000 else "near normal")

    return {
        "region": "Greece", "year": year, "season": season,
        "ssrd_J_m2_day":      ssrd_val,
        "ssrd_MJ_m2_day":     round(ssrd_val  / 1_000_000, 3),
        "baseline_J_m2_day":  ssrd_base,
        "baseline_MJ_m2_day": round(ssrd_base / 1_000_000, 3),
        "baseline_period": "1991–2020",
        "anomaly_J_m2_day":   anomaly,
        "anomaly_MJ_m2_day":  round(anomaly   / 1_000_000, 3),
        "category": category,
        "interpretation": "Higher SSRD = more solar energy; amplifies heat stress and wildfire risk.",
        "source": "ERA5 monthly (ssrd) — Copernicus CDS",
    }

def sea_surface_temperature(year: int, month: int,
                              baseline_start: int = 1991,
                              baseline_end:   int = 2020) -> dict:
    mon        = f"{month:02d}"
    base_years = [str(y) for y in range(baseline_start, baseline_end + 1)]

    def _sst(ds):
        da = ds["sst"] - 273.15
        return float(_spatial_mean(da.where(da > -100)).mean())

    sst_target = _sst(_fetch_era5_monthly("sea_surface_temperature",
                                           [str(year)], [mon], bbox=GREEK_SEAS_BBOX))
    sst_base   = _sst(_fetch_era5_monthly("sea_surface_temperature",
                                           base_years, [mon], bbox=GREEK_SEAS_BBOX))

    anomaly  = round(sst_target - sst_base, 3)
    category = ("strongly above normal — marine heat wave risk" if anomaly >  1.5 else
                "above normal"                                   if anomaly >  0.3 else
                "strongly below normal"                          if anomaly < -1.5 else
                "below normal"                                   if anomaly < -0.3 else
                "near normal")

    return {
        "region": "Greek seas (Aegean + Ionian)",
        "year": year, "month": month,
        "sst_c": round(sst_target, 3), "baseline_mean_c": round(sst_base, 3),
        "baseline_period": f"{baseline_start}–{baseline_end}",
        "anomaly_c": anomaly, "category": category,
        "interpretation": "SST anomaly > +1.5 °C can trigger marine heatwaves and intensify Medicanes.",
        "source": "ERA5 monthly (sst) — Copernicus CDS",
    }


# ── MCP TOOL REGISTRY ─────────────────────────────────────────────────────────

@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="get_heat_events",
            description=(
                "Returns heat event statistics for Greece for a specific year and month. "
                "Counts heat days (daily Tmax > threshold °C), detects heatwave periods "
                "(≥3 consecutive hot days), and reports peak temperatures. "
                "Use when asked about heatwaves, hot spells, extreme heat, or high "
                "temperature events in Greece."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "year":        {"type": "integer", "description": "Year, e.g. 2021"},
                    "month":       {"type": "integer", "description": "Month number 1–12"},
                    "threshold_c": {"type": "number",  "description": "Heat day threshold in °C (default 35)"},
                },
                "required": ["year", "month"],
            },
        ),
        Tool(
            name="get_water_stress",
            description=(
                "Returns a water stress index (WSI) for Greece for a given year and season. "
                "WSI = evaporation / precipitation from ERA5 monthly data. "
                "WSI > 1 means more water lost than received — indicates stress. "
                "Use when asked about water availability, drought risk, irrigation "
                "pressure, or hydrological stress in Greece."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "year":   {"type": "integer", "description": "Year, e.g. 2022"},
                    "season": {"type": "string",  "description": "spring / summer / autumn / winter"},
                },
                "required": ["year", "season"],
            },
        ),
        Tool(
            name="get_temperature_anomaly",
            description=(
                "Returns the 2m temperature anomaly for Greece vs the 1991–2020 "
                "climatological baseline for a specific year and month. "
                "Use when asked whether a month was unusually warm or cold, or about "
                "temperature departures from normal in Greece."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "year":           {"type": "integer", "description": "Year"},
                    "month":          {"type": "integer", "description": "Month 1–12"},
                    "baseline_start": {"type": "integer", "description": "Baseline start year (default 1991)"},
                    "baseline_end":   {"type": "integer", "description": "Baseline end year (default 2020)"},
                },
                "required": ["year", "month"],
            },
        ),
        Tool(
            name="get_drought_index",
            description=(
                "Returns a standardised precipitation index (SPI-like) for Greece, "
                "comparing seasonal precipitation to the 1991–2020 baseline. "
                "Negative SPI = drier than normal. "
                "Use when asked about drought severity, precipitation deficit, or "
                "dry/wet seasonal conditions in Greece."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "year":   {"type": "integer", "description": "Year"},
                    "season": {"type": "string",  "description": "spring / summer / autumn / winter"},
                },
                "required": ["year", "season"],
            },
        ),
        Tool(
            name="get_wind_speed_anomaly",
            description=(
                "Returns 10m wind speed anomaly for Greece vs the 1991–2020 baseline "
                "for a specific year and month. Derived from ERA5 u10/v10 components. "
                "Includes Beaufort scale classification. "
                "Use when asked about wind conditions, Meltemi anomalies, renewable energy "
                "potential, or wildfire spread risk in Greece."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "year":           {"type": "integer", "description": "Year"},
                    "month":          {"type": "integer", "description": "Month 1–12"},
                    "baseline_start": {"type": "integer", "description": "Baseline start year (default 1991)"},
                    "baseline_end":   {"type": "integer", "description": "Baseline end year (default 2020)"},
                },
                "required": ["year", "month"],
            },
        ),
        Tool(
            name="get_solar_radiation",
            description=(
                "Returns surface solar downwelling radiation (SSRD) for Greece for a "
                "given year and season, in MJ/m²/day, compared to the 1991–2020 baseline. "
                "Use when asked about solar energy potential, sunshine anomalies, "
                "photovoltaic output estimates, or radiation-driven wildfire and heat risk."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "year":   {"type": "integer", "description": "Year"},
                    "season": {"type": "string",  "description": "spring / summer / autumn / winter"},
                },
                "required": ["year", "season"],
            },
        ),
        Tool(
            name="get_sea_surface_temp",
            description=(
                "Returns sea surface temperature (SST) anomaly for Greek seas "
                "(Aegean + Ionian) vs the 1991–2020 baseline for a specific year and month. "
                "Use when asked about marine heatwaves, Medicane conditions, "
                "fisheries stress, or coastal sea temperatures in Greece."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "year":           {"type": "integer", "description": "Year"},
                    "month":          {"type": "integer", "description": "Month 1–12"},
                    "baseline_start": {"type": "integer", "description": "Baseline start year (default 1991)"},
                    "baseline_end":   {"type": "integer", "description": "Baseline end year (default 2020)"},
                },
                "required": ["year", "month"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    try:
        if name == "get_heat_events":
            result = heat_events(year=arguments["year"], month=arguments["month"],
                                  threshold_c=arguments.get("threshold_c", 35.0))
        elif name == "get_water_stress":
            result = water_stress(year=arguments["year"], season=arguments["season"])
        elif name == "get_temperature_anomaly":
            result = temperature_anomaly(year=arguments["year"], month=arguments["month"],
                                          baseline_start=arguments.get("baseline_start", 1991),
                                          baseline_end=arguments.get("baseline_end", 2020))
        elif name == "get_drought_index":
            result = drought_index(year=arguments["year"], season=arguments["season"])
        elif name == "get_wind_speed_anomaly":
            result = wind_speed_anomaly(year=arguments["year"], month=arguments["month"],
                                         baseline_start=arguments.get("baseline_start", 1991),
                                         baseline_end=arguments.get("baseline_end", 2020))
        elif name == "get_solar_radiation":
            result = solar_radiation(year=arguments["year"], season=arguments["season"])
        elif name == "get_sea_surface_temp":
            result = sea_surface_temperature(year=arguments["year"], month=arguments["month"],
                                              baseline_start=arguments.get("baseline_start", 1991),
                                              baseline_end=arguments.get("baseline_end", 2020))
        else:
            result = {"error": f"Unknown tool: {name}"}

    except Exception as exc:
        result = {"error": str(exc), "tool": name, "arguments": arguments}

    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

async def main():
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
