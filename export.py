"""Reproyecta las probabilidades a una malla lat/lon regular y las empaqueta para la web."""
import gzip
import json
from pathlib import Path

import numpy as np

from engine import THRESHOLDS

CELL_DEG = 0.04  # ~4,4 km


def latlon_grid(geo, h, w, cell=CELL_DEG):
    """Caja envolvente (lat/lon) de la ventana y mallas de índices fuente para cada celda."""
    ec = np.concatenate([np.arange(w), np.full(h, w - 1), np.arange(w), np.zeros(h)])
    er = np.concatenate([np.zeros(w), np.arange(h), np.full(w, h - 1), np.arange(h)])
    lon_e, lat_e = geo.pixel_to_lonlat(ec, er)
    lon_w = np.floor(np.nanmin(lon_e) / cell) * cell
    lon_e_ = np.ceil(np.nanmax(lon_e) / cell) * cell
    lat_s = np.floor(np.nanmin(lat_e) / cell) * cell
    lat_n = np.ceil(np.nanmax(lat_e) / cell) * cell
    nlon = int(round((lon_e_ - lon_w) / cell))
    nlat = int(round((lat_n - lat_s) / cell))
    lons = lon_w + (np.arange(nlon) + 0.5) * cell
    lats = lat_n - (np.arange(nlat) + 0.5) * cell
    LON, LAT = np.meshgrid(lons, lats)
    col, row = geo.lonlat_to_pixel(LON, LAT)
    ci, ri = np.round(col).astype(int), np.round(row).astype(int)
    inside = (ci >= 0) & (ci < w) & (ri >= 0) & (ri < h)
    return {"lon_w": lon_w, "lat_n": lat_n, "nlon": nlon, "nlat": nlat, "cell": cell,
            "ci": ci, "ri": ri, "inside": inside}


def export(out_dir, geo, probs, observed_rate, leads_min, issued, generated, source, mode,
           members, motion_ok, stale_after_min=30, motion=None):
    """probs (L, T, H, W) uint8; observed_rate (H, W) -> plazo 0 observado."""
    out_dir = Path(out_dir)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    nl, nt, h, w = probs.shape
    obs = np.where(np.isfinite(observed_rate),
                   np.stack([(observed_rate >= t) * 100 for t in THRESHOLDS]), 255).astype(np.uint8)
    full = np.concatenate([obs.reshape(1, nt, h, w), probs], axis=0)  # (L+1, T, H, W)
    g = latlon_grid(geo, h, w)
    grid = np.full((nt, nl + 1, g["nlat"], g["nlon"]), 255, np.uint8)
    ins = g["inside"]
    for ti in range(nt):
        for li in range(nl + 1):
            grid[ti, li][ins] = full[li, ti][g["ri"][ins], g["ci"][ins]]
    with gzip.open(out_dir / "data" / "probs.bin.gz", "wb", compresslevel=6) as f:
        f.write(grid.tobytes())  # orden: [umbral][plazo][lat][lon]
    meta = {
        "issued": issued.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated": generated.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": source, "mode": mode, "members": members, "motion_ok": bool(motion_ok),
        "motion": motion or ("radar" if motion_ok else "ninguno"),
        "thresholds_mmh": list(THRESHOLDS), "leads_min": [0] + list(leads_min),
        "lat_n": float(g["lat_n"]), "lon_w": float(g["lon_w"]), "cell_deg": g["cell"],
        "nlat": g["nlat"], "nlon": g["nlon"], "stale_after_min": stale_after_min,
        "km_per_pixel": geo.km_per_pixel,
    }
    (out_dir / "data" / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta
