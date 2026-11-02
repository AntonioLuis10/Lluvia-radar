"""Diagnóstico de fuentes: ¿hay datos de radar utilizables en España, con qué retraso y formato?

Uso:  python diagnose.py        (escribe también en $GITHUB_STEP_SUMMARY si existe)
"""
import os
from datetime import datetime, timezone

import numpy as np

import sources
from georef import crop_frame

CITIES = {
    "Almería": (36.84, -2.46, "A"), "Mar de Pulpí": (37.40, -1.74, "P"), "Valencia": (39.47, -0.38, "V"),
    "Madrid": (40.42, -3.70, "M"), "Barcelona": (41.39, 2.17, "B"), "Sevilla": (37.39, -5.98, "S"),
    "A Coruña": (43.36, -8.41, "C"), "Palma": (39.57, 2.65, "L"),
}


def zone_stats(frame, lat, lon, half_km):
    half = max(1, int(half_km / frame.geo.km_per_pixel))
    f = crop_frame(frame, lat, lon, 2 * half + 1).dbz
    n = f.size
    return (np.isnan(f).sum() / n, (f == -32.0).sum() / n, ((f > -32.0) & (f < 15)).sum() / n,
            (f >= 15).sum() / n)


def point_state(frame, lat, lon):
    c, r = frame.geo.lonlat_to_pixel(lon, lat)
    ci, ri = int(round(float(c))), int(round(float(r)))
    h, w = frame.dbz.shape
    if not (0 <= ri < h and 0 <= ci < w):
        return "fuera de la rejilla"
    v = frame.dbz[ri, ci]
    if np.isnan(v):
        return "SIN DATO"
    return "sin eco" if v == -32.0 else f"eco {v:.0f} dBZ"


def ascii_map(frame, lon0=-10.0, lon1=4.6, lat0=35.0, lat1=44.4, cols=96, rows=36):
    """Mapa de cobertura: ' ' sin dato, '.' sin eco, '+' eco débil, '#' eco >= 15 dBZ, letras = ciudades."""
    lons, lats = np.linspace(lon0, lon1, cols), np.linspace(lat1, lat0, rows)
    LON, LAT = np.meshgrid(lons, lats)
    c, r = frame.geo.lonlat_to_pixel(LON, LAT)
    ci, ri = np.round(c).astype(int), np.round(r).astype(int)
    h, w = frame.dbz.shape
    ins = (ci >= 0) & (ci < w) & (ri >= 0) & (ri < h)
    v = np.full(LON.shape, np.nan, np.float32)
    v[ins] = frame.dbz[ri[ins], ci[ins]]
    g = np.full(LON.shape, " ", dtype="<U1")
    g[np.isfinite(v) & (v == -32.0)] = "."
    g[np.isfinite(v) & (v > -32.0) & (v < 15)] = "+"
    g[np.isfinite(v) & (v >= 15)] = "#"
    for _, (la, lo, ch) in CITIES.items():
        i = int(round((lat1 - la) / (lat1 - lat0) * (rows - 1)))
        j = int(round((lo - lon0) / (lon1 - lon0) * (cols - 1)))
        if 0 <= i < rows and 0 <= j < cols:
            g[i, j] = ch
    return "\n".join("".join(row) for row in g)


def sample(frame, LON, LAT):
    c, r = frame.geo.lonlat_to_pixel(LON, LAT)
    ci, ri = np.round(c).astype(int), np.round(r).astype(int)
    h, w = frame.dbz.shape
    ins = (ci >= 0) & (ci < w) & (ri >= 0) & (ri < h)
    v = np.full(LON.shape, np.nan, np.float32)
    v[ins] = frame.dbz[ri[ins], ci[ins]]
    return v


def rain_map(v, cols, rows, marks):
    """' ' sin dato, '.' sin lluvia, '+' >= 23 dBZ (~0,5 mm/h), '#' >= 35 dBZ, X = punto de referencia."""
    g = np.full(v.shape, " ", dtype="<U1")
    g[np.isfinite(v)] = "."
    g[np.isfinite(v) & (v >= 23)] = "+"
    g[np.isfinite(v) & (v >= 35)] = "#"
    for i, j in marks:
        g[i, j] = "X"
    return "\n".join("".join(r) for r in g)


def compare_sources(op, rv):
    """Misma región, misma hora aproximada: ¿ven lo mismo OPERA y RainViewer?"""
    lon0, lon1, lat0, lat1 = -9.5, 3.3, 36.0, 43.8
    cols, rows = 100, 36
    lons, lats = np.linspace(lon0, lon1, cols), np.linspace(lat1, lat0, rows)
    LON, LAT = np.meshgrid(lons, lats)
    fine_lon, fine_lat = np.meshgrid(np.arange(lon0, lon1, 0.05), np.arange(lat0, lat1, 0.05))
    out = ["", "## Comparación OPERA vs RainViewer sobre la península",
           f"- OPERA: frame de las {op.time:%H:%M} UTC · RainViewer: frame de las {rv.time:%H:%M} UTC"]
    stats = {}
    for name, fr in (("OPERA", op), ("RainViewer", rv)):
        v = sample(fr, fine_lon, fine_lat)
        stats[name] = (int(np.isfinite(v).sum()), int((v >= 23).sum()), int((v >= 35).sum()))
        out.append(f"- {name}: {stats[name][0]} puntos con dato, {stats[name][1]} con lluvia (≥ 23 dBZ), "
                   f"{stats[name][2]} con lluvia fuerte (≥ 35 dBZ)")
    marks = [(int(round((lat1 - la) / (lat1 - lat0) * (rows - 1))), int(round((lo - lon0) / (lon1 - lon0) * (cols - 1))))
             for la, lo in ((41.88, -5.05), (37.40, -1.74))]
    for name, fr in (("OPERA", op), ("RainViewer", rv)):
        out += ["", f"{name} (X = Medina de Rioseco y Mar de Pulpí):", "```",
                rain_map(sample(fr, LON, LAT), cols, rows, marks), "```"]
    if stats["RainViewer"][1] > 3 * max(stats["OPERA"][1], 10):
        out += ["", "**Conclusión: RainViewer ve mucha más lluvia que OPERA sobre España.** "
                "El composite OPERA no está aportando los radares españoles."]
    return out


def main():
    lines = ["# Diagnóstico de fuentes de radar", ""]
    op_fr = rv_fr = None
    now = datetime.now(timezone.utc)
    lines += ["## OPERA (composite europeo, S3 anónimo)"]
    try:
        keys = sources.opera_latest_keys(sources._session(), 4)
        if not keys:
            lines.append("- No hay ficheros DBZH de composite en las últimas 48 h.")
        else:
            lag = (now - keys[-1][0]).total_seconds() / 60
            sp = [(b[0] - a[0]).total_seconds() / 60 for a, b in zip(keys[:-1], keys[1:])]
            lines.append(f"- Último frame: {keys[-1][0]:%Y-%m-%d %H:%M} UTC (hace {lag:.0f} min). "
                         f"Espaciado: {sp} min")
            fr = sources.load_opera(1)[-1]
            op_fr = fr
            lines.append(f"- Rejilla: {fr.dbz.shape}, {fr.geo.km_per_pixel:.2f} km/px")
            lines += ["", "| Lugar | en el punto | ±10 km sin dato | ±50 km sin dato | ±50 km sin eco | ±50 km eco |",
                      "|---|---|---|---|---|---|"]
            for name, (la, lo, _) in CITIES.items():
                n10 = zone_stats(fr, la, lo, 10)[0]
                nd, ne, wk, st = zone_stats(fr, la, lo, 50)
                lines.append(f"| {name} | {point_state(fr, la, lo)} | {n10:.0%} | {nd:.0%} | {ne:.0%} | {wk + st:.0%} |")
            lines += ["", "Mapa de cobertura (' ' sin dato, '.' sin eco, '+' eco débil, '#' eco ≥ 15 dBZ; "
                      "A Almería, P Pulpí, V Valencia, M Madrid, B Barcelona, S Sevilla, C A Coruña, L Palma):",
                      "", "```", ascii_map(fr), "```"]
    except Exception as e:
        lines.append(f"- FALLO: `{e}`")
    lines += ["", "## RainViewer (respaldo)"]
    try:
        s = sources._session()
        j = s.get(sources.RV_API, timeout=sources.TIMEOUT).json()
        past = j["radar"]["past"]
        lag = (now.timestamp() - past[-1]["time"]) / 60
        lines.append(f"- {len(past)} frames; último hace {lag:.0f} min; "
                     f"espaciado {(past[-1]['time'] - past[-2]['time']) / 60:.0f} min")
        fr = sources.load_rainviewer((-12.0, 34.0, 5.5, 45.0), n=1)[-1]
        rv_fr = fr
        lines.append(f"- Mosaico: {fr.dbz.shape}, {fr.geo.km_per_pixel:.2f} km/px (Mercator), "
                     f"{np.isnan(fr.dbz).mean():.0%} sin dato")
    except Exception as e:
        lines.append(f"- FALLO: `{e}`")
    if op_fr is not None and rv_fr is not None:
        try:
            lines += compare_sources(op_fr, rv_fr)
        except Exception as e:
            lines.append(f"- Comparación fallida: `{e}`")
    out = "\n".join(lines)
    print(out)
    p = os.environ.get("GITHUB_STEP_SUMMARY")
    if p:
        with open(p, "a") as f:
            f.write(out + "\n")


if __name__ == "__main__":
    main()
