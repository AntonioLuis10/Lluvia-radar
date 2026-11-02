"""Ejecuta un ciclo: descarga -> predicción -> exportación -> verificación.

Uso:  python run.py --out site [--source auto|opera|rainviewer] [--no-hybrid]

Modo mixto (por defecto): OPERA donde tiene datos y RainViewer en los huecos de OPERA
(p. ej. costa de Murcia/Almería oriental). Para alinear ambos en el tiempo se trabaja
con pasos de 10 min (el ritmo de RainViewer).
"""
import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

import engine, export, sources, verify
from georef import Frame, crop_frame, downsample, target_downsample

log = logging.getLogger("nowcast")

# Ventana por defecto: península ibérica y Baleares (576 px de 2 km ~ 1150 km)
DEFAULT_CENTER = (39.8, -2.5)
BBOX_RV = (-12.0, 34.0, 5.5, 45.0)
TOL = timedelta(seconds=150)


def pick_regular(frames, max_frames=3):
    """Últimos frames con espaciado regular (el de los dos más recientes)."""
    frames = sorted(frames, key=lambda f: f.time)
    if len(frames) < 2:
        return frames
    step = frames[-1].time - frames[-2].time
    out = [frames[-1], frames[-2]]
    for f in reversed(frames[:-2]):
        if len(out) >= max_frames:
            break
        if abs((out[-1].time - f.time) - step) <= timedelta(seconds=90):
            out.append(f)
        else:
            break
    return sorted(out, key=lambda f: f.time)


def prepare(frames, center, size):
    """Elige frames regulares, recorta la ventana y reduce a ~2 km."""
    frames = pick_regular(frames)
    if not frames:
        return frames
    ds = target_downsample(frames[-1])
    return [downsample(crop_frame(f, center[0], center[1], size * ds), ds) for f in frames]


def fuse(opera: Frame, rv: Frame) -> Frame:
    """Rellena los píxeles sin dato de OPERA con el valor de RainViewer en esa posición."""
    holes = ~np.isfinite(opera.dbz)
    if not holes.any():
        return opera
    rr, cc = np.nonzero(holes)
    lon, lat = opera.geo.pixel_to_lonlat(cc, rr)
    c, r = rv.geo.lonlat_to_pixel(lon, lat)
    ci, ri = np.round(c).astype(int), np.round(r).astype(int)
    hh, ww = rv.dbz.shape
    ok = (ci >= 0) & (ci < ww) & (ri >= 0) & (ri < hh)
    vals = np.full(len(rr), np.nan, np.float32)
    vals[ok] = rv.dbz[ri[ok], ci[ok]]
    out = opera.dbz.copy()
    out[rr, cc] = vals
    return Frame(opera.time, out, opera.geo, "OPERA+RainViewer")


def process(frames, out_dir, center, size=576, members=20, horizon=120, now=None, workers=2,
            fill=None):
    now = now or datetime.now(timezone.utc)
    frames = prepare(frames, center, size)
    if fill:
        fused = []
        for f in frames:
            m = min(fill, key=lambda x: abs(x.time - f.time))
            fused.append(fuse(f, m) if abs(m.time - f.time) <= TOL else f)
        frames = fused
    if len(frames) < 2:
        raise RuntimeError("Hacen falta al menos 2 frames consecutivos.")
    step = int(round((frames[-1].time - frames[-2].time).total_seconds() / 60.0))
    if step <= 0 or horizon % step:
        raise RuntimeError(f"Paso temporal no válido: {step} min")
    geo = frames[-1].geo
    rates = np.stack([engine.dbz_to_rate(f.dbz) for f in frames])
    cover = float(np.isfinite(rates[-1]).mean())
    res = engine.forecast_probabilities(rates, step, geo.km_per_pixel, horizon, members,
                                        num_workers=workers)
    t_obs = frames[-1].time
    meta = export.export(out_dir, geo, res["probs"], rates[-1], res["leads_min"], t_obs, now,
                         frames[-1].source, res["mode"], members, res["motion_ok"])
    meta["coverage"] = cover
    verify.score(f"{out_dir}/state", f"{out_dir}/data/stats.json", t_obs, rates[-1], step)
    verify.save_forecast(f"{out_dir}/state", t_obs, res["probs"], res["leads_min"], rates[-1])
    return meta


def window_coverage(frames, center, size):
    f = prepare(frames[-1:], center, size)[0]
    return float(np.isfinite(f.dbz).mean())


def _age_min(t):
    return (datetime.now(timezone.utc) - t).total_seconds() / 60.0


def acquire(a, center):
    """-> (frames, fill, motivos). fill = frames de RainViewer para rellenar huecos de OPERA."""
    session = sources._session()
    why, keys, rv = [], [], None
    if a.source in ("auto", "opera"):
        try:
            keys = sources.opera_latest_keys(session, 24)
            if not keys:
                raise RuntimeError("sin ficheros")
            if _age_min(keys[-1][0]) > a.max_age:
                raise RuntimeError(f"último frame con {_age_min(keys[-1][0]):.0f} min de retraso")
        except Exception as e:
            why.append(f"OPERA: {e}")
            keys = []
    if a.source in ("auto", "rainviewer") and (not keys or not a.no_hybrid):
        try:
            rv = sources.load_rainviewer(BBOX_RV, n=4, session=session)
            if _age_min(rv[-1].time) > a.max_age:
                raise RuntimeError(f"último frame con {_age_min(rv[-1].time):.0f} min de retraso")
        except Exception as e:
            why.append(f"RainViewer: {e}")
            rv = None
    frames, fill = None, None
    if keys:
        sel = keys[-4:]
        if rv:
            rvt = [f.time for f in rv]
            common = [(t, k) for t, k in keys if any(abs(t - x) <= TOL for x in rvt)]
            if len(common) >= 3:
                sel, fill = common[-3:], rv
            else:
                why.append("OPERA y RainViewer sin instantes comunes: no se rellenan huecos")
        try:
            frames = sources.load_opera_keys(session, sel)
            cov = window_coverage(frames, center, a.size)
            log.info("OPERA: %.0f %% de la ventana con dato; último frame hace %.0f min",
                     cov * 100, _age_min(frames[-1].time))
            if cov < a.min_coverage:
                why.append(f"OPERA cubre sólo {cov:.0%} de la ventana")
                frames, fill = None, None
        except Exception as e:
            why.append(f"OPERA: {e}")
            frames, fill = None, None
    if frames is None and rv:
        frames = rv
    return frames, fill, why


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site")
    ap.add_argument("--source", default="auto", choices=["auto", "opera", "rainviewer"])
    ap.add_argument("--no-hybrid", action="store_true", help="no rellenar huecos de OPERA con RainViewer")
    ap.add_argument("--lat", type=float, default=DEFAULT_CENTER[0])
    ap.add_argument("--lon", type=float, default=DEFAULT_CENTER[1])
    ap.add_argument("--size", type=int, default=576)
    ap.add_argument("--members", type=int, default=20)
    ap.add_argument("--max-age", type=int, default=45, help="minutos máx. del último frame")
    ap.add_argument("--min-coverage", type=float, default=0.15)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    center = (a.lat, a.lon)
    frames, fill, why = acquire(a, center)
    if frames is None:
        log.error("Sin datos utilizables: %s", "; ".join(why))
        return 1
    if why:
        log.warning("Avisos: %s", "; ".join(why))
    log.info("Fuente: %s%s", frames[-1].source, " + RainViewer en huecos" if fill else "")
    meta = process(frames, a.out, center, a.size, a.members, fill=fill)
    log.info("Listo: %s", meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
