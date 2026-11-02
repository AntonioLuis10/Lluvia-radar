"""Ejecuta un ciclo: descarga -> predicción -> exportación -> verificación.

Uso:  python run.py --out site [--source auto|opera|rainviewer]
"""
import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

import engine, export, sources, verify
from georef import crop_frame, downsample, target_downsample

log = logging.getLogger("nowcast")

# Ventana por defecto: península ibérica y Baleares (576 px de 2 km ~ 1150 km)
DEFAULT_CENTER = (39.8, -2.5)
BBOX_RV = (-12.0, 34.0, 5.5, 45.0)


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


def process(frames, out_dir, center, size=576, members=20, horizon=120, now=None, workers=2):
    now = now or datetime.now(timezone.utc)
    frames = prepare(frames, center, size)
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site")
    ap.add_argument("--source", default="auto", choices=["auto", "opera", "rainviewer"])
    ap.add_argument("--lat", type=float, default=DEFAULT_CENTER[0])
    ap.add_argument("--lon", type=float, default=DEFAULT_CENTER[1])
    ap.add_argument("--size", type=int, default=576)
    ap.add_argument("--members", type=int, default=20)
    ap.add_argument("--max-age", type=int, default=45, help="minutos máx. del último frame")
    ap.add_argument("--min-coverage", type=float, default=0.15)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    center = (a.lat, a.lon)
    frames, why = None, []
    if a.source in ("auto", "opera"):
        try:
            frames = sources.load_opera(4)
            k = max(1, int(round(2.0 / frames[0].geo.km_per_pixel)))
            frames = [sources.downsample(f, k) for f in frames]
            cov = window_coverage(frames, center, a.size)
            age = (datetime.now(timezone.utc) - frames[-1].time).total_seconds() / 60.0
            log.info("OPERA: cobertura de la ventana %.0f %%, último frame hace %.0f min", cov * 100, age)
            if cov < a.min_coverage:
                why.append(f"OPERA cubre sólo {cov:.0%} de la ventana")
                frames = None
            elif age > a.max_age:
                why.append(f"OPERA con {age:.0f} min de retraso")
                frames = None
        except Exception as e:
            why.append(f"OPERA falló: {e}")
            frames = None
    if frames is None and a.source in ("auto", "rainviewer"):
        try:
            frames = sources.load_rainviewer(BBOX_RV)
            age = (datetime.now(timezone.utc) - frames[-1].time).total_seconds() / 60.0
            if age > a.max_age:
                why.append(f"RainViewer con {age:.0f} min de retraso")
                frames = None
        except Exception as e:
            why.append(f"RainViewer falló: {e}")
            frames = None
    if frames is None:
        log.error("Sin datos utilizables: %s", "; ".join(why))
        return 1
    if why:
        log.warning("Se usa %s porque: %s", frames[-1].source, "; ".join(why))
    meta = process(frames, a.out, center, a.size, a.members)
    log.info("Listo: %s", meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
