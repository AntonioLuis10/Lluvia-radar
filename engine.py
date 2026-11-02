"""Motor de predicción: movimiento (Lucas-Kanade) + conjunto STEPS de pySTEPS.

Salida: probabilidad (0-100 %) de superar cada umbral de intensidad, por plazo y píxel.
"""
import logging

import numpy as np

log = logging.getLogger("nowcast")

THRESHOLDS = (0.5, 2.5, 10.0)   # mm/h: lluvia, moderada, fuerte
RAIN_MIN = 0.1                  # mm/h; por debajo, "sin lluvia"
DB_ZERO = -15.0
DB_THR = 10 * np.log10(RAIN_MIN)  # -10 dBR


def dbz_to_rate(dbz):
    """Marshall-Palmer Z = 200 R^1.6, con tope de 55 dBZ (granizo). nan se respeta."""
    z = 10.0 ** (np.clip(dbz, -32.0, 55.0) / 10.0)
    r = (z / 200.0) ** (1.0 / 1.6)
    r = np.where(np.isfinite(dbz), r, np.nan)
    r = np.where(r < RAIN_MIN, 0.0, r)
    return r.astype(np.float32)


def to_db(rate):
    r = np.nan_to_num(rate, nan=0.0)
    return np.where(r >= RAIN_MIN, 10.0 * np.log10(np.maximum(r, 1e-6)), DB_ZERO).astype(np.float32)


def from_db(db):
    r = 10.0 ** (db / 10.0)
    return np.where(db < DB_THR, 0.0, r).astype(np.float32)


def estimate_motion(db_stack):
    """Campo de velocidad (2, H, W) en px/paso. Si no hay con qué, ceros."""
    try:
        from pysteps import motion
        v = motion.get_method("LK")(db_stack)
        if v is not None and np.all(np.isfinite(v)):
            return v.astype(np.float32), True
    except Exception as e:  # pragma: no cover - depende de los datos
        log.warning("Lucas-Kanade falló: %s", e)
    return np.zeros((2,) + db_stack.shape[1:], np.float32), False


def forecast_probabilities(rate_stack, timestep_min, km_per_pixel, horizon_min=120,
                           n_members=20, seed=42, num_workers=2):
    """rate_stack: (T, H, W) mm/h (nan fuera de cobertura), el último es el más reciente.

    Devuelve dict con probs (L, nThr, H, W) uint8 (255 = sin cobertura), leads_min, mode, motion_ok.
    """
    n_leads = int(horizon_min // timestep_min)
    last = rate_stack[-1]
    coverage = np.isfinite(last)
    h, w = last.shape
    probs = np.zeros((n_leads, len(THRESHOLDS), h, w), np.uint8)
    leads = [timestep_min * (k + 1) for k in range(n_leads)]
    out = {"leads_min": leads, "mode": "sin-lluvia", "motion_ok": False}

    wet = np.nansum(last >= RAIN_MIN)
    if wet < 25:  # prácticamente nada de lluvia: no hay qué extrapolar
        probs[:, :, ~coverage] = 255
        out["probs"] = probs
        return out

    db = np.stack([to_db(r) for r in rate_stack[-3:]])
    ar_order = db.shape[0] - 1
    V, ok = estimate_motion(db)
    out["motion_ok"] = ok

    counts = np.zeros((n_leads, len(THRESHOLDS), h, w), np.float32)
    state = {"k": 0}

    def cb(arr):  # arr: (miembros, H, W) en dB, un paso por llamada
        r = from_db(arr)
        k = state["k"]
        for i, thr in enumerate(THRESHOLDS):
            counts[k, i] = (r >= thr).mean(axis=0)
        state["k"] = k + 1

    from pysteps import nowcasts
    done = False
    # 1) STEPS con AR(2); 2) si el ajuste AR no es estacionario, STEPS con AR(1); 3) extrapolación.
    for order in sorted({ar_order, 1}, reverse=True):
        if order > ar_order:
            continue
        state["k"] = 0
        counts[:] = 0
        try:
            nowcasts.get_method("steps")(
                db[-(order + 1):], V, n_leads, n_ens_members=n_members, n_cascade_levels=6,
                precip_thr=DB_THR, kmperpixel=km_per_pixel, timestep=timestep_min,
                ar_order=order, noise_method="nonparametric", vel_pert_method="bps",
                mask_method="incremental", seed=seed, num_workers=num_workers,
                callback=cb, return_output=False,
            )
            if state["k"] != n_leads:
                raise RuntimeError(f"STEPS devolvió {state['k']} de {n_leads} pasos")
            out["mode"] = f"STEPS-{n_members}" + ("" if order == ar_order else f" (AR{order})")
            done = True
            break
        except Exception as e:
            log.warning("STEPS con AR(%d) falló (%s)", order, e)
    if not done:
        log.warning("Uso extrapolación determinista")
        from pysteps.extrapolation.semilagrangian import extrapolate
        fc = extrapolate(db[-1], V, n_leads)
        r = from_db(fc)
        counts[:] = 0
        for k in range(n_leads):
            for i, thr in enumerate(THRESHOLDS):
                counts[k, i] = (r[k] >= thr).astype(np.float32)
        out["mode"] = "extrapolacion"

    probs = np.round(np.clip(counts, 0, 1) * 100).astype(np.uint8)
    probs[:, :, ~coverage] = 255
    out["probs"] = probs
    return out
