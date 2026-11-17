"""Motor de predicción: movimiento (Lucas-Kanade) + conjunto STEPS de pySTEPS.

Salida: probabilidad (0-100 %) de superar cada umbral de intensidad, por plazo y píxel.
"""
import logging

import numpy as np
from scipy.ndimage import shift as _shift

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
        # LK devuelve un campo todo ceros cuando no encuentra rasgos que seguir: eso es un fallo
        if v is not None and np.all(np.isfinite(v)) and float(np.abs(v).max()) > 1e-6:
            return v.astype(np.float32), True
    except Exception as e:  # pragma: no cover - depende de los datos
        log.warning("Lucas-Kanade falló: %s", e)
    return np.zeros((2,) + db_stack.shape[1:], np.float32), False


def blend_displacement(lead_min, tau0):
    """Desplazamiento extra (en minutos de viento) al pasar de radar puro a la mezcla
    V(t) = w*Vradar + (1-w)*Vviento con w = exp(-t/tau0): integral de (1-w) = t - tau0*(1-exp(-t/tau0))."""
    return lead_min - tau0 * (1.0 - np.exp(-lead_min / tau0))


def blend_probs(probs, V_radar_px, V_wind_px, step_min, leads, tau0):
    """Traslada cada campo de probabilidad la diferencia entre mezcla y radar puro.
    V_*: (vx, vy) px/paso. Las celdas sin cobertura (255) se conservan."""
    out = probs.copy()
    dv = (np.asarray(V_wind_px) - np.asarray(V_radar_px)) / float(step_min)  # px/min
    for k, lead in enumerate(leads):
        d = blend_displacement(lead, tau0)
        dx, dy = dv[0] * d, dv[1] * d
        if abs(dx) < 0.05 and abs(dy) < 0.05:
            continue
        for i in range(probs.shape[1]):
            f = probs[k, i]
            nocov = f == 255
            g = _shift(np.where(nocov, 0, f).astype(np.float32), (dy, dx), order=1, mode="constant", cval=0.0)
            g = np.round(g).astype(np.uint8)
            g[nocov] = 255
            out[k, i] = g
    return out


def forecast_probabilities(rate_stack, timestep_min, km_per_pixel, horizon_min=120,
                           n_members=20, seed=42, num_workers=2, ambient_fn=None, steering_fn=None, tau0=60.0):
    """rate_stack: (T, H, W) mm/h (nan fuera de cobertura), el último es el más reciente.

    Devuelve dict con probs (L, nThr, H, W) uint8 (255 = sin cobertura), leads_min, mode, motion_ok.
    """
    n_leads = int(horizon_min // timestep_min)
    last = rate_stack[-1]
    coverage = np.isfinite(last)
    h, w = last.shape
    probs = np.zeros((n_leads, len(THRESHOLDS), h, w), np.uint8)
    leads = [timestep_min * (k + 1) for k in range(n_leads)]
    out = {"leads_min": leads, "mode": "sin-lluvia", "motion_ok": False, "motion": "ninguno"}

    wet = np.nansum(last >= RAIN_MIN)
    if wet < 25:  # prácticamente nada de lluvia: no hay qué extrapolar
        probs[:, :, ~coverage] = 255
        out["probs"] = probs
        return out

    db = np.stack([to_db(r) for r in rate_stack[-3:]])
    ar_order = db.shape[0] - 1
    V, ok = estimate_motion(db)
    out["motion_ok"] = ok
    out["motion"] = "radar" if ok else "ninguno"
    if not ok and ambient_fn is not None:
        try:  # poca lluvia para medir el movimiento: se usa el viento a 500-700 hPa
            u, v_north = ambient_fn()
            px = 60.0 * timestep_min / (km_per_pixel * 1000.0)
            V = np.zeros((2,) + db.shape[1:], np.float32)
            V[0], V[1] = u * px, -v_north * px   # x hacia el este, y (filas) hacia el sur
            out["motion"] = "viento"
        except Exception as e:
            log.warning("Sin viento ambiente de respaldo: %s", e)

    steer_px = None
    if steering_fn is not None:
        try:
            st = steering_fn()
            strong = np.nanmax(last) >= dbz_to_rate(np.float32(40.0))
            u, v_n = (st["bunkers"] if (strong and st.get("bunkers")) else st["mean"])
            pxs = 60.0 * timestep_min / (km_per_pixel * 1000.0)
            steer_px = (u * pxs, -v_n * pxs)
            out["steering"] = {"u": round(float(u), 1), "v": round(float(v_n), 1),
                               "bunkers": bool(strong and st.get("bunkers")),
                               "shear": st.get("shear")}
        except Exception as e:
            log.warning("Sin viento director: %s", e)
    if not ok and steer_px is not None and out["motion"] == "ninguno":
        V = np.zeros((2,) + db.shape[1:], np.float32)
        V[0], V[1] = steer_px
        out["motion"] = "viento"

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
    out["probs_blend"] = None
    if steer_px is not None and out["motion"] == "radar":
        wetm = last >= RAIN_MIN
        wetm = wetm & np.isfinite(last)
        if wetm.sum() >= 25:
            Vr = (float(np.median(V[0][wetm])), float(np.median(V[1][wetm])))
            out["probs_blend"] = blend_probs(probs, Vr, steer_px, timestep_min, leads, tau0)
            out["tau0"] = tau0
    return out
