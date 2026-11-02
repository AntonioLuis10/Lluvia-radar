"""Verificación continua contra el propio radar (no contra pluviómetros).

Cada pronóstico guarda sus probabilidades en los plazos VERIFY_LEADS. Cuando llega un frame
observado que coincide con (emisión + plazo), se acumulan tablas de contingencia y puntuación
de Brier, y se compara con la persistencia (suponer que sigue como en la emisión).
"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from engine import THRESHOLDS

VERIFY_LEADS = (10, 30, 60, 120)
KEEP_MIN = max(VERIFY_LEADS) + 30


def _fname(t):
    return f"fc_{t:%Y%m%dT%H%M}.npz"


def save_forecast(state_dir, issue_time, probs, leads_min, observed_rate):
    """Guarda sólo los plazos verificables. probs: (L, T, H, W)."""
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    idx = [i for i, m in enumerate(leads_min) if m in VERIFY_LEADS]
    if not idx:
        return
    persist = np.stack([np.where(np.isfinite(observed_rate), observed_rate >= t, False)
                        for t in THRESHOLDS])
    np.savez_compressed(state_dir / _fname(issue_time),
                        probs=probs[idx], leads=np.array([leads_min[i] for i in idx]),
                        persist=persist, issue=issue_time.strftime("%Y%m%dT%H%M"))


def _empty():
    return {"hits": 0, "misses": 0, "false_alarms": 0, "correct_neg": 0,
            "brier_sum": 0.0, "brier_persist_sum": 0.0, "n": 0}


def load_stats(path):
    path = Path(path)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {"thresholds_mmh": list(THRESHOLDS), "leads_min": list(VERIFY_LEADS),
            "table": {str(l): {str(t): _empty() for t in THRESHOLDS} for l in VERIFY_LEADS},
            "done": [], "since": None, "updated": None}


def score(state_dir, stats_path, obs_time, observed_rate, timestep_min):
    """Puntúa todos los pronósticos guardados cuyo (emisión + plazo) coincide con obs_time."""
    state_dir = Path(state_dir)
    stats = load_stats(stats_path)
    tol = timedelta(minutes=timestep_min / 2.0 + 0.01)
    done = set(stats["done"])
    scored = 0
    for f in sorted(state_dir.glob("fc_*.npz")):
        try:
            z = np.load(f)
            issue = datetime.strptime(str(z["issue"]), "%Y%m%dT%H%M").replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if obs_time - issue > timedelta(minutes=KEEP_MIN):
            f.unlink(missing_ok=True)
            continue
        for li, lead in enumerate(z["leads"]):
            lead = int(lead)
            key = f"{issue:%Y%m%dT%H%M}|{lead}"
            if key in done or abs((issue + timedelta(minutes=lead)) - obs_time) > tol:
                continue
            for ti, thr in enumerate(THRESHOLDS):
                p = z["probs"][li, ti]
                valid = (p != 255) & np.isfinite(observed_rate)
                if valid.sum() == 0:
                    continue
                o = (observed_rate >= thr)[valid]
                pf = p[valid].astype(np.float32) / 100.0
                pp = z["persist"][ti][valid]
                yes = pf >= 0.5
                c = stats["table"][str(lead)][str(thr)]
                c["hits"] += int((yes & o).sum())
                c["misses"] += int((~yes & o).sum())
                c["false_alarms"] += int((yes & ~o).sum())
                c["correct_neg"] += int((~yes & ~o).sum())
                c["brier_sum"] += float(((pf - o) ** 2).sum())
                c["brier_persist_sum"] += float(((pp.astype(np.float32) - o) ** 2).sum())
                c["n"] += int(valid.sum())
            done.add(key)
            scored += 1
    cutoff = obs_time - timedelta(hours=6)
    stats["done"] = sorted(k for k in done
                           if datetime.strptime(k.split("|")[0], "%Y%m%dT%H%M").replace(
                               tzinfo=timezone.utc) > cutoff)
    stats["updated"] = obs_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    if stats["since"] is None:
        stats["since"] = stats["updated"]
    Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
    Path(stats_path).write_text(json.dumps(stats))
    return scored
