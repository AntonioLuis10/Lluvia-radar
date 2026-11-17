import gzip
import io
import json
from datetime import datetime, timedelta, timezone

import h5py
import numpy as np
import pytest

import engine, run, sources, verify
from georef import Frame, Georef, crop_frame

LAEA = "+proj=laea +lat_0=55 +lon_0=10 +x_0=1950000 +y_0=-2100000 +ellps=WGS84 +units=m +no_defs"
T0 = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)


def make_geo(n=700, km=2.0):
    # esquina superior izquierda cerca del NO de Iberia, en proyección LAEA
    from pyproj import Transformer
    x0, y0 = Transformer.from_crs(4326, LAEA, always_xy=True).transform(-12.0, 46.0)
    return Georef(LAEA, x0, y0, km * 1000, km * 1000)


def blobs(shape, shift_px, seed=1):
    """Células gaussianas que se desplazan hacia el este."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    field = np.full(shape, -32.0, np.float32)
    for _ in range(14):
        cy, cx = rng.uniform(150, shape[0] - 150), rng.uniform(100, shape[1] - 250)
        s, a = rng.uniform(8, 22), rng.uniform(30, 50)
        d = a * np.exp(-((yy - cy) ** 2 + (xx - cx - shift_px) ** 2) / (2 * s ** 2)) - 5
        field = np.maximum(field, d)
    field[field < 5] = -32.0
    return field


def synth_frames(n_frames=4, step_min=5, px_per_step=2.0, t_end=T0):
    geo = make_geo()
    out = []
    for k in range(n_frames):
        shift = px_per_step * (k - (n_frames - 1))
        t = t_end - timedelta(minutes=step_min * (n_frames - 1 - k))
        out.append(Frame(t, blobs((700, 700), shift), geo, "TEST"))
    return out, geo


def test_odim_reader(tmp_path):
    raw = np.array([[0, 255, 100], [150, 60, 0]], np.uint8)
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        f.create_group("what").attrs.update({"date": b"20261101", "time": b"120000", "object": b"COMP"})
        f.create_group("where").attrs.update({
            "projdef": LAEA.encode(), "xsize": 3, "ysize": 2, "xscale": 2000.0, "yscale": 2000.0,
            "UL_lon": -12.0, "UL_lat": 46.0})
        d = f.create_group("dataset1/data1")
        d.create_dataset("data", data=raw)
        d.create_group("what").attrs.update({"gain": 0.5, "offset": -32.0, "nodata": 255.0,
                                             "undetect": 0.0, "quantity": b"DBZH"})
    fr = sources.read_odim_composite(buf.getvalue())
    assert fr.time == T0
    assert fr.dbz[0, 0] == -32.0 and np.isnan(fr.dbz[0, 1])
    assert fr.dbz[0, 2] == pytest.approx(18.0) and fr.dbz[1, 0] == pytest.approx(43.0)
    lon, lat = fr.geo.pixel_to_lonlat(-0.5, -0.5)  # esquina exterior superior izquierda
    assert lon == pytest.approx(-12.0, abs=1e-4) and lat == pytest.approx(46.0, abs=1e-4)


def test_s3_listing_and_key_time():
    xml = """<?xml version="1.0"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
    <IsTruncated>false</IsTruncated>
    <Contents><Key>2026/11/01/OPERA/COMP/OPERA@20261101T1200@0@DBZH.h5</Key></Contents>
    <Contents><Key>2026/11/01/OPERA/COMP/OPERA@20261101T1200@0@RATE.h5</Key></Contents>
    <Contents><Key>2026/11/01/OPERA/COMP/OPERA@20261101T1155@0@DBZH.h5</Key></Contents></ListBucketResult>"""

    class R:
        content = xml.encode()
        def raise_for_status(self): pass

    class S:
        def get(self, *a, **k): return R()

    keys = sources.s3_list(S(), "2026/11/01/OPERA/COMP/")
    assert len(keys) == 3
    got = sources.opera_latest_keys(S(), 4, now=T0)
    assert [t.strftime("%H%M") for t, _ in got] == ["1155", "1200"]
    assert sources.parse_key_time(keys[1]) is None  # RATE se ignora


def test_rainviewer_palette_decoding():
    rgba = np.zeros((2, 3, 4), np.uint8)
    rgba[0, 0] = (255, 197, 0, 255)   # fila 70 -> 38 dBZ
    rgba[0, 1] = (255, 68, 0, 255)    # fila 77 -> 45 dBZ
    out = sources.rgba_to_dbz(rgba)
    assert out[0, 0] == 38 and out[0, 1] == 45 and out[0, 2] == -32.0


def test_rate_conversion():
    r = engine.dbz_to_rate(np.array([-32, 20, 30, 40, 70, np.nan], np.float32))
    assert r[0] == 0 and 0.5 < r[1] < 0.8 and 2.4 < r[2] < 3.0 and 10 < r[3] < 13
    assert r[4] == pytest.approx(engine.dbz_to_rate(np.array([55.0]))[0])  # tope de granizo
    assert np.isnan(r[5])


def test_forecast_moves_rain_east(tmp_path):
    frames, geo = synth_frames()
    center = (39.5, -3.0)
    meta = run.process(frames, tmp_path, center, size=512, members=10, workers=2)
    assert meta["mode"].startswith("STEPS") and meta["motion_ok"]
    # exportación: leer el binario y comprobar que la lluvia avanza hacia el este con el plazo
    grid = np.frombuffer(gzip.decompress((tmp_path / "data/probs.bin.gz").read_bytes()), np.uint8)
    grid = grid.reshape(3, len(meta["leads_min"]), meta["nlat"], meta["nlon"])
    cols = np.arange(meta["nlon"])

    def centroid(li):
        p = np.where(grid[0, li] == 255, 0, grid[0, li]).astype(float)
        return (p.sum(0) * cols).sum() / p.sum()

    c0, c60, c120 = centroid(0), centroid(12), centroid(24)
    assert c0 < c60 < c120
    # desplazamiento esperado: 2 px/5 min * 2 km = 0,8 km/min -> 96 km en 2 h ≈ 2,2° ≈ 54 celdas de 0,04°
    assert 30 < (c120 - c0) < 80, (c0, c120)
    # las probabilidades son coherentes (0-100 o 255)
    assert grid.max() == 255 and np.all((grid <= 100) | (grid == 255))
    assert meta["lat_n"] > 36 and meta["lon_w"] < -8
    json.loads((tmp_path / "data/meta.json").read_text())


def test_no_rain_gives_zero_probability(tmp_path):
    geo = make_geo()
    frames = [Frame(T0 - timedelta(minutes=5 * (3 - k)), np.full((700, 700), -32.0, np.float32), geo, "TEST")
              for k in range(4)]
    meta = run.process(frames, tmp_path, (39.5, -3.0), size=512, members=5)
    grid = np.frombuffer(gzip.decompress((tmp_path / "data/probs.bin.gz").read_bytes()), np.uint8)
    valid = grid[grid != 255]
    assert meta["mode"] == "sin-lluvia" and valid.max() == 0


def test_verification_accumulates(tmp_path):
    center = (39.5, -3.0)
    # emisiones cada 10 min; la "realidad" sigue el mismo movimiento
    for i in range(0, 8):
        t_end = T0 + timedelta(minutes=10 * i)
        frames, _ = synth_frames(t_end=t_end, px_per_step=2.0)
        # el campo real avanza: desplazar la fase según el tiempo transcurrido
        shift = 2.0 * (10 * i / 5)
        geo = frames[0].geo
        frames = [Frame(f.time, blobs((700, 700), 2.0 * (k - 3) + shift), geo, "TEST")
                  for k, f in enumerate(frames)]
        run.process(frames, tmp_path, center, size=512, members=6)
    stats = json.loads((tmp_path / "data/stats.json").read_text())
    c10 = stats["table"]["10"]["0.5"]
    assert c10["n"] > 0 and c10["hits"] > 0
    skill = 1 - c10["brier_sum"] / c10["brier_persist_sum"]
    assert skill > -0.5  # no destroza a la persistencia en un caso de traslación pura
    assert len(stats["done"]) >= 1


def test_crop_outside_domain_is_nan():
    frames, geo = synth_frames(1)
    f = crop_frame(frames[0], 30.0, -3.0, 256)  # sur del dominio
    assert np.isnan(f.dbz).any()


def test_downsample_keeps_nan_and_averages_linearly():
    from georef import downsample
    geo = Georef(LAEA, 0.0, 0.0, 1000.0, 1000.0)
    a = np.full((4, 4), -32.0, np.float32)
    a[0, 0] = 30.0                      # un píxel con eco en un bloque 2x2
    a[2:, 2:] = np.nan                  # bloque sin dato
    a[0, 2] = np.nan                    # bloque con 1 de 4 válidos -> nan
    a[1, 2] = np.nan
    a[1, 3] = np.nan
    out = downsample(Frame(T0, a, geo, "TEST"), 2)
    assert out.dbz.shape == (2, 2) and out.geo.dx == 2000.0
    assert out.dbz[0, 0] == pytest.approx(10 * np.log10(10 ** 3.0 / 4), abs=0.05)  # media lineal
    assert np.isnan(out.dbz[1, 1]) and np.isnan(out.dbz[0, 1])


def test_process_with_1km_grid(tmp_path):
    geo = make_geo(km=1.0)
    n = 1400
    frames = [Frame(T0 - timedelta(minutes=5 * (3 - k)), blobs((n, n), 4.0 * (k - 3), seed=3), geo, "TEST")
              for k in range(4)]
    meta = run.process(frames, tmp_path, (39.8, -2.5), size=256, members=5)
    assert meta["km_per_pixel"] == pytest.approx(2.0) and meta["mode"].startswith("STEPS")


def _rv_frame(t, spot_lonlat=None):
    from pyproj import Transformer
    from sources import MERC
    x0, y0 = Transformer.from_crs(4326, MERC, always_xy=True).transform(-12.0, 45.0)
    res = 40075016.685578488 / (256 * 2 ** 6)
    geo = Georef(MERC, x0, y0, res, res)
    a = np.full((900, 900), -32.0, np.float32)
    if spot_lonlat:
        c, r = geo.lonlat_to_pixel(*spot_lonlat)
        a[int(r) - 3:int(r) + 4, int(c) - 3:int(c) + 4] = 30.0
    return Frame(t, a, geo, "RainViewer")


def test_fuse_fills_holes_only():
    from run import fuse
    geo = make_geo(km=2.0)
    spot = (-1.74, 37.40)  # Mar de Pulpí
    c, r = geo.lonlat_to_pixel(*spot)
    ci, ri = int(round(float(c))), int(round(float(r)))
    a = np.full((700, 700), -32.0, np.float32)
    a[ri - 4:ri + 5, ci - 4:ci + 5] = np.nan          # hueco de OPERA
    a[ri + 20, ci + 20] = 40.0                          # dato OPERA que no debe tocarse
    out = fuse(Frame(T0, a, geo, "OPERA"), _rv_frame(T0, spot))
    assert out.dbz[ri, ci] == 30.0 and out.source == "OPERA+RainViewer"
    assert out.dbz[ri + 20, ci + 20] == 40.0
    assert np.isfinite(out.dbz[ri - 4:ri + 5, ci - 4:ci + 5]).all()


def test_acquire_hybrid_and_process(tmp_path, monkeypatch):
    import types
    geo = make_geo(km=2.0)
    keys = [(T0 - timedelta(minutes=5 * k), f"k{k}") for k in range(8, -1, -1)]
    rv = [_rv_frame(T0 - timedelta(minutes=10 * k)) for k in (3, 2, 1, 0)]

    def fake_load(session, sel):
        out = []
        for t, _ in sel:
            minutes = (T0 - t).total_seconds() / 60.0
            f = blobs((700, 700), -2.0 * minutes / 5)
            f[300:380, 100:180] = np.nan  # hueco
            out.append(Frame(t, f, geo, "OPERA"))
        return out

    monkeypatch.setattr(sources, "_session", lambda: None)
    monkeypatch.setattr(sources, "opera_latest_keys", lambda s, n: keys)
    monkeypatch.setattr(sources, "load_opera_keys", fake_load)
    monkeypatch.setattr(sources, "load_rainviewer", lambda *a, **k: rv)
    monkeypatch.setattr(run, "_age_min", lambda t: 3.0)
    args = types.SimpleNamespace(source="hybrid", max_age=45, size=256, min_coverage=0.15)
    frames, fill, why = run.acquire(args, (39.8, -2.5))
    assert fill is rv and [f.time for f in frames] == [T0 - timedelta(minutes=20), T0 - timedelta(minutes=10), T0]
    meta = run.process(frames, tmp_path, (39.8, -2.5), size=256, members=5, fill=fill)
    assert meta["source"] == "OPERA+RainViewer" and meta["leads_min"][1] == 10 and meta["leads_min"][-1] == 120


def test_auto_prefers_rainviewer_and_falls_back_to_opera(monkeypatch):
    import types
    geo = make_geo(km=2.0)
    rv = [_rv_frame(T0 - timedelta(minutes=10 * k), (-5.0, 41.9)) for k in (3, 2, 1, 0)]
    keys = [(T0 - timedelta(minutes=5 * k), f"k{k}") for k in range(4, -1, -1)]
    monkeypatch.setattr(sources, "_session", lambda: None)
    monkeypatch.setattr(sources, "opera_latest_keys", lambda s, n: keys)
    monkeypatch.setattr(sources, "load_opera_keys",
                        lambda s, sel: [Frame(t, blobs((700, 700), 0.0), geo, "OPERA") for t, _ in sel])
    monkeypatch.setattr(run, "_age_min", lambda t: 2.0)
    args = types.SimpleNamespace(source="auto", max_age=45, size=256, min_coverage=0.15)
    monkeypatch.setattr(sources, "load_rainviewer", lambda *a, **k: rv)
    frames, fill, why = run.acquire(args, (39.8, -2.5))
    assert frames[-1].source == "RainViewer" and fill is None and not why

    def boom(*a, **k):
        raise RuntimeError("caído")
    monkeypatch.setattr(sources, "load_rainviewer", boom)
    frames, fill, why = run.acquire(args, (39.8, -2.5))
    assert frames[-1].source == "OPERA" and "rainviewer: caído" in why[0]


def test_ambient_wind_fallback_moves_rain(tmp_path, monkeypatch):
    """Con poca lluvia LK no encuentra movimiento: se usa el viento (oeste a 20 m/s -> hacia el este)."""
    from engine import forecast_probabilities, dbz_to_rate
    h = w = 128
    a = np.full((h, w), -32.0, np.float32)
    a[60:70, 30:40] = 35.0   # una sola célula, sin textura para LK
    rates = np.stack([dbz_to_rate(a)] * 3)
    res = forecast_probabilities(rates, 10, 2.0, 60, 4, ambient_fn=lambda: (20.0, 0.0))
    p = res["probs"][:, 0].astype(float)  # (leads, h, w)
    cx = [(np.where(p[k] == 255, 0, p[k]).sum(0) * np.arange(w)).sum() / max(np.where(p[k] == 255, 0, p[k]).sum(), 1)
          for k in range(p.shape[0])]
    assert res["motion"] in ("viento", "radar") and cx[-1] > cx[0] + 3, (res["motion"], cx)


def test_blend_shifts_with_wind_difference():
    import engine
    p = np.zeros((2, 1, 40, 40), np.uint8)
    p[:, 0, 20, 10] = 100
    p[:, 0, 0, 0] = 255
    out = engine.blend_probs(p, (0.0, 0.0), (1.0, 0.0), 10, [10, 120], 60.0)
    assert out[1, 0, 20, 10] == 0
    f = np.where(out[1, 0] == 255, 0, out[1, 0])
    cols = np.argwhere(f == f.max())[:, 1]
    assert f.max() > 0 and cols.min() > 10          # viento hacia el este desplaza la lluvia al este
    assert out[0, 0, 0, 0] == 255   # sin cobertura se conserva
    assert engine.blend_displacement(120, 60) > engine.blend_displacement(10, 60)


def test_bunkers_and_blend_in_forecast():
    import engine
    rng = np.random.default_rng(1)
    base = np.zeros((90, 90), np.float32)
    base[30:50, 20:40] = 20.0
    stack = np.stack([np.roll(base, 2 * i, axis=1) for i in range(4)])
    res = engine.forecast_probabilities(stack, 10, 2.0, 60, 4, num_workers=1,
                                        steering_fn=lambda: {"mean": (10.0, 0.0), "bunkers": (12.0, -3.0), "shear": 25})
    assert res["probs_blend"] is not None or res["motion"] != "radar"
    assert res["steering"]["bunkers"] is True
