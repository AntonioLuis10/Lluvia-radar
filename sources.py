"""Fuentes de radar: composite europeo OPERA (S3 anónimo) y RainViewer (respaldo)."""
import io
import math
import re
import warnings
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import h5py
import numpy as np
import requests
from PIL import Image
from scipy.spatial import cKDTree

from georef import Frame, Georef

TIMEOUT = 40
S3_ENDPOINT = "https://s3.waw3-1.cloudferro.com"
S3_BUCKET = "openradar-24h"
KEY_RE = re.compile(r"@(\d{8}T\d{4})@\d+@DBZH\.h5$")
MERC = "+proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +k=1 +units=m +nadgrids=@null +no_defs"
RV_API = "https://api.rainviewer.com/public/weather-maps.json"


def _session():
    s = requests.Session()
    s.headers["User-Agent"] = "nowcast-web/1.0 (+github actions)"
    return s


def _txt(v):
    return v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v)


# ---------------------------------------------------------------- OPERA
def s3_list(session, prefix):
    """Lista claves de un prefijo del bucket anónimo (S3 ListObjectsV2, con paginación)."""
    keys, token = [], None
    while True:
        params = {"list-type": 2, "prefix": prefix}
        if token:
            params["continuation-token"] = token
        r = session.get(f"{S3_ENDPOINT}/{S3_BUCKET}", params=params, timeout=TIMEOUT)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        keys += [e.text for e in root.iter() if e.tag.split("}")[-1] == "Key"]
        trunc = next((e.text for e in root.iter() if e.tag.split("}")[-1] == "IsTruncated"), "false")
        nxt = next((e.text for e in root.iter() if e.tag.split("}")[-1] == "NextContinuationToken"), None)
        if trunc != "true" or not nxt:
            return keys
        token = nxt


def parse_key_time(key):
    m = KEY_RE.search(key)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M").replace(tzinfo=timezone.utc)


def opera_latest_keys(session, n=4, now=None):
    """Las n claves DBZH más recientes (hoy y ayer en UTC), ordenadas de antigua a reciente."""
    now = now or datetime.now(timezone.utc)
    found = {}
    for day in (now - timedelta(days=1), now):
        for k in s3_list(session, f"{day:%Y/%m/%d}/OPERA/COMP/"):
            t = parse_key_time(k)
            if t:
                found[t] = k
    times = sorted(found)[-n:]
    return [(t, found[t]) for t in times]


def read_odim_composite(content: bytes) -> Frame:
    """ODIM HDF5 de composite -> Frame en dBZ. nodata -> nan; undetect -> -32."""
    with h5py.File(io.BytesIO(content), "r") as f:
        w = f["where"].attrs
        projdef = _txt(w["projdef"])
        dx, dy = float(w["xscale"]), float(w["yscale"])
        from pyproj import CRS, Transformer
        fwd = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_user_input(projdef), always_xy=True)
        x0, y0 = fwd.transform(float(w["UL_lon"]), float(w["UL_lat"]))
        d = f["dataset1/data1"]
        raw = d["data"][:]
        wa = d["what"].attrs
        gain, offset = float(wa.get("gain", 1.0)), float(wa.get("offset", 0.0))
        nodata, undetect = float(wa.get("nodata", 255)), float(wa.get("undetect", 0))
        qty = _txt(wa.get("quantity", "DBZH"))
        dbz = raw.astype(np.float32) * gain + offset
        dbz[raw == undetect] = -32.0
        dbz[raw == nodata] = np.nan
        ra = f["what"].attrs
        t = datetime.strptime(_txt(ra["date"]) + _txt(ra["time"]), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    if qty not in ("DBZH", "DBZ", "TH"):
        raise ValueError(f"Cantidad ODIM inesperada: {qty}")
    return Frame(t, dbz, Georef(projdef, x0, y0, dx, dy), "OPERA")


def load_opera(n=4, session=None):
    session = session or _session()
    out = []
    for t, key in opera_latest_keys(session, n):
        r = session.get(f"{S3_ENDPOINT}/{S3_BUCKET}/{key}", timeout=120)
        r.raise_for_status()
        fr = read_odim_composite(r.content)
        fr.time = t  # el nombre de fichero manda: es lo que usa la API
        out.append(fr)
    return out


# ------------------------------------------------------------ RainViewer
_PAL = [
    (99, 97, 89), (102, 99, 90), (105, 102, 92), (108, 104, 93), (111, 107, 95),
    (114, 110, 97), (117, 112, 98), (120, 115, 100), (124, 117, 101), (127, 120, 103),
    (130, 123, 105), (133, 125, 106), (136, 128, 108), (139, 130, 109), (142, 133, 111),
    (146, 136, 113), (158, 147, 117), (170, 158, 121), (182, 169, 126), (194, 180, 130),
    (206, 192, 135), (210, 196, 139), (214, 200, 143), (218, 204, 147), (222, 208, 151),
    (136, 221, 238), (108, 209, 235), (81, 197, 232), (54, 186, 229), (27, 174, 226),
    (0, 163, 224), (0, 154, 213), (0, 145, 202), (0, 136, 191), (0, 127, 180),
    (0, 119, 170), (0, 112, 163), (0, 105, 156), (0, 98, 149), (0, 91, 142),
    (0, 85, 136), (0, 81, 128), (0, 78, 120), (0, 74, 112), (0, 71, 104),
    (255, 238, 0), (255, 224, 0), (255, 210, 0), (255, 197, 0), (255, 183, 0),
    (255, 170, 0), (255, 159, 0), (255, 149, 0), (255, 139, 0), (255, 129, 0),
    (255, 68, 0), (242, 54, 0), (230, 40, 0), (217, 27, 0), (205, 13, 0),
    (193, 0, 0), (168, 0, 0), (143, 0, 0), (118, 0, 0), (93, 0, 0),
    (255, 170, 255), (255, 159, 255), (255, 149, 255), (255, 139, 255), (255, 129, 255),
    (255, 119, 255), (255, 108, 255), (255, 98, 255), (255, 88, 255), (255, 78, 255),
    (255, 255, 255),
]  # filas 22..96 de la paleta Universal Blue (dBZ = fila - 32) y blanco (>= 65 dBZ)
_PAL_DBZ = np.arange(len(_PAL), dtype=np.float32) - 10.0
_PAL_TREE = cKDTree(np.array(_PAL, dtype=np.float64))


def rgba_to_dbz(rgba):
    dbz = np.full(rgba.shape[:2], -32.0, np.float32)
    m = rgba[..., 3] > 0
    if m.any():
        _, idx = _PAL_TREE.query(rgba[..., :3][m].astype(np.float64))
        dbz[m] = _PAL_DBZ[idx]
    dbz[dbz < 5.0] = -32.0
    return dbz


def _tile_xy(lon, lat, z):
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    return x, y


def load_rainviewer(bbox, z=6, n=4, session=None):
    """Mosaico XYZ de los n últimos frames. bbox = (lon_o, lat_s, lon_e, lat_n)."""
    session = session or _session()
    meta = session.get(RV_API, timeout=TIMEOUT)
    meta.raise_for_status()
    j = meta.json()
    host, past = j["host"], j["radar"]["past"][-n:]
    x0, y1 = _tile_xy(bbox[0], bbox[1], z)
    x1, y0 = _tile_xy(bbox[2], bbox[3], z)
    size = 256
    origin = 20037508.342789244
    res = 2 * origin / (size * 2 ** z)
    geo = Georef(MERC, -origin + x0 * size * res, origin - y0 * size * res, res, res)
    frames = []
    for f in past:
        mosaic = np.full(((y1 - y0 + 1) * size, (x1 - x0 + 1) * size), np.nan, np.float32)
        for tx in range(x0, x1 + 1):
            for ty in range(y0, y1 + 1):
                url = f"{host}{f['path']}/{size}/{z}/{tx}/{ty}/2/0_0.png"
                try:
                    r = session.get(url, timeout=TIMEOUT)
                    r.raise_for_status()
                    rgba = np.array(Image.open(io.BytesIO(r.content)).convert("RGBA"))
                    mosaic[(ty - y0) * size:(ty - y0 + 1) * size,
                           (tx - x0) * size:(tx - x0 + 1) * size] = rgba_to_dbz(rgba)
                except Exception:
                    pass  # tile ausente -> nan
        t = datetime.fromtimestamp(int(f["time"]), tz=timezone.utc)
        frames.append(Frame(t, mosaic, geo, "RainViewer"))
    return frames


def downsample(frame, k):
    """Reduce la resolución k veces promediando en Z lineal (los nan no cuentan)."""
    if k <= 1:
        return frame
    h, w = frame.dbz.shape
    h2, w2 = h // k * k, w // k * k
    a = frame.dbz[:h2, :w2].reshape(h2 // k, k, w2 // k, k)
    z = np.where(np.isfinite(a), 10.0 ** (np.maximum(a, -32.0) / 10.0), np.nan).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        zm = np.nanmean(z, axis=(1, 3))
    dbz = (10.0 * np.log10(np.maximum(zm, 1e-3))).astype(np.float32)
    dbz[dbz < -31.5] = -32.0
    dbz[np.isnan(zm)] = np.nan
    g = frame.geo
    return Frame(frame.time, dbz, Georef(g.proj4, g.x0, g.y0, g.dx * k, g.dy * k), frame.source)
