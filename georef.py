"""Georreferenciación mínima: proyección + esquina superior izquierda + tamaño de píxel."""
from dataclasses import dataclass, replace
from datetime import datetime

import numpy as np
from pyproj import CRS, Transformer

WGS84 = CRS.from_epsg(4326)


@dataclass(frozen=True)
class Georef:
    proj4: str
    x0: float  # x de la esquina superior izquierda EXTERIOR (m)
    y0: float  # y de la esquina superior izquierda EXTERIOR (m)
    dx: float  # m por píxel (>0)
    dy: float  # m por píxel (>0)

    def _fwd(self):
        return Transformer.from_crs(WGS84, CRS.from_user_input(self.proj4), always_xy=True)

    def _inv(self):
        return Transformer.from_crs(CRS.from_user_input(self.proj4), WGS84, always_xy=True)

    def lonlat_to_pixel(self, lon, lat):
        """-> (col, fila) con el centro del primer píxel en (0, 0)."""
        x, y = self._fwd().transform(np.asarray(lon, float), np.asarray(lat, float))
        return (x - self.x0) / self.dx - 0.5, (self.y0 - y) / self.dy - 0.5

    def pixel_to_lonlat(self, col, row):
        x = self.x0 + (np.asarray(col, float) + 0.5) * self.dx
        y = self.y0 - (np.asarray(row, float) + 0.5) * self.dy
        lon, lat = self._inv().transform(x, y)
        return lon, lat

    def crop(self, col0, row0):
        return replace(self, x0=self.x0 + col0 * self.dx, y0=self.y0 - row0 * self.dy)

    @property
    def km_per_pixel(self):
        return self.dx / 1000.0


@dataclass
class Frame:
    time: datetime          # UTC, con zona horaria
    dbz: np.ndarray         # (filas, cols) float32; nan = sin dato; -32 = sin eco
    geo: Georef
    source: str


def crop_frame(frame: Frame, center_lat: float, center_lon: float, size: int) -> Frame:
    """Ventana size x size centrada en un punto; lo que cae fuera del dominio queda en nan."""
    c, r = frame.geo.lonlat_to_pixel(center_lon, center_lat)
    c0, r0 = int(round(float(c))) - size // 2, int(round(float(r))) - size // 2
    out = np.full((size, size), np.nan, np.float32)
    h, w = frame.dbz.shape
    sy0, sy1 = max(r0, 0), min(r0 + size, h)
    sx0, sx1 = max(c0, 0), min(c0 + size, w)
    if sy1 > sy0 and sx1 > sx0:
        out[sy0 - r0:sy1 - r0, sx0 - c0:sx1 - c0] = frame.dbz[sy0:sy1, sx0:sx1]
    return Frame(frame.time, out, frame.geo.crop(c0, r0), frame.source)


def downsample(frame: Frame, f: int) -> Frame:
    """Promedia bloques f x f en unidades lineales (Z), conservando nan si falta más de la mitad."""
    if f <= 1:
        return frame
    h, w = frame.dbz.shape
    h2, w2 = h // f * f, w // f * f
    a = frame.dbz[:h2, :w2].reshape(h2 // f, f, w2 // f, f)
    valid = np.isfinite(a)
    z = np.where(valid, 10.0 ** (np.maximum(a, -32.0) / 10.0), 0.0)
    n = valid.sum(axis=(1, 3))
    zm = z.sum(axis=(1, 3)) / np.maximum(n, 1)
    out = (10.0 * np.log10(np.maximum(zm, 10.0 ** -3.2))).astype(np.float32)
    out[n < (f * f) / 2.0] = np.nan
    g = frame.geo
    return Frame(frame.time, out, Georef(g.proj4, g.x0, g.y0, g.dx * f, g.dy * f), frame.source)


def target_downsample(frame: Frame, target_m: float = 2000.0) -> int:
    """Factor entero para acercar la resolución a ~2 km (el motor está dimensionado para eso)."""
    return max(1, int(round(target_m / frame.geo.dx)))
