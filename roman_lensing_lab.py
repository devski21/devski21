"""
================================================================================
 ROMAN LENSING LAB
 A telescope-data pipeline and relativistic visualization engine
================================================================================

A single-file scientific instrument that ingests real panoramic survey imagery
(FITS), then pushes every one of its pixels through an exact general-relativistic
ray-tracer.  The background sky is not a decorative texture: it is the *source
plane*, and what you see on screen is the image plane produced by solving the
lens equation for every pixel, every frame.

Where an exact result is impractical in real time the code uses a NAMED physical
approximation and says so in the comment beside it.  Nothing here is eyeballed.

--------------------------------------------------------------------------------
 UNIT SYSTEM
--------------------------------------------------------------------------------
 Geometrized units, G = c = 1.

   * Length    1 world unit == 1 screen pixel at zoom 1.
   * Mass      GM/c^2 has dimensions of LENGTH, so the hole's `M` IS its
               gravitational radius r_g.
   * Time      1 world time unit = the light-crossing time of 1 pixel.
   * Velocity  dimensionless, in units of c.  v = 1.0 IS the speed of light.

 Everything textbook then falls out at readable screen sizes:

     Schwarzschild radius  r_s   = 2M
     Photon sphere         r_ph  = 3M                     (a = 0)
     Shadow radius         b_c   = 3*sqrt(3) M = 2.598 r_s
     ISCO                  r_isco= 6M                     (a = 0)

 `PhysicalScale` maps those pixels onto real SI quantities with
 astropy.constants and astropy.cosmology, so the HUD can report the true
 Schwarzschild radius in km and the true Einstein radius in arcseconds for a
 chosen lens mass and redshift.  The RENDER is exaggerated (a real Einstein ring
 for a stellar-mass lens is microarcseconds across); the READOUT is not, and the
 HUD labels which is which.

--------------------------------------------------------------------------------
 PHYSICS INDEX  (equation -> implementation)
--------------------------------------------------------------------------------
  Exact Schwarzschild light deflection (Darwin, elliptic
      integrals, scipy.special) .................... DeflectionTable
  Critical impact parameter b_c = 3*sqrt(3) M ...... DeflectionTable.b_critical
  Thin-lens equation  beta = theta - D_LS/D_S alpha  LensModel.radial_map
  Einstein radius (scipy.optimize.brentq) .......... LensModel.einstein_radius
  Lens magnification  mu = [(b/r)(db/dr)]^-1 ....... LensModel.magnification
  Kerr frame dragging of the image plane ........... LensModel._drag_twist
  Kerr horizon / ergosphere / photon sphere ........ BlackHole
  Paczynski-Wiita pseudo-Newtonian potential ....... GravityField.central_acceleration
  Lense-Thirring gravitomagnetic acceleration ...... GravityField.frame_drag_acceleration
  de Sitter cosmological repulsion a = H^2 r ....... QuintessenceField
  FRW scale factor a(t) = exp(H t) ................. QuintessenceField
  Relativistic dynamics dp/dt = F, p = gamma m v ... GravityField.proper_accel
  Runge-Kutta 4th order ............................ RK4Integrator
  Schwarzschild time dilation ...................... Spacecraft.dilation_factor
  Tidal strain 2GM dr/r^3 (spaghettification) ...... BodyPopulation
  Zapolsky-Salpeter mass-radius relation ........... StellarStructure
  Chandrasekhar / TOV collapse thresholds .......... StellarStructure
  Alcubierre metric shape function + York time ..... WarpBubble
  Penrose superradiance, bounded by Omega_H ........ DarkMatterField
  Beer-Lambert dust extinction ..................... DustFog
  Sersic galaxy profiles, PSF convolution .......... SkySurvey (astropy + scipy)

--------------------------------------------------------------------------------
 CONTROLS
--------------------------------------------------------------------------------
  Mouse   left click ..... spawn selected class     left drag on body .. add mass
          right drag ..... pan                      wheel .............. zoom
  Keys    W / S ........ Alcubierre warp / slingshot     arrows ... thrust or steer
          D / L / U .... dark matter / laser sensor / deep underground shielding
          Z / T / B .... dust fog / trails / boundary labels
          M / R / C .... magnification map / reset ship / clear bodies
          1 / 2 / 3 .... spawn planet / star / dark matter
          F ............ chase camera        SPACE ... pause        ESC ... quit

  CLI     python roman_lensing_lab.py [--fits PATH]
          python roman_lensing_lab.py --selftest
          python roman_lensing_lab.py --headless N
================================================================================
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

from scipy.special import ellipk, ellipkinc          # noqa: E402  exact deflection
from scipy.optimize import brentq                    # noqa: E402  Einstein radius
from scipy.ndimage import gaussian_filter            # noqa: E402  PSF convolution

from astropy.io import fits                          # noqa: E402
from astropy.wcs import WCS                          # noqa: E402
from astropy.modeling.models import Sersic2D         # noqa: E402
from astropy.visualization import (                  # noqa: E402
    ZScaleInterval, AsinhStretch, PercentileInterval)
from astropy import constants as const               # noqa: E402
from astropy import units as u                       # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning, module="astropy")


# ==============================================================================
#  SECTION 0 -- CONFIGURATION
# ==============================================================================

@dataclass
class Config:
    """Every tunable in one place.  Lengths are world units (= px at zoom 1)."""

    # --- window -------------------------------------------------------------
    width: int = 1600
    height: int = 900
    sidebar_w: int = 340
    target_fps: int = 60
    caption: str = "Roman Lensing Lab -- FITS ray-tracing / relativistic sandbox"

    # --- lensing raster -----------------------------------------------------
    # The lens map is evaluated on a decimated raster and upscaled.  The lensed
    # field is smooth on sub-pixel scales everywhere except at the critical
    # curve, so a factor of 2 is visually free and costs a quarter of the work.
    lens_downsample: int = 2
    smooth_upscale: bool = False   # mip pre-filtering already band-limits

    # --- central black hole -------------------------------------------------
    bh_mass: float = 26.0
    bh_mass_min: float = 8.0
    bh_mass_max: float = 70.0
    bh_spin: float = 0.85

    # --- lens geometry ------------------------------------------------------
    # Effective lensing depth  D_eff = D_L D_LS / D_S, in world units.  Sets the
    # Einstein radius via theta_E = sqrt(2 r_s D_eff) in the weak-field limit.
    lens_depth: float = 190.0
    lens_depth_min: float = 0.0
    lens_depth_max: float = 620.0

    # --- cosmology ----------------------------------------------------------
    hubble: float = 0.00055
    hubble_min: float = 0.0
    hubble_max: float = 0.0035

    # --- integration --------------------------------------------------------
    dt: float = 0.55
    dt_min: float = 0.0
    dt_max: float = 2.0
    max_substep: float = 0.70        # bound the RK4 STEP SIZE, not the count

    # --- populations --------------------------------------------------------
    max_bodies: int = 200
    max_body_mj: float = 24000.0
    max_dark_particles: int = 900
    trail_len: int = 190

    # --- survey field -------------------------------------------------------
    sky_w: int = 1500
    sky_h: int = 1100
    sky_galaxies: int = 46
    sky_stars: int = 900
    sky_seed: int = 20260901

    eps: float = 1e-9


CFG = Config()


class Palette:
    VOID          = (4, 5, 11)
    PANEL         = (12, 15, 25)
    PANEL_EDGE    = (36, 44, 68)
    TEXT          = (214, 224, 244)
    TEXT_DIM      = (122, 136, 168)
    ACCENT        = (118, 206, 255)
    WARN          = (255, 176, 74)
    DANGER        = (255, 92, 104)
    OK            = (120, 240, 168)

    HORIZON       = (0, 0, 0)
    PHOTON        = (255, 226, 138)
    ERGO          = (168, 122, 255)
    EINSTEIN      = (120, 255, 224)
    ISCO          = (86, 116, 158)

    PLANET        = (110, 170, 224)
    ROCK          = (176, 150, 128)
    BROWN_DWARF   = (208, 96, 64)
    STAR          = (255, 238, 190)
    DARK_MATTER   = (176, 120, 255)
    SHIP          = (150, 255, 214)
    WARP          = (128, 236, 255)
    GRID_CONTRACT = (86, 172, 255)
    GRID_EXPAND   = (255, 122, 96)


# ==============================================================================
#  SECTION 1 -- VECTOR MATHEMATICS
# ==============================================================================

def norm(v: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    return np.sqrt(np.sum(v * v, axis=axis, keepdims=keepdims))


def lorentz_gamma(v: np.ndarray) -> np.ndarray:
    """gamma = 1/sqrt(1 - v^2/c^2), c = 1.  Speed is clamped just below c."""
    s2 = np.minimum(np.sum(v * v, axis=-1), 1.0 - 1e-9)
    return 1.0 / np.sqrt(1.0 - s2)


def smoothstep(edge0: float, edge1: float, x: np.ndarray | float) -> np.ndarray:
    """
    Hermite 3t^2 - 2t^3, clamped.  Edges may be DESCENDING (edge0 > edge1) to
    get a falling ramp: the span keeps its sign and only its magnitude is
    floored, so a reversed pair inverts the ramp instead of saturating it.
    """
    span = edge1 - edge0
    if abs(span) < CFG.eps:
        span = math.copysign(CFG.eps, span if span != 0.0 else 1.0)
    t = np.clip((x - edge0) / span, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def lerp_color(a: Sequence[float], b: Sequence[float], t: float):
    t = min(max(t, 0.0), 1.0)
    return (int(a[0] + (b[0] - a[0]) * t), int(a[1] + (b[1] - a[1]) * t),
            int(a[2] + (b[2] - a[2]) * t))


def pack_rgb(rgb: np.ndarray) -> np.ndarray:
    """
    Pack an (H, W, 3) uint8 image into (H, W) uint32 as 0x00RRGGBB.

    This is the single most important optimization in the renderer.  A 32-bit
    pygame surface stores exactly this layout, so the ray-tracer can gather ONE
    contiguous 4-byte value per pixel instead of three strided bytes -- measured
    at ~5x faster for the same result -- and blit it with zero conversion.
    """
    r = rgb[..., 0].astype(np.uint32)
    g = rgb[..., 1].astype(np.uint32)
    b = rgb[..., 2].astype(np.uint32)
    return (r << 16) | (g << 8) | b


# ==============================================================================
#  SECTION 2 -- THE INTEGRATOR
#
#  Classical Runge-Kutta 4th order.  Local error O(h^5), global O(h^4).  Euler is
#  O(h) globally and secularly pumps orbital energy -- near this black hole it
#  would turn every bound orbit into a lie.  Euler appears nowhere in this file.
#
#      k1 = f(t, y);  k2 = f(t+h/2, y+h k1/2);  k3 = f(t+h/2, y+h k2/2)
#      k4 = f(t+h, y+h k3);   y <- y + h(k1 + 2k2 + 2k3 + k4)/6
#
#  `y` is an (N, 4) array [x, y, vx, vy], so an entire population advances in
#  four vectorized derivative evaluations regardless of N.
# ==============================================================================

class RK4Integrator:
    __slots__ = ("derivative",)

    def __init__(self, derivative: Callable[[np.ndarray, float], np.ndarray]):
        self.derivative = derivative

    def step(self, state: np.ndarray, t: float, h: float) -> np.ndarray:
        if state.size == 0 or h == 0.0:
            return state
        f = self.derivative
        k1 = f(state, t)
        k2 = f(state + (0.5 * h) * k1, t + 0.5 * h)
        k3 = f(state + (0.5 * h) * k2, t + 0.5 * h)
        k4 = f(state + h * k3, t + h)
        return state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def integrate(self, state: np.ndarray, t: float, h: float,
                  substeps: int) -> np.ndarray:
        substeps = max(1, substeps)
        sub = h / substeps
        for i in range(substeps):
            state = self.step(state, t + i * sub, sub)
        return state

    @staticmethod
    def substeps_for(dt: float) -> int:
        """
        RK4 error is governed by the STEP SIZE, so bound h and derive the count.
        Fixing the count instead would let accuracy degrade linearly as the user
        drags the dt slider.
        """
        return max(1, int(math.ceil(abs(dt) / CFG.max_substep)))


# ==============================================================================
#  SECTION 3 -- THE QUINTESSENCE FIELD  (dark energy)
#
#  Space is modelled as a medium with an equation of state, driven by a
#  homogeneous scalar field whose potential energy acts as a cosmological
#  constant Lambda.  From the second Friedmann equation with w = -1,
#
#      a_ddot / a = Lambda c^2 / 3 = H^2       ->      a(t) = exp(H t)
#
#  The dynamical consequence, exact for a Lambda-dominated universe, is an
#  outward acceleration on every particle:
#
#      a_vac(r) = (Lambda c^2 / 3) r = H^2 r
#
#  which is why no special-case code is needed to "unbind distant galaxies": a
#  body is bound iff local gravity beats H^2 r.  The same scale factor stretches
#  the SOURCE-PLANE sampling coordinates, so background galaxies physically
#  recede from one another in the image as the simulation runs.
# ==============================================================================

class QuintessenceField:
    def __init__(self, hubble: float = CFG.hubble):
        self.hubble = hubble
        self.log_a = 0.0          # ln a(t), stored logarithmically
        self.time = 0.0

    def advance(self, dt: float) -> None:
        self.log_a += self.hubble * dt
        self.time += dt

    @property
    def scale_factor(self) -> float:
        return math.exp(self.log_a)

    def vacuum_acceleration(self, pos: np.ndarray) -> np.ndarray:
        if self.hubble <= CFG.eps:
            return np.zeros_like(pos)
        return (self.hubble * self.hubble) * pos

    def turnaround_radius(self, mass: float, r_s: float = 0.0) -> float:
        """
        Where inward gravity exactly cancels outward dark energy.

        Newtonian answer is (GM/H^2)^(1/3), but this engine's force law is
        Paczynski-Wiita, so the honest balance solves

            M/(r - r_s)^2 = H^2 r   <=>   H^2 r (r - r_s)^2 - M = 0

        Near a hole the two differ by an order of magnitude, so the cubic root
        is what gets reported.
        """
        if self.hubble <= CFG.eps:
            return float("inf")
        h2 = self.hubble ** 2
        if r_s <= 0.0:
            return (mass / h2) ** (1.0 / 3.0)
        roots = np.roots([h2, -2.0 * h2 * r_s, h2 * r_s * r_s, -mass])
        real = [float(z.real) for z in roots
                if abs(z.imag) < 1e-9 and z.real > r_s * 1.000001]
        return max(real) if real else float("inf")


# ==============================================================================
#  SECTION 4 -- KERR GEOMETRY
#
#  A rotating hole viewed down its spin axis.  Boyer-Lindquist results, G = c = 1,
#  a = a* M:
#
#      Outer horizon        r+    = M + sqrt(M^2 - a^2)
#      Ergosphere (equator) r_E   = 2M
#      Photon orbits        r_ph  = 2M{1 + cos( (2/3) arccos( -/+ a/M ) )}
#      ISCO                 Bardeen-Press-Teukolsky Z1/Z2 construction
#      Horizon ang. vel.    Omega_H = a / (2 M r+)
# ==============================================================================

class BlackHole:
    def __init__(self, mass: float = CFG.bh_mass, spin: float = CFG.bh_spin):
        self.mass = float(mass)
        self.spin = float(np.clip(spin, 0.0, 0.999))
        self.position = np.zeros(2, dtype=np.float64)
        self.phase = 0.0

    @property
    def a(self) -> float:
        """Kerr spin parameter with length dimensions, a = J/(Mc)."""
        return self.spin * self.mass

    @property
    def r_s(self) -> float:
        return 2.0 * self.mass

    @property
    def r_horizon(self) -> float:
        m, a = self.mass, self.a
        return m + math.sqrt(max(m * m - a * a, 0.0))

    @property
    def r_ergosphere(self) -> float:
        """Equatorial static limit: M + sqrt(M^2 - a^2 cos^2 th) = 2M at th=pi/2."""
        return 2.0 * self.mass

    @property
    def r_photon(self) -> float:
        return 2.0 * self.mass * (1.0 + math.cos((2.0 / 3.0) * math.acos(-self.spin)))

    @property
    def r_photon_retro(self) -> float:
        return 2.0 * self.mass * (1.0 + math.cos((2.0 / 3.0) * math.acos(self.spin)))

    @property
    def r_isco(self) -> float:
        m, astar = self.mass, self.spin
        z1 = 1.0 + (1.0 - astar ** 2) ** (1.0 / 3.0) * (
            (1.0 + astar) ** (1.0 / 3.0) + (1.0 - astar) ** (1.0 / 3.0))
        z2 = math.sqrt(3.0 * astar ** 2 + z1 * z1)
        return m * (3.0 + z2 - math.sqrt(max((3.0 - z1) * (3.0 + z1 + 2.0 * z2), 0.0)))

    @property
    def angular_momentum(self) -> float:
        return self.a * self.mass

    @property
    def omega_horizon(self) -> float:
        """Omega_H = a/(2 M r+): the ceiling on superradiant amplification."""
        return self.a / max(2.0 * self.mass * self.r_horizon, CFG.eps)

    def lapse(self, r: np.ndarray | float) -> np.ndarray | float:
        """sqrt(1 - r_s/r) = dtau/dt for a static observer."""
        val = 1.0 - self.r_s / np.maximum(r, self.r_s * 1.0000001)
        return np.sqrt(np.clip(val, 1e-9, 1.0))

    def circular_speed(self, r: np.ndarray | float) -> np.ndarray | float:
        """
        Circular-orbit speed under the engine's ACTUAL equation of motion.

        The force law is Paczynski-Wiita but the dynamics are relativistic:
        dv/dt = (1/gamma)[a - (v.a)v].  On a circular orbit v.a = 0, so the
        condition is v^2/r = |a|/gamma, i.e. v^2 gamma = r|a| = K.  With u = v^2
        that is u/sqrt(1-u) = K, a quadratic u^2 + K^2 u - K^2 = 0 with root

            u = ( -K^2 + sqrt(K^4 + 4K^2) ) / 2

        reducing to sqrt(M/r) in the weak field.  Using the Newtonian value
        instead would launch every "circular" orbit visibly eccentric.
        """
        r = np.maximum(r, self.r_s * 1.02)
        k = r * self.mass / np.maximum((r - self.r_s) ** 2, CFG.eps)
        k2 = k * k
        u = 0.5 * (-k2 + np.sqrt(k2 * k2 + 4.0 * k2))
        return np.sqrt(np.clip(u, 0.0, 0.9801))


# ==============================================================================
#  SECTION 5 -- EXACT LIGHT DEFLECTION  (Darwin 1959, via elliptic integrals)
#
#  The bending of a light ray by a Schwarzschild mass is NOT 4GM/(c^2 b).  That
#  is only the first term of a series which fails badly near the hole -- at the
#  photon sphere the true deflection diverges logarithmically while 4M/b is a
#  finite 0.77 rad.  Since the whole point of this instrument is the strong-field
#  regime, the exact result is used instead.
#
#  For a ray with closest approach r_0, the impact parameter is fixed by the
#  metric:
#
#      b(r_0) = r_0 / sqrt(1 - 2M/r_0)
#
#  and the total deflection has the closed form (Darwin; Chandrasekhar Ch.3)
#
#      Q^2      = (r_0 - 2M)(r_0 + 6M)
#      k^2      = (Q - r_0 + 6M) / (2Q)
#      sin^2 z  = (Q - r_0 + 2M) / (Q - r_0 + 6M)
#      alpha    = 4 sqrt(r_0/Q) [ K(k) - F(z, k) ] - pi
#
#  with K the complete and F the incomplete elliptic integral of the first kind,
#  evaluated here by scipy.special (which parametrizes by m = k^2).
#
#  Two limits the self-test checks against:
#      r_0 -> infinity   alpha -> 4M/b        (Einstein's 1915 result)
#      r_0 -> 3M         alpha -> infinity    (the photon sphere; b -> 3 sqrt(3) M)
#
#  EVALUATION STRATEGY.  Elliptic integrals are far too slow to call per pixel
#  per frame.  Instead the exact curve is tabulated once over a logarithmically
#  spaced set of r_0 (dense near the photon sphere where alpha varies fastest)
#  and evaluated at render time by np.interp -- exact where it matters, and
#  roughly three orders of magnitude cheaper.
# ==============================================================================

class DeflectionTable:
    """Tabulated exact Schwarzschild deflection alpha(b), rebuilt only on mass change."""

    SAMPLES = 4096

    def __init__(self, mass: float):
        self.mass = 0.0
        self.b: np.ndarray = np.zeros(0)
        self.alpha: np.ndarray = np.zeros(0)
        self.rebuild(mass)

    # ------------------------------------------------------------- closed form
    @staticmethod
    def deflection_exact(r0: np.ndarray, mass: float) -> np.ndarray:
        """alpha(r_0) in radians.  Valid for r_0 > 3M (outside the photon sphere)."""
        r0 = np.asarray(r0, dtype=np.float64)
        m = mass
        q = np.sqrt((r0 - 2.0 * m) * (r0 + 6.0 * m))
        # scipy's elliptic integrals take the parameter m = k^2.
        k2 = (q - r0 + 6.0 * m) / (2.0 * q)
        sin2 = (q - r0 + 2.0 * m) / (q - r0 + 6.0 * m)
        zeta = np.arcsin(np.sqrt(np.clip(sin2, 0.0, 1.0)))
        k2 = np.clip(k2, 0.0, 1.0 - 1e-15)
        return 4.0 * np.sqrt(r0 / q) * (ellipk(k2) - ellipkinc(zeta, k2)) - np.pi

    @staticmethod
    def impact_parameter(r0: np.ndarray, mass: float) -> np.ndarray:
        """b(r_0) = r_0 / sqrt(1 - 2M/r_0)."""
        r0 = np.asarray(r0, dtype=np.float64)
        return r0 / np.sqrt(1.0 - 2.0 * mass / r0)

    @property
    def b_critical(self) -> float:
        """
        b_c = 3 sqrt(3) M = 2.598 r_s.

        Rays with b < b_c are captured, so this -- not the horizon -- is the
        radius of the black hole SHADOW an observer actually sees.  It is why the
        Event Horizon Telescope image of M87* is noticeably larger than 2 r_s.
        """
        return 3.0 * math.sqrt(3.0) * self.mass

    # ---------------------------------------------------------------- building
    def rebuild(self, mass: float) -> None:
        if abs(mass - self.mass) < 1e-12:
            return
        self.mass = float(mass)
        m = self.mass
        # r_0 sampled from just outside the photon sphere out to a far field.
        # The 1/x^3 spacing packs samples where alpha varies fastest.
        t = np.linspace(0.0, 1.0, self.SAMPLES) ** 3
        r0 = 3.0 * m * (1.0 + 1e-6) + t * (4000.0 * m)
        self.alpha = self.deflection_exact(r0, m)
        self.b = self.impact_parameter(r0, m)
        # np.interp needs a strictly increasing abscissa; b(r_0) is monotone for
        # r_0 > 3M, but guard against duplicate values from finite precision.
        keep = np.concatenate([[True], np.diff(self.b) > 0.0])
        self.b, self.alpha = self.b[keep], self.alpha[keep]

    # ------------------------------------------------------------- evaluation
    def __call__(self, b: np.ndarray) -> np.ndarray:
        """
        alpha(b) for an array of impact parameters, vectorized.

        Below b_c the ray is captured and no deflection is defined; the table's
        first entry is returned there, and the caller masks those pixels into the
        shadow so the value is never displayed.

        BEYOND the tabulated range the table must NOT be clamped to a constant:
        that would leave a fixed residual deflection out to infinity.  The
        post-Newtonian series is asymptotically exact there, to a relative error
        of order (M/b)^2 -- below 1e-12 at the switchover -- so it takes over:

            alpha(b) = 4M/b + 15 pi M^2 / (4 b^2) + O(M^3/b^3)
        """
        b = np.asarray(b, dtype=np.float64)
        out = np.interp(b, self.b, self.alpha,
                        left=self.alpha[0], right=np.nan)
        far = b > self.b[-1]
        if np.any(far):
            m = self.mass
            bf = b[far]
            out[far] = 4.0 * m / bf + 15.0 * math.pi * m * m / (4.0 * bf * bf)
        return out


# ==============================================================================
#  SECTION 6 -- THE LENS MODEL
#
#  Inverse ray tracing.  For every pixel of the IMAGE plane we ask: which point
#  of the SOURCE plane does the light arriving here come from?  The thin-lens
#  equation, with the exact alpha above, answers it:
#
#      beta = theta - (D_LS / D_S) alpha(theta)
#
#  Writing screen radius r = theta D_L and folding the distance ratios into a
#  single effective depth D_eff = D_L D_LS / D_S (the `lens_depth` slider), the
#  radial map becomes
#
#      r_source(r) = r - D_eff * alpha(b = r)
#
#  Everything the instrument shows follows from that one scalar function:
#
#  * EINSTEIN RING.  r_source(r_E) = 0 has a root: light from a source exactly
#    behind the lens arrives from every azimuth at once.  Solved to machine
#    precision with scipy.optimize.brentq rather than the weak-field guess.
#  * SECONDARY IMAGE.  Inside r_E the map goes negative: the ray has crossed the
#    optic axis, so the image is inverted.  Taking |r_source| with phi -> phi+pi
#    produces the counter-image automatically, which is why lensed arcs appear in
#    pairs without a single line of special-case code.
#  * MAGNIFICATION.  For a circularly symmetric lens the Jacobian is diagonal in
#    polar coordinates, with tangential and radial eigenvalues:
#
#        mu^-1 = (r_source / r) * (d r_source / d r)
#
#    It diverges on the critical curve (the Einstein ring) and changes sign
#    across it -- the sign flip IS the parity inversion of the counter-image.
#  * FRAME DRAGGING.  Viewed down the spin axis, the Kerr gravitomagnetic
#    deflection is purely azimuthal, alpha_t = 4aM/b^2 (the spin correction to
#    the Schwarzschild term).  Converting that transverse displacement to an
#    angle gives a twist  dphi = D_eff 4 a M / r^3, falling off as 1/r^3 exactly
#    like Lense-Thirring precession, which shears the background into the
#    characteristic whirlpool.
#
#  PERFORMANCE.  Every quantity above depends only on RADIUS, so the expensive
#  part is a handful of 1-D functions.  They are evaluated on a 1-D radial grid
#  once and broadcast to the 2-D raster through np.interp, and the resulting
#  source coordinates are cached until the mass, spin, depth or camera changes.
#  The per-frame cost is then one integer gather (see LensRenderer).
# ==============================================================================

class LensModel:
    """Radial lens mapping, magnification, and the cached 2-D deflection field."""

    def __init__(self, hole: BlackHole, depth: float = CFG.lens_depth):
        self.hole = hole
        self.depth = float(depth)
        self.table = DeflectionTable(hole.mass)

    def sync(self) -> None:
        self.table.rebuild(self.hole.mass)

    # ------------------------------------------------------------ radial map
    def radial_map(self, r: np.ndarray) -> np.ndarray:
        """r_source(r) = r - D_eff alpha(r).  Negative inside the Einstein ring."""
        return r - self.depth * self.table(r)

    def deflection(self, r: np.ndarray) -> np.ndarray:
        return self.table(r)

    def einstein_radius(self) -> float:
        """
        Root of r_source(r) = 0, found by Brent's method.

        Bracketed between the critical impact parameter (where alpha diverges,
        so r_source is hugely negative) and a far radius (where alpha is
        negligible, so r_source ~ r > 0).  A sign change is therefore guaranteed
        whenever the lens is strong enough to form a ring at all.
        """
        if self.depth <= CFG.eps:
            return 0.0
        lo = self.table.b_critical * (1.0 + 1e-6)
        hi = max(50.0 * self.hole.mass, 40.0 * self.depth)
        f = lambda r: float(self.radial_map(np.array([r]))[0])
        if f(lo) > 0.0 or f(hi) < 0.0:
            return 0.0                       # no critical curve for this geometry
        return float(brentq(f, lo, hi, xtol=1e-10, rtol=1e-12))

    def magnification(self, r: np.ndarray) -> np.ndarray:
        """
        mu(r) = [ (r_src/r) * d(r_src)/dr ]^-1, evaluated by central differences.

        Returned SIGNED: negative values are parity-inverted images, which is the
        physical distinction between the two sides of the critical curve.
        """
        r = np.maximum(np.asarray(r, dtype=np.float64), CFG.eps)
        h = np.maximum(r * 1e-4, 1e-6)
        rs = self.radial_map(r)
        drs = (self.radial_map(r + h) - self.radial_map(r - h)) / (2.0 * h)
        denom = (rs / r) * drs
        return 1.0 / np.where(np.abs(denom) < 1e-12, 1e-12, denom)

    def stretch_factors(self, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Local linear stretch of the lens mapping, radial and tangential.

            radial      |d r_src / d r|
            tangential  |r_src / r|

        Their product is 1/|mu|.  Individually they are what the ray tracer needs
        for texture filtering: they say how many source pixels are compressed
        into one image pixel along each axis.
        """
        r = np.maximum(np.asarray(r, dtype=np.float64), CFG.eps)
        h = np.maximum(r * 1e-4, 1e-6)
        rs = self.radial_map(r)
        drs = (self.radial_map(r + h) - self.radial_map(r - h)) / (2.0 * h)
        return np.abs(drs), np.abs(rs / r)

    def _drag_twist(self, r: np.ndarray) -> np.ndarray:
        """
        Azimuthal shear from Kerr frame dragging, dphi = D_eff 4 a M / r^3.

        Capped near the shadow: the expansion is a weak-field spin correction and
        has no business winding the image through many turns where it is no
        longer valid.
        """
        a = self.hole.a
        if a <= CFG.eps:
            return np.zeros_like(r)
        rr = np.maximum(r, self.table.b_critical)
        twist = self.depth * 4.0 * a * self.hole.mass / (rr ** 3)
        return np.clip(twist, -1.2, 1.2)

    # ------------------------------------------------- 2-D source-plane field
    def build_field(self, xs: np.ndarray, ys: np.ndarray,
                    pixel_width: float, n_mips: int) -> tuple:
        """
        Map a raster of lens-plane coordinates to source-plane coordinates.

        `xs`, `ys` are 2-D arrays of world offsets from the hole in (W, H) order,
        matching pygame's surfarray convention so no transpose is ever needed.
        `pixel_width` is the world size of one raster sample (downsample / zoom),
        used to pick the texture-filtering level.

        THE COUNTER-IMAGE IS FREE.  Inside the critical curve the radial map goes
        negative, meaning the ray crossed the optic axis.  The inverted image
        sits at (|r_src|, phi + pi) -- but that is identically (r_src, phi),
        because a negative radius at the same azimuth already points the other
        way.  So lensed arcs appear in pairs with no branch, no absolute value
        and no special case: the sign of r_src carries the parity by itself.

        PERFORMANCE.  Every quantity here is a function of RADIUS alone, so all
        of it -- including the sines and cosines of the frame-drag twist and the
        log2 of the mip level -- is evaluated on a 1-D radial table and broadcast
        to the raster by lookup.  Three details make the rebuild affordable:

          * The table is UNIFORM in radius, so the index is one multiply rather
            than the binary search np.interp needs per element.
          * The raster maths runs in float32.  Coordinates reach ~2000 px and
            float32 carries ~7 digits, so the error is ~1e-4 px -- three orders
            below one display pixel -- while the arithmetic is ~3x faster than
            float64.  This was measured, not assumed: it is the single largest
            term in the rebuild.
          * The magnification is NOT gathered here.  It is only needed by the
            false-colour distortion map, which is itself lazy, so the index and
            the table are handed back and the gather happens only if that view
            is ever opened.

        Returns (sx, sy, captured, idx, mu_table, mip_level).
        """
        xs = np.asarray(xs, dtype=np.float32)
        ys = np.asarray(ys, dtype=np.float32)
        r = np.sqrt(xs * xs + ys * ys)
        np.maximum(r, np.float32(1e-6), out=r)
        cos_p = xs / r
        sin_p = ys / r

        # -- 1-D radial table, uniform so lookup is O(1) ----------------------
        n_tab = 8192
        r_max = float(r.max()) * 1.02 + 1.0
        grid = np.linspace(0.0, r_max, n_tab)
        grid[0] = min(1e-3, r_max / n_tab)          # keep r = 0 out of the maths
        inv_dr = np.float32((n_tab - 1) / r_max)

        tw_g = self._drag_twist(grid)
        rad_g, tan_g = self.stretch_factors(grid)
        # Mip level: one raster sample spans `pixel_width` world units, which the
        # lens compresses by max(radial, tangential) source pixels.  The level
        # whose blur matches that width is log2(width) -- the standard rule, and
        # here just the statement that a detector integrates over the solid
        # angle it subtends.
        width_g = np.maximum(rad_g, tan_g) * pixel_width
        level_g = np.clip(np.log2(np.maximum(width_g, 1.0)), 0, n_mips - 1)

        t_src = self.radial_map(grid).astype(np.float32)
        t_cos = np.cos(tw_g).astype(np.float32)     # pre-rotated: no 2-D trig
        t_sin = np.sin(tw_g).astype(np.float32)
        t_lev = level_g.astype(np.int32)
        mu_table = self.magnification(grid)

        idx = r * inv_dr
        np.clip(idx, 0, n_tab - 1, out=idx)
        idx = idx.astype(np.int32)

        r_src = t_src[idx]
        cos_t = t_cos[idx]
        sin_t = t_sin[idx]
        # r_src carries its own sign, so this is the full lens map including the
        # inverted counter-image (see the note above).
        sx = r_src * (cos_p * cos_t - sin_p * sin_t)      # r * cos(phi + twist)
        sy = r_src * (sin_p * cos_t + cos_p * sin_t)      # r * sin(phi + twist)

        captured = r < np.float32(self.table.b_critical)
        return sx, sy, captured, idx, mu_table, t_lev[idx]


# ==============================================================================
#  SECTION 7 -- THE DATA INGESTION ENGINE  (Astropy / FITS)
#
#  The source plane is real telescope data.  This class is the pipeline front
#  end: it finds a FITS file, pulls the first image HDU with 2 or more spatial
#  axes, collapses any extra axes (spectral cubes, detector stacks) and hands
#  back a single float32 intensity matrix plus its World Coordinate System.
#
#  When no file is present it SYNTHESIZES a Roman-like wide-field exposure and
#  writes it out as a genuine FITS file, then reads it back through exactly the
#  same loader.  The ingestion path is therefore exercised identically whether
#  or not real data is available -- there is no "mock mode" branch downstream.
#
#  The synthetic field is built from the same components a real one contains:
#
#    * Galaxies      astropy.modeling.models.Sersic2D, the standard surface
#                    brightness law  I(R) = I_e exp{-b_n[(R/R_e)^(1/n) - 1]},
#                    with n drawn across the de Vaucouleurs (n=4, ellipticals)
#                    to exponential-disc (n=1, spirals) range.
#    * Stars         delta functions convolved with the instrumental PSF.
#    * PSF           scipy.ndimage.gaussian_filter -- a Gaussian core is the
#                    standard first-order model for a diffraction-limited PSF.
#    * Background    a smooth zodiacal-light gradient.
#    * Noise         Poisson (photon shot noise) + Gaussian (detector read
#                    noise), which is what actually sets the faint-end limit.
#
#  DISPLAY STRETCH.  Astronomical dynamic range is enormous, so the data is
#  normalized with astropy.visualization: ZScaleInterval (the IRAF algorithm
#  used by DS9 to pick display cuts from the pixel distribution) composed with
#  AsinhStretch, which is linear near zero and logarithmic at the bright end --
#  the standard choice for showing faint structure without blowing out cores.
# ==============================================================================

class SkySurvey:
    """Loads or synthesizes the source-plane image and prepares it for the ray tracer."""

    FITS_EXT = (".fits", ".fit", ".fts", ".fits.gz", ".fit.gz")
    DEFAULT_MOCK = "roman_mock_field.fits"
    MOCK_VERSION = 2          # bump to invalidate a previously cached mock

    def __init__(self, path: str | None = None, quiet: bool = False):
        self.quiet = quiet
        self.source_name = ""
        self.synthetic = False
        self.data: np.ndarray = np.zeros((4, 4), dtype=np.float32)
        self.header: fits.Header = fits.Header()
        self.wcs: WCS | None = None
        self._load(path)
        self.rgb = self._to_rgb(self.data)
        self.packed = pack_rgb(self.rgb)                 # (H, W) uint32
        # Stored in (W, H) order so pygame.surfarray needs no transpose at blit.
        self.packed_wh = np.ascontiguousarray(self.packed.T)
        self.sw = self.packed_wh.shape[0]                # width  (x extent)
        self.sh = self.packed_wh.shape[1]                # height (y extent)
        self._build_mip_pyramid()

    # ------------------------------------------------------------ mip levels
    MIP_SIGMAS = (0.0, 1.2, 3.0, 7.0)

    def _build_mip_pyramid(self) -> None:
        """
        Pre-blurred copies of the source image, for texture filtering.

        Near the shadow the lens map compresses a huge range of source radius
        into a thin annulus of image radius, so a single output pixel covers
        many source pixels.  Point-sampling that produces the classic aliasing
        rings.  The physically correct answer is that a detector pixel INTEGRATES
        over the source solid angle it subtends, so the ray tracer selects a
        pre-filtered level whose blur matches the local footprint -- mip-mapping,
        motivated by the instrument rather than by graphics convention.

        All levels live in ONE flat array so level selection is a single integer
        offset added to the gather index: no branching, no second gather.
        """
        levels = []
        for sigma in self.MIP_SIGMAS:
            if sigma <= 0.0:
                levels.append(self.packed_wh)
                continue
            # Blur in RGB, not in the packed integer (which would mix channels).
            blurred = np.empty_like(self.rgb)
            for c in range(3):
                blurred[..., c] = gaussian_filter(
                    self.rgb[..., c].astype(np.float32), sigma=sigma,
                    mode="nearest").astype(np.uint8)
            levels.append(np.ascontiguousarray(pack_rgb(blurred).T))
        self.n_mips = len(levels)
        self.level_stride = self.sw * self.sh
        flat = [lv.ravel() for lv in levels]
        # A single black pixel at the very end: captured rays are pointed here,
        # which produces the shadow through the same gather as everything else.
        flat.append(np.zeros(1, dtype=np.uint32))
        self.flat = np.concatenate(flat)
        self.black_index = self.flat.size - 1

    def log(self, msg: str) -> None:
        if not self.quiet:
            print(f"[ingest] {msg}")

    # ---------------------------------------------------------------- loading
    def _discover(self) -> str | None:
        for name in sorted(os.listdir(".")):
            if name.lower().endswith(self.FITS_EXT):
                return name
        return None

    def _load(self, path: str | None) -> None:
        target = path or self._discover()
        if target is None or not os.path.exists(target):
            target = self._synthesize_to_disk()
        try:
            self._read_fits(target)
        except Exception as exc:                          # corrupt / unreadable
            self.log(f"could not read {target} ({exc}); synthesizing instead")
            self._read_fits(self._synthesize_to_disk(force=True))

    def _read_fits(self, path: str) -> None:
        """Pull the first HDU holding a >=2-D image; collapse any extra axes."""
        with fits.open(path, memmap=False) as hdul:
            chosen = None
            for hdu in hdul:
                if hdu.data is not None and np.ndim(hdu.data) >= 2:
                    chosen = hdu
                    break
            if chosen is None:
                raise ValueError("no image HDU with >= 2 dimensions")
            arr = np.asarray(chosen.data, dtype=np.float64)
            # Spectral cubes and detector stacks collapse along the leading axes;
            # a median is used because it rejects cosmic rays for free.
            while arr.ndim > 2:
                arr = np.median(arr, axis=0)
            self.header = chosen.header.copy()
            try:
                self.wcs = WCS(self.header).celestial
            except Exception:
                self.wcs = None
            self.source_name = os.path.basename(path)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        self.data = self._fit_to_canvas(arr).astype(np.float32)
        self.log(f"loaded {self.source_name}  shape={self.data.shape}  "
                 f"range=[{self.data.min():.4g}, {self.data.max():.4g}]"
                 + ("  WCS: yes" if self.wcs else "  WCS: none"))

    @staticmethod
    def _fit_to_canvas(arr: np.ndarray) -> np.ndarray:
        """
        Resample an arbitrary survey frame onto the working raster.

        Nearest-neighbour on purpose: this is a display pipeline, and preserving
        the exact pixel VALUES of the science array matters more here than
        interpolation smoothness (a resampled intensity is no longer the
        instrument's measurement).
        """
        th, tw = CFG.sky_h, CFG.sky_w
        h, w = arr.shape
        if (h, w) == (th, tw):
            return arr
        yi = np.linspace(0, h - 1, th).astype(np.int32)
        xi = np.linspace(0, w - 1, tw).astype(np.int32)
        return arr[yi][:, xi]

    # ------------------------------------------------------------ synthesis
    def _synthesize_to_disk(self, force: bool = False) -> str:
        out = self.DEFAULT_MOCK
        if os.path.exists(out) and not force:
            try:
                if fits.getheader(out).get("MOCKVER") == self.MOCK_VERSION:
                    return out
            except Exception:
                pass
        self.log("no FITS input found -- synthesizing a Roman-like wide field")
        data = self.synthesize(CFG.sky_w, CFG.sky_h, CFG.sky_seed)
        hdu = fits.PrimaryHDU(data.astype(np.float32))
        h = hdu.header
        # A plausible tangent-plane WCS so downstream code has real sky coords.
        h["TELESCOP"] = ("ROMAN-SIM", "synthetic wide-field exposure")
        h["INSTRUME"] = ("WFI-SIM", "wide field instrument (simulated)")
        h["FILTER"] = ("F158", "near-infrared band")
        h["BUNIT"] = ("e-/s", "intensity units")
        h["EXPTIME"] = (1200.0, "seconds")
        h["CTYPE1"], h["CTYPE2"] = "RA---TAN", "DEC--TAN"
        h["CRPIX1"], h["CRPIX2"] = CFG.sky_w / 2.0, CFG.sky_h / 2.0
        h["CRVAL1"], h["CRVAL2"] = 265.17500, -28.99200   # toward the galactic centre
        h["CDELT1"], h["CDELT2"] = -0.11 / 3600.0, 0.11 / 3600.0   # 0.11"/px, Roman WFI
        h["CUNIT1"], h["CUNIT2"] = "deg", "deg"
        h["MOCKVER"] = (self.MOCK_VERSION, "roman_lensing_lab mock format version")
        h["COMMENT"] = "Synthesized by roman_lensing_lab.py; not real observations."
        hdu.writeto(out, overwrite=True)
        self.synthetic = True
        self.log(f"wrote {out}  ({CFG.sky_w}x{CFG.sky_h})")
        return out

    @staticmethod
    def synthesize(w: int, h: int, seed: int) -> np.ndarray:
        """Build a scientifically-shaped intensity matrix (see class docstring)."""
        rng = np.random.default_rng(seed)
        img = np.zeros((h, w), dtype=np.float64)

        # -- zodiacal / diffuse background: a smooth low-order gradient --------
        yy, xx = np.mgrid[0:h, 0:w]
        nx, ny = xx / w, yy / h
        img += 12.0 + 4.0 * nx + 3.0 * np.sin(2.4 * ny + 0.7) + 2.0 * np.cos(3.1 * nx)

        # -- the lensing TARGETS ------------------------------------------------
        # A real lensing programme points at a field with a bright source close
        # to the optic axis, because that is the only geometry that produces a
        # full ring.  Three are placed deliberately: one on axis (which maps to
        # a complete Einstein ring) and two slightly offset (which break into
        # the classic arc-plus-counter-image pairs).
        targets = [(0.0, 0.0, 220.0, 30.0, 1.1, 0.05),
                   (46.0, -28.0, 130.0, 20.0, 1.4, 0.35),
                   (-62.0, 40.0, 95.0, 17.0, 2.6, 0.25)]
        for dx, dy, amp, r_eff, n, ellip in targets:
            x0, y0 = w * 0.5 + dx, h * 0.5 + dy
            reach = int(min(max(6.0 * r_eff, 40.0), 320.0))
            xlo, xhi = int(max(x0 - reach, 0)), int(min(x0 + reach, w))
            ylo, yhi = int(max(y0 - reach, 0)), int(min(y0 + reach, h))
            gy, gx = np.mgrid[ylo:yhi, xlo:xhi]
            model = Sersic2D(amplitude=amp / (r_eff ** 0.4), r_eff=r_eff, n=n,
                             x_0=x0, y_0=y0, ellip=ellip,
                             theta=float(rng.uniform(0, math.pi)))
            img[ylo:yhi, xlo:xhi] += np.nan_to_num(
                np.asarray(model(gx, gy), dtype=np.float64), posinf=0.0, nan=0.0)

        # -- field galaxies: Sersic2D, evaluated on a local box each ------------
        # Evaluating each profile only over the box where it is non-negligible
        # turns an O(N_gal * W * H) rasterization into a few milliseconds total.
        for _ in range(CFG.sky_galaxies):
            x0 = rng.uniform(0, w)
            y0 = rng.uniform(0, h)
            n = float(rng.uniform(0.8, 4.2))          # exponential disc -> de Vaucouleurs
            r_eff = float(rng.uniform(7.0, 46.0))
            amp = float(rng.uniform(6.0, 150.0)) / (r_eff ** 0.4)
            ellip = float(rng.uniform(0.0, 0.65))
            theta = float(rng.uniform(0.0, math.pi))
            reach = int(min(max(6.0 * r_eff, 40.0), 320.0))
            xlo, xhi = int(max(x0 - reach, 0)), int(min(x0 + reach, w))
            ylo, yhi = int(max(y0 - reach, 0)), int(min(y0 + reach, h))
            if xhi - xlo < 3 or yhi - ylo < 3:
                continue
            gy, gx = np.mgrid[ylo:yhi, xlo:xhi]
            model = Sersic2D(amplitude=amp, r_eff=r_eff, n=n,
                             x_0=x0, y_0=y0, ellip=ellip, theta=theta)
            patch = np.asarray(model(gx, gy), dtype=np.float64)
            img[ylo:yhi, xlo:xhi] += np.nan_to_num(patch, posinf=0.0, nan=0.0)

        # -- stars: delta functions, brightness from a power-law luminosity fn --
        n_star = CFG.sky_stars
        sx = rng.integers(0, w, n_star)
        sy = rng.integers(0, h, n_star)
        flux = 40.0 * (rng.pareto(1.35, n_star) + 1.0)
        np.add.at(img, (sy, sx), flux)

        # -- instrumental PSF: convolve everything that is light -----------------
        img = gaussian_filter(img, sigma=1.35, mode="nearest")
        # A faint broad halo reproduces the PSF wings real optics always have.
        img += 0.06 * gaussian_filter(img, sigma=9.0, mode="nearest")

        # -- noise: Poisson shot noise, then Gaussian read noise ----------------
        img = rng.poisson(np.clip(img, 0.0, 1e7)).astype(np.float64)
        img += rng.normal(0.0, 3.1, img.shape)
        return np.clip(img, 0.0, None)

    # ------------------------------------------------------------- display
    @staticmethod
    def _colormap() -> np.ndarray:
        """
        256-entry LUT tuned for near-infrared survey imagery: a cold floor so
        empty sky reads as black, warming through the mid-tones, into a white
        core for saturated sources.
        """
        t = np.linspace(0.0, 1.0, 256)
        stops = [(0.00, (2, 3, 7)), (0.22, (9, 14, 34)), (0.44, (34, 40, 82)),
                 (0.62, (118, 76, 106)), (0.78, (208, 132, 88)),
                 (0.91, (248, 204, 146)), (1.00, (255, 253, 246))]
        lut = np.zeros((256, 3))
        for i in range(len(stops) - 1):
            t0, c0 = stops[i]
            t1, c1 = stops[i + 1]
            m = (t >= t0) & (t <= t1)
            f = ((t[m] - t0) / (t1 - t0))[:, None]
            lut[m] = np.array(c0)[None, :] * (1 - f) + np.array(c1)[None, :] * f
        return lut.astype(np.uint8)

    def _to_rgb(self, data: np.ndarray) -> np.ndarray:
        """
        ZScale cuts + asinh stretch -> LUT -> uint8 RGB.

        ZScaleInterval samples the pixel distribution and fits its central slope,
        which is why it survives a frame dominated by empty sky; a naive min/max
        would map everything into the bottom bin.
        """
        try:
            interval = ZScaleInterval(contrast=0.30)
            vmin, vmax = interval.get_limits(data)
            if not np.isfinite([vmin, vmax]).all() or vmax <= vmin:
                raise ValueError
        except Exception:
            vmin, vmax = PercentileInterval(99.2).get_limits(data)
            if vmax <= vmin:
                vmin, vmax = float(np.min(data)), float(np.max(data)) + 1.0
        # The interval and stretch are applied directly rather than through
        # ImageNormalize, which is a matplotlib adapter and would drag in a
        # dependency this tool does not otherwise need.
        scaled = (np.asarray(data, dtype=np.float64) - vmin) / max(vmax - vmin, 1e-12)
        np.clip(scaled, 0.0, 1.0, out=scaled)
        scaled = np.asarray(AsinhStretch(a=0.12)(scaled, clip=True))
        scaled = np.nan_to_num(np.clip(scaled, 0.0, 1.0), nan=0.0)
        idx = (scaled * 255.0).astype(np.uint8)
        return self._colormap()[idx]

    # ------------------------------------------------------------ metadata
    def describe(self) -> list[tuple[str, str]]:
        rows = [("Source", self.source_name or "(none)"),
                ("Array", f"{self.data.shape[1]} x {self.data.shape[0]}")]
        for key, label in (("TELESCOP", "Telescope"), ("INSTRUME", "Instrument"),
                           ("FILTER", "Filter"), ("EXPTIME", "Exposure")):
            if key in self.header:
                rows.append((label, str(self.header[key])))
        try:
            scale = abs(float(self.header.get("CDELT1", 0.0))) * 3600.0
            if scale > 0:
                rows.append(("Pixel scale", f"{scale:.3f} arcsec/px"))
        except Exception:
            pass
        return rows

    def sky_coord(self, px: float, py: float) -> str:
        """RA/Dec under a source-plane pixel, for the HUD readout."""
        if self.wcs is None:
            return "no WCS"
        try:
            sky = self.wcs.pixel_to_world(px, py)
            return f"RA {sky.ra.deg:8.4f}  Dec {sky.dec.deg:+8.4f}"
        except Exception:
            return "no WCS"


# ==============================================================================
#  SECTION 8 -- THE RAY-TRACING RENDERER
#
#  This is the hot loop of the whole instrument.  Every frame it answers, for
#  every pixel of the viewport, "where in the source image does this light come
#  from?" -- and it has ~16 ms to do it for a million of them.
#
#  Three decisions make that possible:
#
#  1. SEPARATION OF WHAT CHANGES.  The lens mapping depends on mass, spin, depth
#     and camera -- none of which change on a typical frame.  So the source-plane
#     coordinates are computed once into a cache and reused; only the cosmological
#     scale factor, which is a single scalar multiply, varies per frame.
#
#  2. PACKED PIXELS.  The source image is stored as one uint32 per pixel in
#     exactly the layout of a 32-bit pygame surface.  The per-frame work is then
#     a single contiguous integer gather instead of three strided byte gathers,
#     measured at ~5x faster, and blits with no conversion.
#
#  3. DECIMATED RASTER.  The lensed field is smooth except at the critical curve,
#     so it is traced at 1/N resolution and upscaled by the (C-speed) blitter.
#
#  The captured region is handled without a boolean mask: a single black pixel is
#  appended to the source array and rays with b < b_c are simply pointed at it.
#  The shadow is then produced by the same gather as everything else.
# ==============================================================================

class LensRenderer:
    """Inverse ray tracer: source-plane image -> lensed image plane."""

    def __init__(self, sky: SkySurvey, lens: LensModel, viewport: pygame.Rect):
        self.sky = sky
        self.lens = lens
        self.viewport = viewport
        self.ds = max(1, CFG.lens_downsample)
        self.w = max(2, viewport.width // self.ds)
        self.h = max(2, viewport.height // self.ds)

        self.raster = pygame.Surface((self.w, self.h)).convert(32, 0)
        self.scaled = pygame.Surface((viewport.width, viewport.height)).convert(32, 0)

        self._key: tuple | None = None
        self.src_x = np.zeros((self.w, self.h), dtype=np.float32)
        self.src_y = np.zeros((self.w, self.h), dtype=np.float32)
        self.captured = np.zeros((self.w, self.h), dtype=bool)
        self.mu = np.ones((self.w, self.h), dtype=np.float64)
        self.mip_offset = np.zeros((self.w, self.h), dtype=np.int32)
        self._idx = np.zeros((self.w, self.h), dtype=np.int32)
        self._mu_table = np.ones(2, dtype=np.float64)
        self.distortion_packed = np.zeros((self.w, self.h), dtype=np.uint32)
        self.ring_mask = np.zeros((self.w, self.h), dtype=bool)
        self.ring_line = np.zeros((self.w, self.h), dtype=bool)
        self.einstein_radius = 0.0
        self.arc_contrast = 0.0
        self._out = np.zeros((self.w, self.h), dtype=np.uint32)
        self._distortion_dirty = True

        # World offsets of every raster sample from the viewport centre, in
        # (W, H) order so pygame.surfarray needs no transpose anywhere.
        px = (np.arange(self.w, dtype=np.float64) + 0.5) * self.ds
        py = (np.arange(self.h, dtype=np.float64) + 0.5) * self.ds
        self._px = px - viewport.width * 0.5
        self._py = py - viewport.height * 0.5

    # ------------------------------------------------------------- map cache
    def _cache_key(self, camera: "Camera") -> tuple:
        """
        Quantized to the precision that is actually VISIBLE on screen.

        This matters more than it looks: the Penrose process spins the hole down
        by ~1e-6 per frame, so a key quantized to raw float precision would
        invalidate the cache on almost every frame and rebuild the whole
        deflection field -- several milliseconds of work for a change no pixel
        could show.  Three decimals of spin is far below one pixel of twist.
        """
        h = self.lens.hole
        return (round(h.mass, 3), round(h.spin, 3), round(self.lens.depth, 2),
                round(camera.scale, 4), round(float(camera.center[0]), 1),
                round(float(camera.center[1]), 1))

    def ensure_map(self, camera: "Camera") -> bool:
        """Rebuild the cached deflection field if any governing parameter moved."""
        key = self._cache_key(camera)
        if key == self._key:
            return False
        self._key = key
        self.lens.sync()

        # Screen -> world offsets from the black hole (which sits at the world
        # origin).  Dividing by the zoom keeps the lensing physical rather than
        # pinned to the display raster.
        # Kept as a (W, 1) column and a (1, H) row: build_field broadcasts them
        # itself, so the full 2-D coordinate rasters are never materialized.
        wx = (self._px[:, None] / camera.scale
              + camera.center[0]).astype(np.float32)
        wy = (-self._py[None, :] / camera.scale
              + camera.center[1]).astype(np.float32)   # screen y is down

        (self.src_x, self.src_y, self.captured, self._idx, self._mu_table,
         level) = self.lens.build_field(wx, wy,
                                        self.ds / max(camera.scale, 1e-6),
                                        self.sky.n_mips)
        self.mip_offset = level * np.int32(self.sky.level_stride)
        self._distortion_dirty = True
        self.einstein_radius = self.lens.einstein_radius()
        r = np.hypot(wx, wy)
        # A photometric aperture around the critical curve, not a one-pixel line:
        # this is the annulus a real arc-detection pipeline would integrate over.
        band = max(0.10 * self.einstein_radius, 4.0 * self.ds / camera.scale)
        self.ring_mask = np.abs(r - self.einstein_radius) < band
        # A separate, thin mask for DRAWING the critical curve: the photometric
        # aperture above is deliberately wide, and painting it would render the
        # Einstein ring as a fat band rather than the curve it actually is.
        self.ring_line = np.abs(r - self.einstein_radius) < \
            max(1.5 * self.ds / camera.scale, 1.0)
        return True

    def _bake_distortion_map(self) -> None:
        """
        False-colour magnification map: the "pure gravitational distortion"
        product, and the only thing visible under deep-underground shielding.

        log10|mu| is the natural display variable because magnification spans
        many decades and formally diverges on the critical curve.  Parity is
        encoded in the hue: images inside the critical curve are inverted
        (mu < 0) and are drawn cool, images outside are drawn warm -- so the
        Einstein ring appears as the boundary between the two, which is exactly
        what it is.
        """
        # |log10|mu|| = 0 means "undistorted", so undisturbed sky maps to black
        # and the map shows only where the geometry actually does something.
        # The ABSOLUTE value is taken so demagnified regions (|mu| < 1, inside
        # the critical curve) are as visible as magnified ones -- both are
        # distortion, and hiding one half would misrepresent the field.
        # The magnification is gathered here rather than during the rebuild:
        # this view is optional, and most sessions never open it.
        self.mu = self._mu_table[self._idx]
        mag = np.abs(np.log10(np.abs(self.mu) + 1e-6))
        t = np.clip(mag / 1.6, 0.0, 1.0)
        cool = np.stack([0.18 + 0.10 * t, 0.42 + 0.45 * t, 0.55 + 0.45 * t], axis=-1)
        warm = np.stack([0.55 + 0.45 * t, 0.30 + 0.55 * t, 0.16 + 0.30 * t], axis=-1)
        inverted = (self.mu < 0.0)[..., None]
        rgb = np.where(inverted, cool, warm) * (0.10 + 0.90 * t[..., None])
        rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        rgb[self.captured] = 0
        rgb[self.ring_line] = np.array(Palette.EINSTEIN, dtype=np.uint8)
        self.distortion_packed = pack_rgb(rgb)

    # ----------------------------------------------------------- the gather
    def trace(self, camera: "Camera", cosmos: QuintessenceField,
              bubble: "WarpBubble | None", distortion_mode: bool) -> pygame.Surface:
        """Produce the lensed frame.  One integer gather plus one upscale."""
        self.ensure_map(camera)

        if distortion_mode:
            # Shielded: no electromagnetic channel at all, so the sky is never
            # sampled.  Only the geometry is shown.  The false-colour map is
            # baked lazily, so users who never open this view never pay for it.
            if self._distortion_dirty:
                self._bake_distortion_map()
                self._distortion_dirty = False
            pygame.surfarray.blit_array(self.raster, self.distortion_packed)
            return self._present()

        sky = self.sky
        # Dark energy stretches the SOURCE plane: comoving separations grow as
        # a(t), so background galaxies physically recede from one another.
        inv_a = 1.0 / max(cosmos.scale_factor, 1e-6)
        sx = self.src_x * inv_a + sky.sw * 0.5
        sy = self.src_y * inv_a + sky.sh * 0.5

        if bubble is not None and bubble.active:
            self._apply_warp(sx, sy, bubble, camera, inv_a)

        # Sampling coordinates that fall outside the survey frame are MIRRORED
        # back rather than clamped.  Clamping smears the edge row across every
        # ray that overshoots, which shows up as vertical streaks whenever the
        # camera pans or the lens pushes samples off-frame; reflection continues
        # the field seamlessly (no hard tile seam) and costs three extra ops.
        self._reflect(sx, sky.sw)
        self._reflect(sy, sky.sh)
        # flat index into the (W, H)-ordered source array: x * sh + y
        idx = sx.astype(np.int32)
        idx *= sky.sh
        idx += sy.astype(np.int32)
        idx += self.mip_offset                        # texture filtering level
        idx[self.captured] = sky.black_index          # the shadow, b < b_c

        np.take(sky.flat, idx, out=self._out, mode="clip")
        pygame.surfarray.blit_array(self.raster, self._out)
        return self._present()

    @staticmethod
    def _reflect(a: np.ndarray, n: int) -> None:
        """In-place triangle-wave fold of `a` into [0, n): a seamless mirror tile."""
        # Triangle wave: fold into [0, 2n), then reflect the upper half down.
        #   t = a mod 2n            in [0, 2n)
        #   r = n - |t - n|         in [0, n]
        # The final minimum keeps r strictly below n so the int cast stays in
        # bounds; it must be applied AFTER the reflection, not folded into it.
        #
        # The modulo is done as x - floor(x) on the scaled coordinate rather than
        # with np.mod: for float32 inputs np.mod takes a slow fmod path and is
        # measured at 5.2 ms per axis here versus 0.38 ms for the floor form --
        # a 14x difference on the frame budget for bit-identical results.
        inv = 1.0 / (2.0 * n)
        np.multiply(a, inv, out=a)
        np.subtract(a, np.floor(a), out=a)          # fractional part, in [0, 1)
        np.multiply(a, 2.0 * n, out=a)              # t, in [0, 2n)
        np.subtract(a, n, out=a)
        np.abs(a, out=a)
        np.subtract(n, a, out=a)
        np.minimum(a, n - 1e-3, out=a)

    def _apply_warp(self, sx: np.ndarray, sy: np.ndarray, bubble: "WarpBubble",
                    camera: "Camera", inv_a: float) -> None:
        """
        Add the Alcubierre shift vector to the sampling coordinates.

        Only the raster block covering the bubble is touched: the shape function
        is exponentially small outside its wall, so evaluating it over the whole
        frame would be almost entirely wasted work.  The displacement is
        proportional to f(r_s), and because f falls off across the wall, samples
        near the nose are advected less than the bubble centre -- the image
        bunches up ahead and rarefies behind, which is the shift vector's doing
        rather than a separately authored effect.
        """
        cx, cy = camera.to_screen(bubble.center[None, :])[0]
        reach_px = bubble.radius * camera.scale * 2.2
        i0 = max(int((cx - reach_px) / self.ds), 0)
        i1 = min(int((cx + reach_px) / self.ds) + 1, self.w)
        j0 = max(int((cy - reach_px) / self.ds), 0)
        j1 = min(int((cy + reach_px) / self.ds) + 1, self.h)
        if i1 <= i0 or j1 <= j0:
            return

        wx = self._px[i0:i1, None] / camera.scale + camera.center[0]
        wy = -self._py[None, j0:j1] / camera.scale + camera.center[1]
        dx = wx - bubble.center[0]
        dy = wy - bubble.center[1]
        f = bubble.f(np.sqrt(dx * dx + dy * dy))
        amp = bubble.pixel_gain() * inv_a
        sx[i0:i1, j0:j1] += (amp * bubble.heading[0]) * f
        sy[i0:i1, j0:j1] -= (amp * bubble.heading[1]) * f    # world +y is screen -y

    def _present(self) -> pygame.Surface:
        if self.ds == 1:
            return self.raster
        if CFG.smooth_upscale:
            pygame.transform.smoothscale(self.raster, self.scaled.get_size(),
                                         self.scaled)
        else:
            pygame.transform.scale(self.raster, self.scaled.get_size(), self.scaled)
        return self.scaled

    # ------------------------------------------------------ SNR diagnostics
    def measure_arc_contrast(self) -> float:
        """
        RMS luminance contrast in the annulus around the critical curve.

        This is the actual observable a lensing survey works with: the arcs are a
        faint modulation on top of the sky, and whether they are detectable is a
        contrast-versus-noise question, not a "can you see them" question.
        Measured on a subsample of the traced raster, so it costs almost nothing.
        """
        if self.einstein_radius <= 0.0:
            return 0.0
        sub = self._out[::2, ::2]
        m = self.ring_mask[::2, ::2]
        if not m.any():
            return 0.0
        v = sub[m]
        lum = (((v >> 16) & 255) * 0.299 + ((v >> 8) & 255) * 0.587 +
               (v & 255) * 0.114)
        self.arc_contrast = float(np.std(lum))
        return self.arc_contrast


# ==============================================================================
#  SECTION 9 -- CAMERA
# ==============================================================================

class Camera:
    """World <-> screen mapping.  World +y is up; screen +y is down."""

    def __init__(self, viewport: pygame.Rect, scale: float = 1.0):
        self.viewport = viewport
        self.scale = scale
        self.center = np.zeros(2, dtype=np.float64)

    @property
    def origin(self) -> np.ndarray:
        return np.array([self.viewport.width * 0.5, self.viewport.height * 0.5])

    def to_screen(self, pts: np.ndarray) -> np.ndarray:
        rel = (np.asarray(pts, dtype=np.float64) - self.center) * self.scale
        out = np.empty_like(rel)
        o = self.origin
        out[..., 0] = o[0] + rel[..., 0]
        out[..., 1] = o[1] - rel[..., 1]
        return out

    def to_world(self, sx: float, sy: float) -> np.ndarray:
        o = self.origin
        return np.array([(sx - o[0]) / self.scale + self.center[0],
                         (o[1] - sy) / self.scale + self.center[1]])

    def zoom_at(self, sx: float, sy: float, factor: float) -> None:
        anchor = self.to_world(sx, sy)
        self.scale = float(np.clip(self.scale * factor, 0.25, 4.0))
        self.center += anchor - self.to_world(sx, sy)


# ==============================================================================
#  SECTION 10 -- THE ALCUBIERRE WARP BUBBLE
#
#  Alcubierre (1994), Class. Quantum Grav. 11, L73:
#
#      ds^2 = -dt^2 + (dx - v_s(t) f(r_s) dt)^2 + dy^2 + dz^2
#      f(r_s) = [tanh(sigma(r_s + R)) - tanh(sigma(r_s - R))] / (2 tanh(sigma R))
#
#  Two properties drive everything this class does:
#
#  (a) INSIDE, f = 1 identically, so its gradient vanishes and the region is
#      Riemann-flat.  The ship is in free fall in Minkowski space: zero proper
#      acceleration, zero tidal force, and a clock running at the distant rate.
#      That is why a warping ship can sit beside a black hole and not be
#      spaghettified -- the tidal field is outside its wall.
#
#  (b) The expansion scalar of the normal volume elements (the York time) is
#
#          theta = v_s (x_s / r_s) df/dr_s
#
#      Since df/dr_s < 0 everywhere, theta < 0 AHEAD of the ship (space
#      contracted) and theta > 0 BEHIND it (space expanded).  The renderer reads
#      exactly this quantity for its colour, so what is drawn is the metric's
#      own York time, not an artistic impression.
#
#  The ship never moves through its local space; the bubble carries it.  Its
#  coordinate speed v_s is therefore unbounded by c while its local speed is
#  identically zero -- no causality violation in the local frame.
# ==============================================================================

class WarpBubble:
    PIXEL_GAIN = 150.0        # source-plane displacement scale, in source pixels

    def __init__(self, radius: float = 86.0, sigma: float = 0.055):
        self.radius = radius
        self.sigma = sigma
        self.active = False
        self.center = np.zeros(2, dtype=np.float64)
        self.heading = np.array([1.0, 0.0], dtype=np.float64)
        self.v_s = 0.0
        self._norm = math.tanh(self.sigma * self.radius)

    def configure(self, center: np.ndarray, heading: np.ndarray, v_s: float) -> None:
        self.center = np.asarray(center, dtype=np.float64)
        h = np.asarray(heading, dtype=np.float64)
        n = float(norm(h))
        self.heading = h / n if n > CFG.eps else np.array([1.0, 0.0])
        self.v_s = float(v_s)
        self._norm = math.tanh(self.sigma * self.radius)

    # ------------------------------------------------------- shape function
    def f(self, r_s: np.ndarray) -> np.ndarray:
        """1 inside the bubble, 0 outside, smooth across the wall."""
        s, R = self.sigma, self.radius
        return (np.tanh(s * (r_s + R)) - np.tanh(s * (r_s - R))) / (2.0 * self._norm)

    def df(self, r_s: np.ndarray) -> np.ndarray:
        """df/dr_s.  Negative everywhere -- the source of the sign of theta."""
        s, R = self.sigma, self.radius
        sech2 = lambda z: 1.0 / np.cosh(np.clip(z, -30.0, 30.0)) ** 2
        return s * (sech2(s * (r_s + R)) - sech2(s * (r_s - R))) / (2.0 * self._norm)

    def pixel_gain(self) -> float:
        """
        Saturated displacement amplitude for the image warp.

        The SIGN and PROFILE come from the metric; only the magnitude is
        saturated through tanh(v_s), because an unbounded shift would drag the
        sampling coordinates clear across the source image at warp 3 and fold
        the picture over itself, destroying the very compression it should show.
        """
        return self.PIXEL_GAIN * math.tanh(self.v_s / 1.5)

    # ------------------------------------------------------------ field maps
    def interior_fraction(self, pos: np.ndarray) -> np.ndarray:
        if not self.active:
            return np.zeros(pos.shape[0])
        return self.f(norm(pos - self.center[None, :]))

    def exterior_mask(self, pos: np.ndarray) -> np.ndarray:
        """
        (1 - f) as an (N, 1) multiplier on every external acceleration.  This is
        the mathematical statement of fact (a): the deeper inside the bubble, the
        more completely the outside universe's curvature is screened away.
        """
        if not self.active:
            return np.ones((pos.shape[0], 1))
        return (1.0 - self.interior_fraction(pos))[:, None]

    def expansion_scalar(self, pos: np.ndarray) -> np.ndarray:
        """theta = v_s (x_s/r_s) df/dr_s.  < 0 ahead (contraction), > 0 behind."""
        if not self.active:
            return np.zeros(pos.shape[0])
        rel = pos - self.center[None, :]
        r = np.maximum(norm(rel), CFG.eps)
        x_s = rel @ self.heading
        return self.v_s * (x_s / r) * self.df(r)


# ==============================================================================
#  SECTION 11 -- THE COMPOSED FORCE LAW
#
#  Every massive object -- planets, stars, dark matter, the spacecraft -- obeys
#  one acceleration function, so momentum bookkeeping stays consistent between
#  subsystems.  Four contributions superpose:
#
#  (1) CENTRAL HOLE.  Paczynski & Wiita (1980) pseudo-Newtonian potential
#
#          Phi(r) = -GM / (r - r_s)      =>      g = -GM / (r - r_s)^2
#
#      This is the standard relativistic surrogate: it reproduces the
#      Schwarzschild marginally stable orbit at exactly 6M and the marginally
#      bound orbit at exactly 4M, and diverges at the horizon so nothing can
#      hover there.  A plain 1/r^2 law has no ISCO and no plunge region at all.
#
#  (2) FRAME DRAGGING.  Weak-field gravitomagnetism.  In the equatorial plane
#      J is perpendicular to the plane, so J.rhat = 0 and
#
#          B_g  = -(2 G J / c^3 r^3) zhat
#          a_LT = -v x B_g = (2J/r^3)(v_y, -v_x)
#
#      An infalling particle is swept tangentially along the spin: the whirlpool.
#
#  (3) DARK ENERGY.  a_vac = H^2 r, outward.  See QuintessenceField.
#
#  (4) N-BODY.  Plummer-softened Newtonian gravity between spawned bodies,
#      vectorized as an (N, K) pair tensor.
#
#  RELATIVISTIC DYNAMICS.  The engine integrates the true law dp/dt = F with
#  p = gamma m v, whose velocity form is
#
#      dv/dt = (1/gamma) [ a - (v.a) v / c^2 ]
#
#  The subtracted longitudinal term is what makes c an asymptote: as |v| -> 1 the
#  component of acceleration along v is suppressed by 1/gamma^3 while the
#  transverse component is suppressed only by 1/gamma.  A slingshot can therefore
#  be pushed to 0.999 c and never past it, with no artificial speed clamp.
# ==============================================================================

class GravityField:
    def __init__(self, hole: BlackHole, cosmos: QuintessenceField):
        self.hole = hole
        self.cosmos = cosmos
        self.src_pos = np.zeros((0, 2), dtype=np.float64)
        self.src_mass = np.zeros((0,), dtype=np.float64)
        self._src_soft2 = np.zeros((0,), dtype=np.float64)
        self.bubble: WarpBubble | None = None

    def set_sources(self, pos: np.ndarray, mass: np.ndarray, soft: np.ndarray) -> None:
        self.src_pos = pos
        self.src_mass = mass
        self._src_soft2 = soft * soft

    def central_acceleration(self, pos: np.ndarray) -> np.ndarray:
        hole = self.hole
        d = pos - hole.position
        r = np.maximum(norm(d, keepdims=True), CFG.eps)
        unit = d / r
        # Paczynski-Wiita is singular at r_s, so keep the denominator positive.
        denom = np.maximum(r - hole.r_s, 0.35 * hole.r_s)
        return unit * (-hole.mass / (denom * denom)) + self.cosmos.vacuum_acceleration(pos)

    def frame_drag_acceleration(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        J = self.hole.angular_momentum
        if abs(J) <= CFG.eps:
            return np.zeros_like(pos)
        d = pos - self.hole.position
        r = np.maximum(norm(d, keepdims=True), 0.5 * self.hole.r_s)
        swirl = np.stack([vel[..., 1], -vel[..., 0]], axis=-1)
        return (2.0 * J / (r ** 3)) * swirl

    def nbody_acceleration(self, pos: np.ndarray) -> np.ndarray:
        """
        Vectorized Plummer-softened pairwise gravity.

        The squared separation is formed with einsum so no second (N, K, 2)
        temporary is allocated, and r^-3 is built from a reciprocal plus a sqrt
        because np.power(x, -1.5) is several times slower than either.
        """
        if self.src_mass.size == 0 or pos.size == 0:
            return np.zeros_like(pos)
        diff = self.src_pos[None, :, :] - pos[:, None, :]
        r2 = np.einsum("nkc,nkc->nk", diff, diff)
        inv = 1.0 / (r2 + self._src_soft2[None, :] + CFG.eps)
        w = self.src_mass[None, :] * inv * np.sqrt(inv)
        return np.einsum("nk,nkc->nc", w, diff)

    def acceleration(self, pos: np.ndarray, vel: np.ndarray,
                     include_nbody: bool = True) -> np.ndarray:
        acc = self.central_acceleration(pos)
        acc += self.frame_drag_acceleration(pos, vel)
        if include_nbody:
            acc += self.nbody_acceleration(pos)
        return acc

    @staticmethod
    def proper_accel(vel: np.ndarray, acc: np.ndarray) -> np.ndarray:
        """dv/dt = (1/gamma)[a - (v.a)v], exact for dp/dt = F with p = gamma m v."""
        g = lorentz_gamma(vel)[..., None]
        return (acc - np.sum(vel * acc, axis=-1, keepdims=True) * vel) / g

    def make_derivative(self, include_nbody: bool = True,
                        relativistic: bool = True):
        """Return f(state, t) -> dstate/dt for state = [x, y, vx, vy]."""
        def derivative(state: np.ndarray, t: float) -> np.ndarray:
            pos = state[..., 0:2]
            vel = state[..., 2:4]
            acc = self.acceleration(pos, vel, include_nbody=include_nbody)
            if self.bubble is not None and self.bubble.active:
                # Inside an Alcubierre bubble the interior is Riemann-flat, so
                # external tidal acceleration is screened by (1 - f).
                acc = acc * self.bubble.exterior_mask(pos)
            if relativistic:
                acc = self.proper_accel(vel, acc)
            return np.concatenate([vel, acc], axis=-1)
        return derivative


# ==============================================================================
#  SECTION 12 -- ATOMIC EQUILIBRIUM AND THE COLLAPSE LADDER
#
#  A cold self-gravitating sphere is a standing argument between two pressures.
#
#  INWARD -- gravitational compression.  Integrating hydrostatic equilibrium
#  dP/dr = -G m(r) rho / r^2 for a uniform sphere gives the central pressure
#
#      P_grav = 3 G M^2 / (8 pi R^4)      ~ M^2 / R^4
#
#  OUTWARD -- electromagnetic / degeneracy resistance.  At planetary densities
#  this is Coulomb pressure: electron clouds refuse to interpenetrate because
#  the Pauli principle forbids two electrons the same state.  As a polytrope it
#  is non-relativistic electron degeneracy
#
#      P_deg = K rho^(5/3)                ~ (M/R^3)^(5/3)
#
#  Balancing the two gives the celebrated result that a degenerate body SHRINKS
#  when you feed it, R ~ M^(-1/3).  Interpolating between the incompressible
#  low-mass branch (R ~ M^(1/3)) and that degenerate branch is Zapolsky &
#  Salpeter (1969):
#
#      R(M) = C M^(1/3) / [ 1 + (M/M_0)^(4/3) ]
#
#  which peaks at M = M_0 / 3^(3/4).  With M_0 = 8 M_J the maximum radius lands
#  near 3.5 M_J -- exactly where real gas giants top out, which is why Jupiter,
#  Saturn and a 50 M_J brown dwarf are all very nearly the same size.
#
#  As electrons are forced relativistic the polytropic index softens from 5/3 to
#  4/3, pressure support stops growing with mass, and the argument is lost:
#
#      > 13 M_J     deuterium burning              -> BROWN DWARF
#      > 80 M_J     sustained p-p hydrogen fusion  -> STAR
#      > 1508 M_J   Chandrasekhar limit, 1.44 Msun -> DEGENERATE CORE
#      > 3143 M_J   TOV limit, ~3 Msun             -> BLACK HOLE (horizon forms)
# ==============================================================================

class Phase:
    PLANET, GAS_GIANT, BROWN_DWARF, STAR, DEGENERATE, COLLAPSED = range(6)
    NAMES = {0: "Planet", 1: "Gas Giant", 2: "Brown Dwarf",
             3: "Star", 4: "Degenerate Core", 5: "Black Hole"}
    COLORS = {0: Palette.ROCK, 1: Palette.PLANET, 2: Palette.BROWN_DWARF,
              3: Palette.STAR, 4: (222, 236, 255), 5: (10, 8, 16)}


class StellarStructure:
    M_DEUTERIUM = 13.0
    M_HYDROGEN = 80.0
    M_JUP_PER_SUN = 1047.57
    M_CHANDRA = 1.44 * M_JUP_PER_SUN          # 1508 M_J
    M_TOV = 3.00 * M_JUP_PER_SUN              # 3143 M_J
    M_ZS = 8.0                                 # radius peaks at M_ZS / 3^(3/4)
    R_JUP_PX = 11.0
    # Calibrated so a body collapsing at M_TOV acquires geometric mass ~25,
    # essentially equal to the default central hole -- a real gravitational rival.
    GEOM_PER_MJ = 0.008

    @staticmethod
    def phase_of(mass_mj: np.ndarray) -> np.ndarray:
        m = np.asarray(mass_mj, dtype=np.float64)
        ph = np.zeros(m.shape, dtype=np.int32)
        ph = np.where(m > 0.4, Phase.GAS_GIANT, ph)
        ph = np.where(m > StellarStructure.M_DEUTERIUM, Phase.BROWN_DWARF, ph)
        ph = np.where(m > StellarStructure.M_HYDROGEN, Phase.STAR, ph)
        ph = np.where(m > StellarStructure.M_CHANDRA, Phase.DEGENERATE, ph)
        ph = np.where(m > StellarStructure.M_TOV, Phase.COLLAPSED, ph)
        return ph

    @staticmethod
    def physical_radius(mass_mj: np.ndarray) -> np.ndarray:
        """Zapolsky-Salpeter R(M) in Jupiter radii: rises as M^1/3, then falls."""
        m = np.maximum(np.asarray(mass_mj, dtype=np.float64), 1e-6)
        return np.cbrt(m) / (1.0 + np.power(m / StellarStructure.M_ZS, 4.0 / 3.0))

    @staticmethod
    def display_radius(mass_mj: np.ndarray, phase: np.ndarray) -> np.ndarray:
        """
        Screen radius in world units.

        The true relation is kept on the planetary branch, so the "stops growing
        and starts shrinking" behaviour is directly visible.  Fusing objects are
        re-inflated on a log law because a real star is ~10x a gas giant while a
        neutron star is ~1e-5 of one -- unrenderable at any single linear scale.
        This is a DISPLAY map only; every force uses geometric mass, never this.
        """
        m = np.maximum(np.asarray(mass_mj, dtype=np.float64), 1e-6)
        ph = np.asarray(phase)
        r = StellarStructure.physical_radius(m) * StellarStructure.R_JUP_PX
        star_r = 13.0 + 7.0 * np.log10(1.0 + m / StellarStructure.M_HYDROGEN)
        r = np.where(ph >= Phase.BROWN_DWARF, np.maximum(r, star_r * 0.72), r)
        r = np.where(ph >= Phase.STAR, star_r, r)
        r = np.where(ph == Phase.DEGENERATE, 6.0, r)
        # A collapsed body is drawn at its own Schwarzschild radius r_s = 2M.
        r = np.where(ph == Phase.COLLAPSED,
                     np.clip(2.0 * m * StellarStructure.GEOM_PER_MJ, 7.0, 90.0), r)
        return np.maximum(r, 2.0)

    @staticmethod
    def geometric_mass(mass_mj: np.ndarray) -> np.ndarray:
        return np.asarray(mass_mj, dtype=np.float64) * StellarStructure.GEOM_PER_MJ

    @staticmethod
    def pressure_balance(mass_mj: float) -> tuple[float, float]:
        """
        (compression, resistance) on a common scale, for the equilibrium gauge.

        Their RATIO is what matters, and it crosses unity exactly where the
        Zapolsky-Salpeter radius turns over.  Above the Chandrasekhar mass the
        electron gas is relativistic, the index softens to 4/3, and resistance
        can no longer keep up at any radius -- the gauge pegs.
        """
        m = max(float(mass_mj), 1e-6)
        r = float(StellarStructure.physical_radius(m))
        p_grav = m * m / (r ** 4)
        index = 5.0 / 3.0 if m < StellarStructure.M_CHANDRA else 4.0 / 3.0
        p_deg = (m / (r ** 3)) ** index
        return p_grav, p_deg


# ==============================================================================
#  SECTION 13 -- THE BODY POPULATION  (structure-of-arrays particle engine)
#
#  Bodies live in parallel NumPy arrays rather than a list of Python objects, so
#  one RK4 step costs four vectorized derivative evaluations for the whole
#  population instead of 4N interpreted calls.
#
#  SPAGHETTIFICATION.  In the local frame of an infalling body the geodesic
#  deviation equation gives, exactly,
#
#      radial   (stretch)  d a_r = +2 G M dr / r^3
#      transverse (squeeze) d a_t = -  G M dr / r^3
#
#  The factor of two and the opposite signs ARE the phenomenon: the body is drawn
#  into a filament along the radial direction while being pinched across it.
#  `strain` integrates 2GM/r^3, and the render stretches by k while shrinking by
#  1/sqrt(k) -- the 2:-1 ratio the metric dictates, which preserves area.
# ==============================================================================

class BodyPopulation:
    # Strain runs 0 -> 1 over the disruption.  TIDAL_GAIN sets the presentation
    # timescale; MAX_STRAIN_STEP floors the render loop at 50 steps so the
    # filament is always watchable, whatever dt the user has dialled in.
    TIDAL_GAIN = 26.0
    MAX_STRAIN_STEP = 0.020

    def __init__(self, gravity: GravityField, trail_len: int = CFG.trail_len):
        self.gravity = gravity
        self.trail_len = trail_len
        self.integrator = RK4Integrator(gravity.make_derivative(True, True))
        self._alloc(0)

    def _alloc(self, n: int) -> None:
        self.state = np.zeros((n, 4), dtype=np.float64)
        self.mass_mj = np.zeros(n, dtype=np.float64)
        self.phase = np.zeros(n, dtype=np.int32)
        self.radius = np.zeros(n, dtype=np.float64)
        self.geom = np.zeros(n, dtype=np.float64)
        self.strain = np.zeros(n, dtype=np.float64)
        self.trail = np.zeros((n, self.trail_len, 2), dtype=np.float32)
        self.trail_head = 0

    @property
    def count(self) -> int:
        return self.state.shape[0]

    # -------------------------------------------------------------- spawning
    def spawn(self, pos: Sequence[float], vel: Sequence[float], mass_mj: float) -> None:
        if self.count >= CFG.max_bodies:
            return
        st = np.array([[pos[0], pos[1], vel[0], vel[1]]], dtype=np.float64)
        self.state = np.concatenate([self.state, st])
        self.mass_mj = np.concatenate([self.mass_mj, [float(mass_mj)]])
        self.phase = np.concatenate([self.phase, np.zeros(1, dtype=np.int32)])
        self.radius = np.concatenate([self.radius, np.zeros(1)])
        self.geom = np.concatenate([self.geom, np.zeros(1)])
        self.strain = np.concatenate([self.strain, np.zeros(1)])
        self.trail = np.concatenate(
            [self.trail, np.repeat(st[:, None, 0:2].astype(np.float32),
                                   self.trail_len, axis=1)])
        self.refresh_structure()

    def spawn_orbiting(self, hole: BlackHole, radius: float, mass_mj: float,
                       angle: float | None = None, prograde: bool = True) -> None:
        """Circular orbit using the speed that closes under the RELATIVISTIC law."""
        ang = np.random.uniform(0.0, math.tau) if angle is None else angle
        r = max(radius, hole.r_s * 1.6)
        pos = hole.position + np.array([math.cos(ang), math.sin(ang)]) * r
        v = min(float(hole.circular_speed(r)), 0.92)
        tangent = np.array([-math.sin(ang), math.cos(ang)])
        if not prograde:
            tangent = -tangent
        self.spawn(pos, tangent * v, mass_mj)

    def clear(self) -> None:
        self._alloc(0)

    # ------------------------------------------------------------- structure
    def refresh_structure(self) -> None:
        if self.count == 0:
            return
        self.phase = StellarStructure.phase_of(self.mass_mj)
        self.radius = StellarStructure.display_radius(self.mass_mj, self.phase)
        self.geom = StellarStructure.geometric_mass(self.mass_mj)

    def add_mass_at(self, world_pos: np.ndarray, amount_mj: float,
                    grab_radius: float = 26.0) -> int:
        """Pour mass into the nearest body within reach.  Returns its index or -1."""
        if self.count == 0:
            return -1
        d = norm(self.state[:, 0:2] - np.asarray(world_pos, float)[None, :]) - self.radius
        idx = int(np.argmin(d))
        if d[idx] > grab_radius:
            return -1
        self.mass_mj[idx] = min(self.mass_mj[idx] + amount_mj, CFG.max_body_mj)
        self.refresh_structure()
        return idx

    # ---------------------------------------------------------------- physics
    def step(self, t: float, dt: float, substeps: int) -> None:
        if self.count == 0 or dt == 0.0:
            return
        self.state = self.integrator.integrate(self.state, t, dt, substeps)

    def record_trails(self) -> None:
        if self.count == 0:
            return
        self.trail_head = (self.trail_head + 1) % self.trail_len
        self.trail[:, self.trail_head, :] = self.state[:, 0:2].astype(np.float32)

    def gravitational_sources(self):
        """Bodies that still source gravity (the doomed ones no longer do)."""
        if self.count == 0:
            return np.zeros((0, 2)), np.zeros(0), np.zeros(0)
        live = self.strain <= 0.0
        return (self.state[live, 0:2], self.geom[live],
                np.maximum(self.radius[live] * 0.6, 1.5))

    def update_spaghettification(self, hole: BlackHole, dt: float,
                                 bubble: WarpBubble | None = None) -> list[int]:
        """Flag horizon crossings, integrate tidal strain, delete the finished."""
        if self.count == 0:
            return []
        d = norm(self.state[:, 0:2] - hole.position)

        sheltered = np.zeros(self.count, dtype=bool)
        if bubble is not None and bubble.active:
            # The bubble interior is flat, so geodesic deviation across it
            # vanishes identically: matter inside is exempt.
            sheltered = bubble.interior_fraction(self.state[:, 0:2]) > 0.5

        crossing = (d < hole.r_horizon) & (~sheltered)
        self.strain = np.where(crossing & (self.strain <= 0.0), 1e-3, self.strain)

        dying = self.strain > 0.0
        if np.any(dying):
            rr = np.maximum(d, hole.r_horizon * 0.25)
            rate = 2.0 * hole.mass / (rr ** 3)          # tidal tensor eigenvalue
            inc = np.minimum(rate * dt * self.TIDAL_GAIN, self.MAX_STRAIN_STEP)
            self.strain = np.where(dying, self.strain + inc, self.strain)
            # Exponential IN dt rather than per-call, so the disruption looks the
            # same at full speed and in slow motion.
            self.state[dying, 0:2] *= math.exp(-0.030 * dt)
            self.state[dying, 2:4] *= math.exp(-0.110 * dt)

        gone = np.where(self.strain > 1.0)[0]
        if gone.size:
            self.remove(gone)
        return list(map(int, gone))

    def remove(self, indices: Iterable[int]) -> None:
        keep = np.ones(self.count, dtype=bool)
        keep[np.asarray(list(indices), dtype=int)] = False
        for name in ("state", "mass_mj", "phase", "radius", "geom", "strain", "trail"):
            setattr(self, name, getattr(self, name)[keep])


# ==============================================================================
#  SECTION 14 -- DARK MATTER
#
#  STRICT INTERACTION RULE, enforced structurally rather than by convention: this
#  population is stepped through `GravityField` and NOTHING else.  There is no
#  collision code, no pressure term and no electromagnetic coupling anywhere in
#  this class -- it passes through planets, stars, walls and the spacecraft
#  because there is literally no code path by which it could not.  It responds to
#  curvature alone.
#
#  SUPERRADIANCE.  Inside the ergosphere (r < 2M) no observer can remain static;
#  every worldline is dragged prograde.  That permits negative-energy orbits and
#  hence the Penrose process: a particle can carry away more energy than it
#  brought in, drawn from the hole's rotational reservoir.  The amplification
#  condition for a co-rotating mode is
#
#      0 < omega < m Omega_H ,    Omega_H = a c^3 / (2 G M r+)
#
#  so prograde particles in the ergosphere are boosted, the gain vanishes as
#  a* -> 0, and the hole is spun down by the reciprocal bookkeeping.  Energy is
#  not created.
#
#  THE OBSERVER CONTAMINATION HYPOTHESIS.  These are fragile coherent states.
#  Any electromagnetic probe -- an active laser sensor, or simply operating
#  outside a deep-shielded laboratory -- collapses them.  A decohered particle is
#  NOT deleted: it is still there, still pulling on everything gravitationally.
#  It is merely unmeasurable.  That distinction is the entire point.
# ==============================================================================

class DarkMatterField:
    DECOHERE_RATE = 0.9      # per world-time unit under electromagnetic probing
    RECOHERE_RATE = 0.22     # per world-time unit inside deep shielding

    def __init__(self, gravity: GravityField, hole: BlackHole):
        self.gravity = gravity
        self.hole = hole
        # Built once, here, from the SAME field object every other population
        # uses.  Because it only ever calls GravityField, no EM term can leak in.
        self.integrator = RK4Integrator(gravity.make_derivative(True, True))
        self.state = np.zeros((0, 4), dtype=np.float64)
        self.coherence = np.zeros(0, dtype=np.float64)
        self.superradiant = np.zeros(0, dtype=bool)
        self.extracted_energy = 0.0

    @property
    def count(self) -> int:
        return self.state.shape[0]

    def seed_halo(self, n: int, r_min: float, r_max: float,
                  prograde_bias: float = 0.85) -> None:
        """
        Isotropic halo on near-circular orbits.

        Radii follow an r^(1/2) draw so the surface density goes as 1/r -- the
        flat-rotation-curve profile dark matter was invented to explain.
        """
        n = min(n, CFG.max_dark_particles - self.count)
        if n <= 0:
            return
        u = np.random.uniform(0.0, 1.0, n)
        r = np.sqrt(r_min ** 2 + u * (r_max ** 2 - r_min ** 2))
        ang = np.random.uniform(0.0, math.tau, n)
        pos = np.stack([np.cos(ang), np.sin(ang)], axis=1) * r[:, None]
        v_c = np.minimum(self.hole.circular_speed(r), 0.88)
        v_c = v_c * np.random.uniform(0.82, 1.06, n)
        tangent = np.stack([-np.sin(ang), np.cos(ang)], axis=1)
        sign = np.where(np.random.uniform(0, 1, n) < prograde_bias, 1.0, -1.0)
        vel = tangent * (v_c * sign)[:, None] + np.random.normal(0.0, 0.012, (n, 2))
        self.state = np.concatenate([self.state,
                                     np.concatenate([pos, vel], axis=1)])
        self.coherence = np.concatenate([self.coherence, np.ones(n)])
        self.superradiant = np.concatenate([self.superradiant,
                                            np.zeros(n, dtype=bool)])

    def clear(self) -> None:
        self.state = np.zeros((0, 4), dtype=np.float64)
        self.coherence = np.zeros(0)
        self.superradiant = np.zeros(0, dtype=bool)

    def step(self, t: float, dt: float, substeps: int,
             em_probing: bool, superradiance: bool) -> None:
        if self.count == 0 or dt == 0.0:
            return
        self.state = self.integrator.integrate(self.state, t, dt, substeps)
        pos, vel = self.state[:, 0:2], self.state[:, 2:4]
        r = norm(pos)

        # -- Penrose / superradiant amplification inside the ergosphere --------
        if superradiance and self.hole.a > CFG.eps:
            inside = r < self.hole.r_ergosphere
            if np.any(inside):
                lz = pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]   # 2D Lz
                co = inside & (lz > 0.0)
                self.superradiant = co
                if np.any(co):
                    # Bounded by Omega_H, so the gain vanishes as a* -> 0
                    # exactly as superradiance must.
                    gain = 1.0 + self.hole.omega_horizon * dt * 2.4
                    self.state[co, 2:4] *= gain
                    n_co = float(np.sum(co))
                    self.extracted_energy += n_co * (gain - 1.0)
                    # The hole pays for it out of its spin.
                    self.hole.spin = max(0.0, self.hole.spin - 1.2e-6 * n_co * dt)
            else:
                self.superradiant[:] = False
        else:
            self.superradiant[:] = False

        # -- decoherence lifecycle --------------------------------------------
        if em_probing:
            self.coherence *= math.exp(-self.DECOHERE_RATE * dt)
        else:
            self.coherence = np.minimum(1.0, self.coherence + self.RECOHERE_RATE * dt)

        # -- gravity is universal, so the horizon eats dark matter too ---------
        swallowed = r < self.hole.r_horizon
        if np.any(swallowed):
            keep = ~swallowed
            self.state = self.state[keep]
            self.coherence = self.coherence[keep]
            self.superradiant = self.superradiant[keep]

    def visible_mask(self, shielded: bool, laser_on: bool) -> np.ndarray:
        """Renders only when deep shielded, laser off, and state not destroyed."""
        if self.count == 0:
            return np.zeros(0, dtype=bool)
        if laser_on or not shielded:
            return np.zeros(self.count, dtype=bool)
        return self.coherence > 0.06


# ==============================================================================
#  SECTION 15 -- THE SPACECRAFT
#
#  MODE A -- SLINGSHOT / HALO DRIVE.  Standard relativistic propulsion.  Thrust
#  enters as a Newtonian-equivalent force and is converted by proper_accel, so
#  the ship asymptotes to c without ever being clamped.  Two genuine energy
#  sources are available:
#
#    * Frame dragging.  A prograde pass through the ergosphere rides the dragged
#      frame and leaves with more energy than it entered with -- the Penrose
#      process, already present in the force law, requiring no special code.
#    * The Halo Drive (Kipping 2018).  A photon fired ahead of the ship into the
#      photon sphere returns after a near-circular orbit blueshifted by the
#      hole's motion, and the momentum difference thrusts the ship.  Modelled as
#      an impulse available only within a few r_g of r_ph, only on a prograde
#      pass, with a gain scaling in spin -- which is exactly its physical
#      availability window and its actual energy source.
#
#  MODE B -- ALCUBIERRE WARP.  Standard propulsion is cut.  Coordinate motion is
#  imposed by the metric rather than integrated from forces, local velocity is
#  identically zero, and proper time runs at the flat-space rate.
#
#  CLOCKS.  For a Schwarzschild observer moving with local velocity v the exact
#  elapsed proper time is
#
#      dtau = dt sqrt(1 - r_s/r) sqrt(1 - v^2)
#
#  the product of the gravitational (lapse) and kinematic (Lorentz) factors.
#  Both are computed every frame; neither is faked.
# ==============================================================================

class ShipMode:
    BALLISTIC, SLINGSHOT, WARP = range(3)
    NAMES = {0: "Ballistic (free-fall geodesic)",
             1: "Slingshot / Halo Drive",
             2: "Alcubierre Warp Drive"}


class Spacecraft:
    THRUST = 0.0032
    WARP_ACCEL = 0.05
    WARP_MAX = 3.2            # coordinate speed in units of c -- legally superluminal
    HALO_GAIN = 0.010

    def __init__(self, gravity: GravityField, hole: BlackHole, bubble: WarpBubble):
        self.gravity = gravity
        self.hole = hole
        self.bubble = bubble
        self.integrator = RK4Integrator(gravity.make_derivative(True, True))
        self.mode = ShipMode.SLINGSHOT
        self.state = np.zeros((1, 4), dtype=np.float64)
        self.heading = np.array([0.0, 1.0])
        self.thrust_dir = np.zeros(2)
        self.warp_speed = 0.0
        self.halo_boost = 0.0
        self.trail = np.zeros((CFG.trail_len, 2), dtype=np.float32)
        self.trail_head = 0
        self.proper_time = 0.0        # ship clock  (tau)
        self.coord_time = 0.0         # Earth / observer clock  (t)
        self.max_speed = 0.0
        self.reset()

    def reset(self) -> None:
        r0 = max(self.hole.r_isco * 3.1, 240.0)
        v = min(float(self.hole.circular_speed(r0)), 0.55)
        self.state = np.array([[r0, 0.0, 0.0, v]], dtype=np.float64)
        self.heading = np.array([0.0, 1.0])
        self.warp_speed = 0.0
        self.halo_boost = 0.0
        self.proper_time = self.coord_time = 0.0
        self.max_speed = 0.0
        self.trail[:] = self.state[0, 0:2]
        self.mode = ShipMode.SLINGSHOT
        self.bubble.active = False

    # ------------------------------------------------------------ accessors
    @property
    def position(self) -> np.ndarray:
        return self.state[0, 0:2]

    @property
    def velocity(self) -> np.ndarray:
        return self.state[0, 2:4]

    @property
    def speed(self) -> float:
        """Local speed in units of c.  Identically zero inside a warp bubble."""
        return 0.0 if self.mode == ShipMode.WARP else float(norm(self.velocity))

    @property
    def coordinate_speed(self) -> float:
        """What a distant observer clocks -- may exceed c under warp."""
        return abs(self.warp_speed) if self.mode == ShipMode.WARP else self.speed

    @property
    def gamma(self) -> float:
        return float(lorentz_gamma(self.velocity[None, :])[0])

    @property
    def radius(self) -> float:
        return float(norm(self.position))

    def lapse(self) -> float:
        if self.mode == ShipMode.WARP and self.bubble.active:
            return 1.0
        return float(self.hole.lapse(self.radius))

    def dilation_factor(self) -> float:
        """dtau/dt = sqrt(1 - r_s/r) * sqrt(1 - v^2)."""
        if self.mode == ShipMode.WARP and self.bubble.active:
            return 1.0        # flat bubble interior: no dilation at all
        return self.lapse() * math.sqrt(max(1.0 - self.speed ** 2, 1e-12))

    # ------------------------------------------------------------- controls
    def set_thrust(self, direction: np.ndarray) -> None:
        d = np.asarray(direction, dtype=np.float64)
        n = float(norm(d))
        self.thrust_dir = d / n if n > CFG.eps else np.zeros(2)

    def set_mode(self, mode: int) -> None:
        self.mode = mode
        if mode == ShipMode.WARP:
            v = self.velocity.copy()
            n = float(norm(v))
            if n > CFG.eps:
                self.heading = v / n
            self.bubble.active = True
        else:
            if self.bubble.active:
                # Dropping out of warp hands the bubble's coordinate motion back
                # as ordinary momentum, capped just below c.
                self.state[0, 2:4] = self.heading * min(abs(self.warp_speed), 0.94)
            self.bubble.active = False
            self.warp_speed = 0.0

    def steer(self, turn: float) -> None:
        c, s = math.cos(turn), math.sin(turn)
        h = self.heading
        self.heading = np.array([c * h[0] - s * h[1], s * h[0] + c * h[1]])

    # ---------------------------------------------------------------- physics
    def step(self, t: float, dt: float, substeps: int) -> None:
        if dt == 0.0:
            return
        if self.mode == ShipMode.WARP:
            self._step_warp(dt)
        else:
            self._step_propulsive(t, dt, substeps)
        self.coord_time += dt
        self.proper_time += dt * self.dilation_factor()
        self.max_speed = max(self.max_speed, self.coordinate_speed)

    def _step_propulsive(self, t: float, dt: float, substeps: int) -> None:
        self.bubble.active = False
        self.halo_boost = 0.0
        thrust = np.zeros(2)
        if self.mode == ShipMode.SLINGSHOT:
            thrust = self.thrust_dir * self.THRUST + self._halo_drive_impulse()

        base = self.gravity.make_derivative(True, True)
        if float(norm(thrust)) > CFG.eps:
            def derivative(state: np.ndarray, tt: float) -> np.ndarray:
                d = base(state, tt)
                vel = state[..., 2:4]
                # Thrust must go through the relativistic conversion too.
                d[..., 2:4] += GravityField.proper_accel(
                    vel, np.broadcast_to(thrust, vel.shape))
                return d
            integ = RK4Integrator(derivative)
        else:
            integ = self.integrator
        self.state = integ.integrate(self.state, t, dt, substeps)

        n = float(norm(self.velocity))
        if n > CFG.eps:
            self.heading = self.velocity / n

    def _halo_drive_impulse(self) -> np.ndarray:
        """
        Kipping's halo drive: available only in a narrow annulus around the
        photon sphere and only on a prograde pass, with a gain scaling in spin
        because that is where the energy actually comes from.
        """
        r, r_ph = self.radius, self.hole.r_photon
        window = smoothstep(3.2 * r_ph, 1.05 * r_ph, r)   # descending: peaks at r_ph
        if window <= 1e-3:
            return np.zeros(2)
        pos, vel = self.position, self.velocity
        if pos[0] * vel[1] - pos[1] * vel[0] <= 0.0:       # retrograde: no gain
            return np.zeros(2)
        tangent = np.array([-pos[1], pos[0]]) / max(r, CFG.eps)
        gain = self.HALO_GAIN * float(window) * (0.35 + 0.65 * self.hole.spin)
        self.halo_boost = gain
        return tangent * gain

    def _step_warp(self, dt: float) -> None:
        """The metric moves the ship; it does not integrate forces."""
        self.warp_speed = min(self.warp_speed + self.WARP_ACCEL * dt * 5.0,
                              self.WARP_MAX)
        self.state[0, 0:2] += self.heading * self.warp_speed * dt
        self.state[0, 2:4] = 0.0        # local velocity is identically zero
        self.bubble.active = True
        self.bubble.configure(self.position, self.heading, self.warp_speed)

    def record_trail(self) -> None:
        self.trail_head = (self.trail_head + 1) % CFG.trail_len
        self.trail[self.trail_head] = self.position.astype(np.float32)

    def contracted_shape(self, base_len: float = 18.0, base_wid: float = 7.0):
        """
        Lorentz contraction L = L_0/gamma along the boost; transverse width is
        unaffected, and that asymmetry is the observable.  Under warp the ship is
        at rest in its own flat bubble, so it is uncontracted.
        """
        if self.mode == ShipMode.WARP:
            return base_len, base_wid
        return base_len / self.gamma, base_wid


# ==============================================================================
#  SECTION 16 -- PHYSICAL SCALE  (astropy.constants / astropy.cosmology)
#
#  The simulation runs in geometrized pixels, but a scientific instrument should
#  say what those pixels MEAN.  Anchoring the hole to a real mass fixes the
#  conversion for everything else:
#
#      r_s = 2 G M / c^2                     (Schwarzschild radius, metres)
#      px  = r_s(metres) / r_s(pixels)       (metres per pixel)
#
#  and the true angular Einstein radius of a lens at redshift z_L for a source at
#  z_S follows from the standard thin-lens result
#
#      theta_E = sqrt( 4 G M / c^2  *  D_LS / (D_L D_S) )
#
#  with angular diameter distances from a Planck18 cosmology.  This number is the
#  honest one -- typically microarcseconds -- and the HUD reports it alongside
#  the deliberately exaggerated on-screen ring, clearly labelled, so the display
#  never pretends to be to scale.
# ==============================================================================

class PhysicalScale:
    """Maps world pixels onto SI, and reports true observable angular scales."""

    def __init__(self, mass_msun: float = 4.297e6, z_lens: float = 0.5,
                 z_source: float = 2.0):
        self.mass_msun = mass_msun          # default: Sgr A*
        self.z_lens = z_lens
        self.z_source = z_source
        self._cosmo = None

    @property
    def cosmology(self):
        """Loaded lazily: importing a cosmology is slow and often unneeded."""
        if self._cosmo is None:
            try:
                from astropy.cosmology import Planck18
                self._cosmo = Planck18
            except Exception:
                self._cosmo = False
        return self._cosmo or None

    def schwarzschild_radius_m(self) -> float:
        m = self.mass_msun * const.M_sun
        return float((2.0 * const.G * m / const.c ** 2).to(u.m).value)

    def metres_per_pixel(self, r_s_px: float) -> float:
        return self.schwarzschild_radius_m() / max(r_s_px, 1e-9)

    def einstein_angle_arcsec(self) -> float | None:
        """True theta_E for the configured lens/source redshifts, in arcsec."""
        cosmo = self.cosmology
        if cosmo is None or self.z_source <= self.z_lens:
            return None
        try:
            d_l = cosmo.angular_diameter_distance(self.z_lens)
            d_s = cosmo.angular_diameter_distance(self.z_source)
            d_ls = cosmo.angular_diameter_distance(self.z_lens, self.z_source)
            m = self.mass_msun * const.M_sun
            theta = np.sqrt(4.0 * const.G * m / const.c ** 2 * d_ls / (d_l * d_s))
            return float((theta.decompose().value * u.rad).to(u.arcsec).value)
        except Exception:
            return None

    @staticmethod
    def format_length(metres: float) -> str:
        if metres >= 9.461e15:
            return f"{metres / 9.461e15:.3g} ly"
        if metres >= 1.496e11:
            return f"{metres / 1.496e11:.3g} AU"
        if metres >= 1e3:
            return f"{metres / 1e3:.4g} km"
        return f"{metres:.4g} m"


# ==============================================================================
#  SECTION 17 -- ZONE OF AVOIDANCE / GALACTIC DUST
#
#  About 20% of the sky is unobservable in the optical because our own galactic
#  disc lies in the way.  Interstellar dust extinguishes light by the
#  Beer-Lambert law
#
#      I = I_0 exp(-tau),     tau = column density x opacity
#
#  modelled here as a Gaussian column density about the galactic mid-plane,
#  textured with multi-octave value noise for the mottled, patchy structure real
#  dust has.  Being electromagnetic, it lives entirely on the light path: it
#  vanishes under deep-underground shielding and never touches dark matter.
# ==============================================================================

class DustFog:
    def __init__(self, width: int, height: int, plane_angle: float = -0.13,
                 thickness: float = 0.20, opacity: float = 0.94):
        self.enabled = False
        self.width, self.height = width, height
        self.plane_angle = plane_angle
        self.thickness = thickness
        self.opacity = opacity
        self.transmission = np.ones((height, width))
        self.surface = self._bake()

    @staticmethod
    def _octave(w: int, h: int, cells: int, rng: np.random.Generator) -> np.ndarray:
        cw = max(cells, 2)
        ch = max(int(cells * h / max(w, 1)), 2)
        grid = rng.random((ch + 1, cw + 1))
        yi = np.linspace(0, ch, h)
        xi = np.linspace(0, cw, w)
        y0 = np.floor(yi).astype(int); y1 = np.minimum(y0 + 1, ch)
        x0 = np.floor(xi).astype(int); x1 = np.minimum(x0 + 1, cw)
        ty = (yi - y0)[:, None]; tx = (xi - x0)[None, :]
        top = grid[y0][:, x0] * (1 - tx) + grid[y0][:, x1] * tx
        bot = grid[y1][:, x0] * (1 - tx) + grid[y1][:, x1] * tx
        return top * (1 - ty) + bot * ty

    def _bake(self) -> pygame.Surface:
        w, h = self.width, self.height
        rng = np.random.default_rng(20240514)
        noise = np.zeros((h, w)); amp = 1.0; total = 0.0
        for cells in (3, 6, 12, 24, 48):
            noise += amp * self._octave(w, h, cells, rng)
            total += amp
            amp *= 0.5
        noise /= total

        yy, xx = np.mgrid[0:h, 0:w]
        nx = (xx - w * 0.5) / w
        ny = (yy - h * 0.5) / h
        band = ny * math.cos(self.plane_angle) - nx * math.sin(self.plane_angle)
        column = np.exp(-(band / self.thickness) ** 2)

        tau = 3.4 * column * (0.30 + 0.70 * noise)      # optical depth
        alpha = (1.0 - np.exp(-tau)) * 255.0 * self.opacity
        warm = 0.55 + 0.45 * noise
        rgb = np.empty((h, w, 3), dtype=np.uint8)
        rgb[..., 0] = np.clip(46 * warm + 14, 0, 255)
        rgb[..., 1] = np.clip(33 * warm + 10, 0, 255)
        rgb[..., 2] = np.clip(30 * warm + 16, 0, 255)

        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.surfarray.pixels3d(surf)[:] = np.transpose(rgb, (1, 0, 2))
        pygame.surfarray.pixels_alpha(surf)[:] = np.transpose(
            alpha.astype(np.uint8), (1, 0))
        self.transmission = np.exp(-tau)
        return surf

    def extinction_at(self, screen_pts: np.ndarray) -> np.ndarray:
        """Transmission in [0, 1] for objects seen through the dust."""
        if not self.enabled or screen_pts.size == 0:
            return np.ones(max(screen_pts.shape[0], 1))
        xs = np.clip(screen_pts[:, 0].astype(int), 0, self.width - 1)
        ys = np.clip(screen_pts[:, 1].astype(int), 0, self.height - 1)
        return self.transmission[ys, xs]

    def draw(self, surf: pygame.Surface) -> None:
        if self.enabled:
            surf.blit(self.surface, (0, 0))


# ==============================================================================
#  SECTION 18 -- ELECTROMAGNETIC CONTAMINATION
#
#  The Observer Contamination Hypothesis, made quantitative.
#
#  Lensing arcs are not bright objects; they are a faint CONTRAST modulation on
#  top of the sky.  Whether they are detectable is therefore a signal-to-noise
#  question:
#
#      SNR = sigma_arc / sqrt( sigma_read^2 + sigma_contamination^2 )
#
#  Switching on a visible-light sensor, or leaving deep-shielded conditions,
#  injects a large additive noise pedestal.  It does not paint over the arcs -- it
#  buries them, and the SNR readout falls below 1 while the underlying lensing
#  physics continues untouched.  That is precisely why these measurements are
#  made deep underground, and why the arcs "disappear" without anything about the
#  gravity having changed.
# ==============================================================================

class ContaminationField:
    """
    Pre-baked additive photon noise at quantized amplitudes.

    IMPORTANT pygame detail: BLEND_ADD ignores a surface's per-surface alpha, so
    scaling the pedestal with set_alpha silently does nothing and the noise is
    always added at full amplitude.  The amplitude therefore has to be baked
    into the pixel VALUES.  Levels are quantized to sixteenths and built lazily,
    so a session that only ever uses "shielded" and "laser on" pays for exactly
    two surfaces.

    Animation comes from blitting an oversized field at a cycling sub-offset,
    which is free, rather than from storing independent frames.
    """

    MARGIN = 48
    READ_NOISE = 1.6           # irreducible detector noise floor, in DN

    def __init__(self, width: int, height: int, seed: int = 5150):
        self.width, self.height = width, height
        rng = np.random.default_rng(seed)
        w, h = width + self.MARGIN, height + self.MARGIN
        # Correlated (not per-pixel white) noise: real stray light has structure.
        small = rng.random((h // 3 + 1, w // 3 + 1)).astype(np.float32)
        self._noise = np.clip(
            np.repeat(np.repeat(small, 3, axis=0), 3, axis=1)[:h, :w], 0.0, 1.0)
        self._cache: dict[int, pygame.Surface] = {}
        self._phase = 0

    def _surface_for(self, step: int) -> pygame.Surface:
        surf = self._cache.get(step)
        if surf is None:
            v = (self._noise * (255.0 * step / 16.0)).astype(np.uint32).T
            s = pygame.Surface((v.shape[0], v.shape[1])).convert(32, 0)
            pygame.surfarray.blit_array(s, (v << 16) | (v << 8) | v)
            self._cache[step] = surf = s
        return surf

    def draw(self, surf: pygame.Surface, level: float) -> None:
        """Additively blend a photon-noise pedestal of the given strength (0..1)."""
        step = int(round(np.clip(level, 0.0, 1.0) * 16.0))
        if step <= 0:
            return
        self._phase = (self._phase + 7) % (self.MARGIN - 1)
        surf.blit(self._surface_for(step),
                  (-self._phase, -(self._phase * 3) % (self.MARGIN - 1)),
                  special_flags=pygame.BLEND_ADD)

    @staticmethod
    def snr(arc_contrast: float, level: float) -> float:
        """
        SNR = sigma_arc / sqrt(sigma_read^2 + sigma_contamination^2).

        The arcs are a faint contrast modulation, so detectability is a
        noise question, not a "can you see them" question.
        """
        sigma = math.sqrt(ContaminationField.READ_NOISE ** 2
                          + (86.0 * max(level, 0.0)) ** 2)
        return arc_contrast / max(sigma, 1e-6)


# ==============================================================================
#  SECTION 19 -- UI TOOLKIT
# ==============================================================================

class CachedFont:
    """
    Memoizing pygame.font.Font wrapper.

    The sidebar draws ~90 strings per frame and nearly all are fixed labels, so
    rasterizing each once and reusing the surface removes that work entirely.
    The cache is bounded and simply cleared when full; numeric readouts churn a
    few dozen entries a second, which this absorbs without unbounded growth.
    """

    __slots__ = ("font", "_cache", "_limit")

    def __init__(self, font: pygame.font.Font, limit: int = 1400):
        self.font = font
        self._cache: dict = {}
        self._limit = limit

    def render(self, text: str, antialias: bool = True,
               color=(255, 255, 255)) -> pygame.Surface:
        key = (text, tuple(color))
        surf = self._cache.get(key)
        if surf is None:
            if len(self._cache) >= self._limit:
                self._cache.clear()
            surf = self.font.render(text, antialias, color)
            self._cache[key] = surf
        return surf


class Fonts:
    def __init__(self):
        pygame.font.init()
        mono = "dejavusansmono,consolas,monospace"
        sans = "dejavusans,arial,sans"
        self.tiny = CachedFont(pygame.font.SysFont(mono, 11))
        self.small = CachedFont(pygame.font.SysFont(mono, 12))
        self.body = CachedFont(pygame.font.SysFont(mono, 13))
        self.head = CachedFont(pygame.font.SysFont(sans, 14, bold=True))
        self.big = CachedFont(pygame.font.SysFont(sans, 19, bold=True))


class Widget:
    def __init__(self, rect: pygame.Rect, label: str):
        self.rect = rect
        self.label = label

    def handle(self, event: pygame.event.Event) -> bool:
        return False

    def draw(self, surf: pygame.Surface, fonts: Fonts) -> None:
        raise NotImplementedError


class Slider(Widget):
    def __init__(self, rect, label, lo, hi, value, fmt="{:.3f}", unit=""):
        super().__init__(rect, label)
        self.lo, self.hi = lo, hi
        self.value = float(np.clip(value, lo, hi))
        self.fmt, self.unit = fmt, unit
        self.dragging = False

    @property
    def t(self) -> float:
        return (self.value - self.lo) / max(self.hi - self.lo, CFG.eps)

    def _track(self) -> pygame.Rect:
        return pygame.Rect(self.rect.x, self.rect.bottom - 9, self.rect.w, 5)

    def _set_from_x(self, x: float) -> None:
        tr = self._track()
        t = float(np.clip((x - tr.x) / max(tr.w, 1), 0.0, 1.0))
        self.value = self.lo + t * (self.hi - self.lo)

    def handle(self, event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.inflate(0, 10).collidepoint(event.pos):
                self.dragging = True
                self._set_from_x(event.pos[0])
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.dragging:
            self.dragging = False
            return True
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._set_from_x(event.pos[0])
            return True
        return False

    def draw(self, surf, fonts) -> None:
        surf.blit(fonts.small.render(self.label, True, Palette.TEXT_DIM),
                  (self.rect.x, self.rect.y))
        t = fonts.small.render(self.fmt.format(self.value) + self.unit, True,
                               Palette.ACCENT)
        surf.blit(t, (self.rect.right - t.get_width(), self.rect.y))
        tr = self._track()
        pygame.draw.rect(surf, (30, 36, 54), tr, border_radius=3)
        pygame.draw.rect(surf, Palette.ACCENT,
                         pygame.Rect(tr.x, tr.y, int(tr.w * self.t), tr.h),
                         border_radius=3)
        kx = tr.x + int(tr.w * self.t)
        pygame.draw.circle(surf, Palette.TEXT, (kx, tr.centery), 6)
        pygame.draw.circle(surf, Palette.PANEL, (kx, tr.centery), 4)


class Toggle(Widget):
    def __init__(self, rect, label, value=False, key="", color=Palette.OK):
        super().__init__(rect, label)
        self.value, self.key, self.color = value, key, color

    def handle(self, event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 \
                and self.rect.collidepoint(event.pos):
            self.value = not self.value
            return True
        return False

    def draw(self, surf, fonts) -> None:
        box = pygame.Rect(self.rect.x, self.rect.y + 2, 26, 14)
        pygame.draw.rect(surf, self.color if self.value else (38, 44, 62), box,
                         border_radius=7)
        pygame.draw.circle(surf, Palette.PANEL if self.value else (96, 106, 132),
                           (box.right - 7 if self.value else box.x + 7,
                            box.centery), 5)
        surf.blit(fonts.small.render(
            self.label, True, Palette.TEXT if self.value else Palette.TEXT_DIM),
            (box.right + 9, self.rect.y))
        if self.key:
            k = fonts.tiny.render(f"[{self.key}]", True, (78, 90, 118))
            surf.blit(k, (self.rect.right - k.get_width(), self.rect.y + 1))


class Button(Widget):
    def __init__(self, rect, label, color=Palette.PANEL_EDGE):
        super().__init__(rect, label)
        self.color = color
        self.pressed = False
        self.hover = False

    def handle(self, event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 \
                and self.rect.collidepoint(event.pos):
            self.pressed = True
            return True
        return False

    def consume(self) -> bool:
        was, self.pressed = self.pressed, False
        return was

    def draw(self, surf, fonts) -> None:
        bg = tuple(min(255, c + 22) for c in self.color) if self.hover else self.color
        pygame.draw.rect(surf, bg, self.rect, border_radius=4)
        pygame.draw.rect(surf, (58, 70, 100), self.rect, 1, border_radius=4)
        t = fonts.small.render(self.label, True, Palette.TEXT)
        surf.blit(t, t.get_rect(center=self.rect.center))


class RadioGroup(Widget):
    def __init__(self, rect, options, index=0):
        super().__init__(rect, "")
        self.options = list(options)
        self.index = index

    def _cells(self) -> list[pygame.Rect]:
        w = self.rect.w / len(self.options)
        return [pygame.Rect(int(self.rect.x + i * w), self.rect.y, int(w) - 3,
                            self.rect.h) for i in range(len(self.options))]

    def handle(self, event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for i, c in enumerate(self._cells()):
                if c.collidepoint(event.pos):
                    self.index = i
                    return True
        return False

    def draw(self, surf, fonts) -> None:
        for i, (c, name) in enumerate(zip(self._cells(), self.options)):
            on = (i == self.index)
            pygame.draw.rect(surf, (36, 58, 92) if on else (22, 26, 40), c,
                             border_radius=4)
            if on:
                pygame.draw.rect(surf, Palette.ACCENT, c, 1, border_radius=4)
            t = fonts.tiny.render(name, True,
                                  Palette.TEXT if on else Palette.TEXT_DIM)
            surf.blit(t, t.get_rect(center=c.center))


# ==============================================================================
#  SECTION 20 -- OVERLAY RENDERER
#
#  Everything drawn ON TOP of the ray-traced field: the causal boundaries, the
#  particle populations, the ship, and the annotations.  Anything representing
#  light is gated on `light_enabled`; under deep-underground shielding every
#  electromagnetic channel is cut and only gravity-coupled things survive.
# ==============================================================================

class Overlay:
    def __init__(self, camera: Camera, fonts: Fonts):
        self.camera = camera
        self.fonts = fonts
        self.light_enabled = True
        self.show_labels = True
        self.show_trails = True
        self._ring = np.arange(CFG.trail_len)
        self._glow_cache: dict = {}

    def _ring_order(self, head: int) -> np.ndarray:
        return (head + 1 + self._ring) % CFG.trail_len

    # ------------------------------------------------------------ boundaries
    def draw_boundaries(self, surf: pygame.Surface, hole: BlackHole,
                        lens: LensModel, einstein_r: float) -> None:
        cam = self.camera
        cx, cy = cam.to_screen(hole.position[None, :])[0]
        s = cam.scale

        # Frame-dragging streamlines whose winding follows omega ~ 2GJ/(c^2 r^3),
        # so the swirl visibly tightens inward.
        if hole.a > CFG.eps and self.light_enabled:
            rr = np.linspace(hole.r_horizon * 1.02, hole.r_ergosphere, 22)
            phase = 2.0 * hole.angular_momentum / rr ** 3
            phase = (phase - phase[-1]) * 26.0 + hole.phase * 1.6
            for k in range(7):
                a = k * math.tau / 7 + phase
                pts = cam.to_screen(np.stack([np.cos(a), np.sin(a)], axis=1)
                                    * rr[:, None])
                f = int(60 + 90 * hole.spin)
                pygame.draw.lines(surf, (f, int(f * 0.6), f + 40), False,
                                  pts.tolist(), 1)

        self._dashed(surf, (cx, cy), hole.r_ergosphere * s, Palette.ERGO, 9, 2)
        pc = Palette.PHOTON if self.light_enabled else (74, 70, 48)
        self._dashed(surf, (cx, cy), hole.r_photon * s, pc, 6, 2)
        if hole.spin > 0.02:
            self._dashed(surf, (cx, cy), hole.r_photon_retro * s,
                         tuple(int(c * 0.45) for c in pc), 4, 1)
        self._dashed(surf, (cx, cy), hole.r_isco * s, Palette.ISCO, 3, 1)
        # b_c bounds the shadow the ray tracer produced; drawing it makes the
        # relationship between 2M, 3M and 3*sqrt(3)M legible.
        self._dashed(surf, (cx, cy), lens.table.b_critical * s, (120, 140, 190), 5, 1)
        pygame.draw.circle(surf, (168, 214, 255) if self.light_enabled
                           else (48, 60, 84), (int(cx), int(cy)),
                           max(int(hole.r_horizon * s), 2), 2)
        if einstein_r > 0:
            self._dashed(surf, (cx, cy), einstein_r * s, Palette.EINSTEIN, 11, 1)

        if self.show_labels:
            self._labels(surf, (cx, cy), hole, lens, einstein_r, s)

    def _dashed(self, surf, c, r, col, dash=8, width=1) -> None:
        if r < 3 or r > 20000:
            return
        seg = min(max(int(math.tau * r / max(dash * 2.4, 4)), 8), 520)
        a = np.linspace(0, math.tau, seg * 2, endpoint=False)
        pts = np.stack([np.cos(a), np.sin(a)], axis=1) * r
        pts[:, 0] += c[0]
        pts[:, 1] += c[1]
        for i in range(0, len(pts) - 1, 2):
            pygame.draw.line(surf, col, pts[i], pts[i + 1], width)

    def _labels(self, surf, c, hole: BlackHole, lens: LensModel,
                einstein_r: float, s: float) -> None:
        cx, cy = c
        items = [
            (hole.r_horizon, f"EVENT HORIZON  r+={hole.r_horizon:.0f}"
                             "  (no return)", (200, 226, 255), -1.15),
            (hole.r_ergosphere, f"ERGOSPHERE  r=2M={hole.r_ergosphere:.0f}"
                                "  (frame dragging)", Palette.ERGO, -0.62),
            (hole.r_photon, f"PHOTON SPHERE  r={hole.r_photon:.0f}"
                            "  (light orbits)", Palette.PHOTON, -0.12),
            (lens.table.b_critical,
             f"SHADOW  b_c=3sqrt(3)M={lens.table.b_critical:.0f}",
             (150, 170, 210), 0.34),
        ]
        if einstein_r > 0:
            items.append((einstein_r,
                          f"EINSTEIN RING  theta_E={einstein_r:.0f}"
                          "  (critical curve)", Palette.EINSTEIN, 0.80))
        f = self.fonts.tiny
        # Leader length grows with the callout index so labels on nearly the same
        # radius still land on separate lines instead of overprinting.
        for k, (r, text, col, ang) in enumerate(items):
            rp = r * s
            ax, ay = math.cos(ang), math.sin(ang)
            lead = 26 + 16 * k
            p0 = (cx + ax * rp, cy - ay * rp)
            p1 = (cx + ax * (rp + lead), cy - ay * (rp + lead))
            end = (p1[0] + 20, p1[1])
            pygame.draw.line(surf, col, p0, p1, 1)
            pygame.draw.line(surf, col, p1, end, 1)
            surf.blit(f.render(text, True, col), (end[0] + 5, end[1] - 6))

    # ---------------------------------------------------------------- bodies
    def draw_bodies(self, surf, pop: BodyPopulation, hole: BlackHole,
                    fog: DustFog) -> None:
        if pop.count == 0:
            return
        scr = self.camera.to_screen(pop.state[:, 0:2])
        s = self.camera.scale
        clip = self.camera.viewport
        dim = fog.extinction_at(scr) if fog.enabled else np.ones(pop.count)

        if self.show_trails:
            order = self._ring_order(pop.trail_head)
            for i in range(pop.count):
                pts = self.camera.to_screen(pop.trail[i][order])
                col = tuple(int(c * 0.34) + 12 for c in Phase.COLORS[int(pop.phase[i])])
                pygame.draw.lines(surf, col, False, pts.tolist(), 1)

        for i in range(pop.count):
            x, y = float(scr[i, 0]), float(scr[i, 1])
            r = max(pop.radius[i] * s, 1.6)
            ph = int(pop.phase[i])
            if not self.light_enabled:
                # Under shielding a body is known only by its gravity.
                pygame.draw.circle(surf, (52, 60, 82), (int(x), int(y)), int(r), 1)
                continue
            col = tuple(int(c * (0.25 + 0.75 * float(dim[i])))
                        for c in Phase.COLORS[ph])
            if pop.strain[i] > 0.0:
                self._spaghetti(surf, pop, i, (x, y), r, col, hole)
                continue
            if ph in (Phase.BROWN_DWARF, Phase.STAR):
                gr = min(r * (2.7 if ph == Phase.STAR else 1.9), 78.0)
                if -gr < x < clip.width + gr and -gr < y < clip.height + gr:
                    self._glow(surf, (x, y), gr, col,
                               0.5 if ph == Phase.STAR else 0.26)
            if ph == Phase.COLLAPSED:
                pygame.draw.circle(surf, (0, 0, 0), (int(x), int(y)), int(r))
                pygame.draw.circle(surf, (150, 190, 255), (int(x), int(y)), int(r), 2)
                self._dashed(surf, (x, y), r * 1.5, (90, 110, 160), 5, 1)
            else:
                pygame.draw.circle(surf, col, (int(x), int(y)), int(r))
                if ph <= Phase.GAS_GIANT and r > 4:
                    pygame.draw.circle(surf, tuple(int(c * 0.45) for c in col),
                                       (int(x + r * 0.34), int(y + r * 0.24)),
                                       int(r * 0.82))
            if ph == Phase.DEGENERATE:
                pygame.draw.circle(surf, (200, 230, 255), (int(x), int(y)),
                                   int(r) + 3, 1)

    def _spaghetti(self, surf, pop, i, p, r, col, hole: BlackHole) -> None:
        """
        The tidal filament.

        Geodesic deviation gives +2GM dr/r^3 radially and -GM dr/r^3
        transversely.  That 2:-1 ratio is area-preserving to first order, so the
        drawn ellipse stretches by k along the RADIAL direction (which is the
        "vertical" of the infalling body's own frame) and shrinks by 1/sqrt(k)
        across it -- the shape the metric actually predicts.
        """
        strain = float(pop.strain[i])
        k = 1.0 + 13.0 * strain ** 1.5
        a, b = r * k, r / math.sqrt(k)
        d = pop.state[i, 0:2] - hole.position
        ang = math.atan2(-d[1], d[0])              # screen y is flipped
        th = np.linspace(0, math.tau, 26, endpoint=False)
        ex, ey = a * np.cos(th), b * np.sin(th)
        ca, sa = math.cos(ang), math.sin(ang)
        px = p[0] + ex * ca - ey * sa
        py = p[1] + ex * sa + ey * ca
        glow = lerp_color(col, (255, 236, 200), min(strain, 1.0))
        glow = lerp_color(Palette.VOID, glow, float(1.0 - smoothstep(0.72, 1.0, strain)))
        pygame.draw.polygon(surf, glow, np.stack([px, py], axis=1).tolist())

    def _glow(self, surf, p, radius, col, strength) -> None:
        radius = min(radius, 200.0)
        if radius < 2:
            return
        key = (int(radius / 3.0), col, int(strength * 16.0))
        g = self._glow_cache.get(key)
        if g is None:
            rad = max(key[0] * 3.0, 3.0)
            d = int(rad * 2) + 2
            g = pygame.Surface((d, d), pygame.SRCALPHA)
            for i in range(9, 0, -1):
                a = int(255 * (key[2] / 16.0) * (1.0 - i / 10.0) ** 1.7)
                pygame.draw.circle(g, (*col, a), (d // 2, d // 2),
                                   int(rad * i / 9.0))
            if len(self._glow_cache) > 256:
                self._glow_cache.clear()
            self._glow_cache[key] = g
        r = g.get_width() * 0.5
        surf.blit(g, (p[0] - r, p[1] - r), special_flags=pygame.BLEND_ADD)

    # ----------------------------------------------------------- dark matter
    def draw_dark_matter(self, surf, dm: DarkMatterField, shielded: bool,
                         laser_on: bool) -> None:
        """EM-blind, so dust never dims it and the sky never occludes it."""
        if dm.count == 0:
            return
        vis = dm.visible_mask(shielded, laser_on)
        if not vis.any():
            return
        scr = self.camera.to_screen(dm.state[vis, 0:2])
        for p, c, sr in zip(scr, dm.coherence[vis], dm.superradiant[vis]):
            base = (255, 214, 120) if sr else Palette.DARK_MATTER
            col = tuple(int(v * (0.35 + 0.65 * c)) for v in base)
            pygame.draw.circle(surf, col, (int(p[0]), int(p[1])), 2 if sr else 1)

    # ------------------------------------------------------------------ ship
    def draw_ship(self, surf, ship: Spacecraft, bubble: WarpBubble) -> None:
        cam = self.camera
        s = cam.scale
        p = cam.to_screen(ship.position[None, :])[0]

        if self.show_trails:
            pts = cam.to_screen(ship.trail[self._ring_order(ship.trail_head)])
            pygame.draw.lines(surf, (54, 122, 104), False, pts.tolist(), 1)

        if ship.mode == ShipMode.WARP and bubble.active:
            self._warp_wall(surf, bubble, s)

        length, width = ship.contracted_shape()
        length *= s
        width *= s
        h = ship.heading if ship.mode == ShipMode.WARP else (
            ship.velocity / max(float(norm(ship.velocity)), CFG.eps))
        ang = math.atan2(-h[1], h[0])
        ca, sa = math.cos(ang), math.sin(ang)
        hull = np.array([[length, 0.0], [-length * .55, width],
                         [-length * .30, 0.0], [-length * .55, -width]])
        rot = np.stack([hull[:, 0] * ca - hull[:, 1] * sa,
                        hull[:, 0] * sa + hull[:, 1] * ca], axis=1) + p
        col = Palette.WARP if ship.mode == ShipMode.WARP else Palette.SHIP
        pygame.draw.polygon(surf, col, rot.tolist())
        pygame.draw.polygon(surf, (250, 255, 255), rot.tolist(), 1)

        if ship.mode == ShipMode.SLINGSHOT and float(norm(ship.thrust_dir)) > 0:
            t = ship.thrust_dir
            pygame.draw.line(surf, Palette.WARN, p,
                             p - np.array([t[0], -t[1]]) * (length + 9), 2)
        if ship.halo_boost > 1e-4:
            pygame.draw.circle(surf, Palette.PHOTON, (int(p[0]), int(p[1])),
                               int(14 + 260 * ship.halo_boost), 1)

    def _warp_wall(self, surf, bubble: WarpBubble, s: float) -> None:
        """
        The bubble wall, coloured by sign(theta): contraction ahead (blue),
        expansion behind (red).  The interior is shaded to advertise flatness.
        """
        c = self.camera.to_screen(bubble.center[None, :])[0]
        R = bubble.radius * s
        th = np.linspace(0, math.tau, 96, endpoint=False)
        world = bubble.center[None, :] + np.stack([np.cos(th), np.sin(th)],
                                                  axis=1) * bubble.radius
        theta = bubble.expansion_scalar(world)
        scr = self.camera.to_screen(world)
        mx = max(float(np.max(np.abs(theta))), CFG.eps)
        for i in range(len(scr)):
            col = lerp_color(Palette.GRID_EXPAND, Palette.GRID_CONTRACT,
                             0.5 - 0.5 * float(theta[i]) / mx)
            pygame.draw.line(surf, col, scr[i], scr[(i + 1) % len(scr)], 2)
        inner = pygame.Surface((int(R * 2) + 4, int(R * 2) + 4), pygame.SRCALPHA)
        pygame.draw.circle(inner, (40, 120, 150, 30), (int(R) + 2, int(R) + 2),
                           int(R * 0.86))
        surf.blit(inner, (c[0] - R - 2, c[1] - R - 2))
        t = self.fonts.tiny.render("FLAT INTERIOR  (Riemann = 0)", True,
                                   (120, 210, 235))
        surf.blit(t, (c[0] - t.get_width() // 2, c[1] + R * 0.86 + 5))


# ==============================================================================
#  SECTION 21 -- CONTROL PANEL
# ==============================================================================

class ControlPanel:
    """Sidebar: instrument metadata, sliders, toggles and live telemetry."""

    ROW_H = 14
    BAR_H = 22

    def __init__(self, rect: pygame.Rect, fonts: Fonts):
        self.rect = rect
        self.fonts = fonts
        self.widgets: list[Widget] = []
        self.sections: list[tuple[int, str]] = []
        self.x = rect.x + 13
        self.w = rect.w - 26
        self._y = 50

        self._section("SIMULATION CONTROL")
        self.s_dt = self._slider("Time step  dt", CFG.dt_min, CFG.dt_max, CFG.dt)
        bw = (self.w - 12) // 3
        self.b_pause = Button(pygame.Rect(self.x, self._y, bw, 19), "PAUSE")
        self.b_reset = Button(pygame.Rect(self.x + bw + 6, self._y, bw, 19), "RESET")
        self.b_clear = Button(pygame.Rect(self.x + 2 * bw + 12, self._y, bw, 19),
                              "CLEAR")
        self.widgets += [self.b_pause, self.b_reset, self.b_clear]
        self._y += 23

        self._section("LENS  /  BLACK HOLE")
        self.s_mass = self._slider("Mass  M = GM/c^2", CFG.bh_mass_min,
                                   CFG.bh_mass_max, CFG.bh_mass, "{:.1f}", " px")
        self.s_spin = self._slider("Kerr spin  a* = a/M", 0.0, 0.998, CFG.bh_spin)
        self.s_depth = self._slider("Lens depth  D_L D_LS/D_S", CFG.lens_depth_min,
                                    CFG.lens_depth_max, CFG.lens_depth, "{:.0f}", " px")

        self._section("QUINTESSENCE / DARK ENERGY")
        self.s_hubble = self._slider("Expansion rate  H", CFG.hubble_min,
                                     CFG.hubble_max, CFG.hubble, "{:.5f}")

        self._section("OPTICS & OVERLAYS")
        self.t_labels = self._toggle("Boundary annotations", True, "B", Palette.ACCENT)
        self.t_trails = self._toggle("Trajectory trails", True, "T", Palette.ACCENT)
        self.t_fog = self._toggle("Zone of Avoidance dust", False, "Z", Palette.WARN)
        self.t_follow = self._toggle("Camera follows ship", False, "F", Palette.SHIP)

        self._section("DARK SECTOR RESEARCH MODE")
        self.t_dm = self._toggle("Dark matter halo", True, "D", Palette.DARK_MATTER)
        self.t_shield = self._toggle("Deep underground shielding", False, "U",
                                     Palette.DARK_MATTER)
        self.t_laser = self._toggle("Visible light / laser sensor", False, "L",
                                    Palette.DANGER)
        self.t_super = self._toggle("Superradiance (Penrose)", True, "", Palette.WARN)
        self.t_distort = self._toggle("Magnification map", False, "M",
                                      Palette.EINSTEIN)

        self._section("SPAWN CLASS  /  PROPULSION")
        self.r_spawn = self._radio(["Planet", "Star", "Dark Matter"], 0)
        self.r_mode = self._radio(["Ballistic", "Slingshot", "WARP"], 1)

        self._section("TELEMETRY")
        self.telemetry_y = self._y

    # ------------------------------------------------------------ layout
    def _section(self, title: str) -> None:
        self._y += 7
        self.sections.append((self._y, title))
        self._y += 21

    def _slider(self, label, lo, hi, val, fmt="{:.3f}", unit="") -> Slider:
        s = Slider(pygame.Rect(self.x, self._y, self.w, 19), label, lo, hi, val,
                   fmt, unit)
        self.widgets.append(s)
        self._y += 23
        return s

    def _toggle(self, label, val, key, col) -> Toggle:
        t = Toggle(pygame.Rect(self.x, self._y, self.w, 17), label, val, key, col)
        self.widgets.append(t)
        self._y += 17
        return t

    def _radio(self, options, index) -> RadioGroup:
        r = RadioGroup(pygame.Rect(self.x, self._y, self.w, 19), options, index)
        self.widgets.append(r)
        self._y += 22
        return r

    # ---------------------------------------------------------------- events
    def handle(self, event) -> bool:
        used = False
        for wdg in self.widgets:
            used |= wdg.handle(event)
        return used

    # ------------------------------------------------------------- rendering
    def _row(self, surf, y, key, val, col=Palette.TEXT) -> int:
        f = self.fonts
        surf.blit(f.small.render(key, True, Palette.TEXT_DIM), (self.x, y))
        t = f.small.render(val, True, col)
        surf.blit(t, (self.x + self.w - t.get_width(), y))
        return y + self.ROW_H

    def _bar(self, surf, y, label, frac, col, note="") -> int:
        f = self.fonts
        surf.blit(f.tiny.render(label, True, Palette.TEXT_DIM), (self.x, y))
        if note:
            t = f.tiny.render(note, True, col)
            surf.blit(t, (self.x + self.w - t.get_width(), y))
        bar = pygame.Rect(self.x, y + 12, self.w, 6)
        pygame.draw.rect(surf, (28, 34, 50), bar, border_radius=3)
        fw = int(self.w * float(np.clip(frac, 0.0, 1.0)))
        if fw:
            pygame.draw.rect(surf, col, pygame.Rect(bar.x, bar.y, fw, 6),
                             border_radius=3)
        return y + self.BAR_H

    def draw(self, surf: pygame.Surface, sim: "Simulation") -> None:
        pygame.draw.rect(surf, Palette.PANEL, self.rect)
        pygame.draw.line(surf, Palette.PANEL_EDGE, (self.rect.x, self.rect.y),
                         (self.rect.x, self.rect.bottom), 2)
        f = self.fonts
        surf.blit(f.big.render("ROMAN LENSING LAB", True, Palette.TEXT), (self.x, 11))
        surf.blit(f.tiny.render("G = c = 1   |   1 px = 1 GM/c^2 unit", True,
                                Palette.TEXT_DIM), (self.x, 32))
        for y, title in self.sections:
            surf.blit(f.head.render(title, True, Palette.ACCENT), (self.x, y))
            pygame.draw.line(surf, (32, 40, 60), (self.x, y + 17),
                             (self.x + self.w, y + 17), 1)
        for wdg in self.widgets:
            wdg.draw(surf, f)
        self._telemetry(surf, sim)

    def _telemetry(self, surf, sim: "Simulation") -> None:
        f = self.fonts
        ship, hole, lens = sim.ship, sim.hole, sim.lens
        y = self.telemetry_y + 3

        # ---- lensing observables ---------------------------------------------
        rE = sim.lens_renderer.einstein_radius
        y = self._row(surf, y, "Einstein radius (render)",
                      f"{rE:8.1f} px", Palette.EINSTEIN)
        true_theta = sim.scale.einstein_angle_arcsec()
        y = self._row(surf, y, "  true theta_E (z=0.5/2.0)",
                      "n/a" if true_theta is None else f"{true_theta * 1e6:8.2f} uas",
                      Palette.TEXT_DIM)
        y = self._row(surf, y, "Shadow  b_c = 3sqrt(3)M",
                      f"{lens.table.b_critical:8.1f} px")
        mpp = sim.scale.metres_per_pixel(hole.r_s)
        y = self._row(surf, y, "Scale  (Sgr A* anchor)",
                      PhysicalScale.format_length(mpp) + "/px", Palette.TEXT_DIM)
        y = self._row(surf, y, "r_s physical",
                      PhysicalScale.format_length(sim.scale.schwarzschild_radius_m()),
                      Palette.TEXT_DIM)

        # ---- detection SNR ----------------------------------------------------
        snr = sim.lensing_snr
        col = Palette.OK if snr >= 5 else (Palette.WARN if snr >= 1 else Palette.DANGER)
        tag = ("DETECTED" if snr >= 5 else
               ("MARGINAL" if snr >= 1 else "UNMEASURABLE"))
        y = self._bar(surf, y, "Arc detection  SNR", min(snr / 20.0, 1.0), col,
                      f"{snr:.2f}  {tag}")

        # ---- clocks -----------------------------------------------------------
        dil = ship.dilation_factor()
        y = self._row(surf, y, "Earth / observer clock", f"{ship.coord_time:10.2f}")
        y = self._row(surf, y, "Ship proper clock  tau", f"{ship.proper_time:10.2f}",
                      Palette.OK)
        y = self._row(surf, y, "Accumulated lag",
                      f"{ship.coord_time - ship.proper_time:10.2f}", Palette.WARN)
        col = Palette.DANGER if dil < 0.5 else (Palette.WARN if dil < 0.85
                                                else Palette.OK)
        y = self._bar(surf, y, "dtau/dt  grav x kinematic", dil, col, f"{dil:.4f}")

        # ---- kinematics -------------------------------------------------------
        v = ship.coordinate_speed
        y = self._row(surf, y, "Speed", f"{v:9.4f} c",
                      Palette.WARP if v > 1.0 else Palette.TEXT)
        y = self._row(surf, y, "gamma / contraction L/L0",
                      f"{ship.gamma:.3f} / {1.0 / ship.gamma:.4f}")
        r_rs = ship.radius / max(hole.r_s, CFG.eps)
        y = self._row(surf, y, "Radius  r / rs", f"{r_rs:10.3f}",
                      Palette.DANGER if r_rs < 1.2 else Palette.TEXT)
        y = self._bar(surf, y, "Approach to light speed", min(v, 1.0),
                      Palette.WARP if v > 1 else Palette.ACCENT,
                      "SUPERLUMINAL" if v > 1.0 else f"{v * 100:.1f}% c")

        # ---- atomic equilibrium ----------------------------------------------
        surf.blit(f.head.render("ATOMIC EQUILIBRIUM", True, Palette.ACCENT),
                  (self.x, y))
        y += 18
        idx = sim.focus_index
        if 0 <= idx < sim.pop.count:
            m = float(sim.pop.mass_mj[idx])
            ph = int(sim.pop.phase[idx])
            grav, deg = StellarStructure.pressure_balance(m)
            frac = float(np.clip(math.log10(max(grav / max(deg, 1e-30), 1e-12))
                                 / 8.0 + 0.5, 0, 1))
            y = self._row(surf, y, Phase.NAMES[ph], f"{m:9.2f} MJ", Phase.COLORS[ph])
            y = self._bar(surf, y, "compression vs EM resistance", frac,
                          Palette.DANGER if frac > 0.62 else Palette.OK,
                          "COLLAPSE" if ph >= Phase.DEGENERATE else "held")
            for thr, name in ((StellarStructure.M_DEUTERIUM, "brown dwarf"),
                              (StellarStructure.M_HYDROGEN, "fusion ignition"),
                              (StellarStructure.M_CHANDRA, "Chandrasekhar"),
                              (StellarStructure.M_TOV, "horizon forms")):
                if m < thr:
                    y = self._row(surf, y, f"-> {name}", f"{thr - m:9.1f} MJ",
                                  Palette.WARN)
                    break
            else:
                y = self._row(surf, y, "-> fully collapsed", "", Palette.DANGER)
        else:
            y = self._row(surf, y, "No body selected", "click one", Palette.TEXT_DIM)
            y += self.BAR_H

        # ---- status -----------------------------------------------------------
        y += 3
        if self.t_laser.value or not self.t_shield.value:
            msg, c = "DECOHERED (unmeasurable)", Palette.DANGER
        else:
            msg, c = "SHIELDED (measurable)", Palette.OK
        y = self._row(surf, y, "Dark matter", msg, c)
        y = self._row(surf, y, "Superradiant / total",
                      f"{int(np.sum(sim.dm.superradiant))} / {sim.dm.count}",
                      Palette.WARN)
        y = self._row(surf, y, "Bodies / FPS", f"{sim.pop.count} / {sim.fps:.0f}",
                      Palette.OK if sim.fps > 50 else Palette.WARN)
        surf.blit(f.tiny.render(ShipMode.NAMES[sim.ship.mode], True, Palette.WARP),
                  (self.x, y + 1))


# ==============================================================================
#  SECTION 22 -- SIMULATION
#
#  Owns every subsystem and fixes the per-frame order of operations.  That order
#  matters: the gravitational source table must be rebuilt before any population
#  is stepped, or bodies integrate against last frame's geometry.
# ==============================================================================

class Simulation:
    def __init__(self, camera: Camera, fonts: Fonts, panel: ControlPanel,
                 fog: DustFog, sky: SkySurvey, viewport: pygame.Rect):
        self.hole = BlackHole(CFG.bh_mass, CFG.bh_spin)
        self.cosmos = QuintessenceField(CFG.hubble)
        self.gravity = GravityField(self.hole, self.cosmos)
        self.bubble = WarpBubble()
        self.gravity.bubble = self.bubble

        self.pop = BodyPopulation(self.gravity)
        self.dm = DarkMatterField(self.gravity, self.hole)
        self.ship = Spacecraft(self.gravity, self.hole, self.bubble)

        self.camera = camera
        self.sky = sky
        self.lens = LensModel(self.hole, CFG.lens_depth)
        self.lens_renderer = LensRenderer(sky, self.lens, viewport)
        self.contamination = ContaminationField(viewport.width, viewport.height)
        self.overlay = Overlay(camera, fonts)
        self.scale = PhysicalScale()
        self.panel = panel
        self.fog = fog

        self.time = 0.0
        self.paused = False
        self.fps = 60.0
        self.focus_index = -1
        self.lensing_snr = 0.0
        self.populate_default()

    # ------------------------------------------------------------- scenarios
    def populate_default(self) -> None:
        self.pop.clear()
        self.dm.clear()
        h = self.hole
        self.pop.spawn_orbiting(h, h.r_isco * 2.4, 1.4, angle=0.6)
        self.pop.spawn_orbiting(h, h.r_isco * 3.6, 9.0, angle=2.9)
        self.pop.spawn_orbiting(h, h.r_isco * 5.1, 240.0, angle=4.6)
        self.pop.spawn_orbiting(h, h.r_isco * 6.8, 0.6, angle=1.9)
        self.dm.seed_halo(500, h.r_ergosphere * 1.05, h.r_isco * 7.5)
        self.focus_index = 1

    # ------------------------------------------------------------- accessors
    @property
    def shielded(self) -> bool:
        return self.panel.t_shield.value

    @property
    def laser_on(self) -> bool:
        return self.panel.t_laser.value

    @property
    def em_probing(self) -> bool:
        """Any electromagnetic interrogation of the dark sector."""
        return self.laser_on or not self.shielded

    @property
    def contamination_level(self) -> float:
        """
        Additive photon-noise pedestal.

        An active laser sensor floods the array; simply being unshielded admits
        a smaller but non-zero ambient background.  Deep shielding is the only
        state with none.
        """
        if self.laser_on:
            return 0.85
        return 0.0 if self.shielded else 0.06

    # ---------------------------------------------------------------- update
    def sync_controls(self) -> None:
        p = self.panel
        self.hole.mass = p.s_mass.value
        self.hole.spin = p.s_spin.value
        self.lens.depth = p.s_depth.value
        self.cosmos.hubble = p.s_hubble.value
        self.overlay.show_labels = p.t_labels.value
        self.overlay.show_trails = p.t_trails.value
        self.fog.enabled = p.t_fog.value
        # Shielding cuts every light-rendering vector; an active laser is itself
        # light, so it overrides the shielding for rendering purposes.
        self.overlay.light_enabled = (not self.shielded) or self.laser_on
        if p.r_mode.index != self.ship.mode:
            self.ship.set_mode(p.r_mode.index)
        if not p.t_dm.value and self.dm.count:
            self.dm.clear()

    def step(self, dt: float) -> None:
        if dt <= 0.0:
            return
        sub = RK4Integrator.substeps_for(dt)

        # 1. Rebuild the source table (bodies only; the ship and dark matter are
        #    test particles and do not source curvature here).
        pos, mass, soft = self.pop.gravitational_sources()
        self.gravity.set_sources(pos, mass, soft)

        # 2. Integrate every population against the same field.
        self.pop.step(self.time, dt, sub)
        self.dm.step(self.time, dt, sub, self.em_probing, self.panel.t_super.value)
        self.ship.step(self.time, dt, sub)

        # 3. Structural and tidal bookkeeping.
        self.pop.refresh_structure()
        if self.pop.update_spaghettification(self.hole, dt, self.bubble):
            self.focus_index = -1

        # 4. The vacuum evolves.
        self.cosmos.advance(dt)
        self.hole.phase += dt
        self.time += dt

        self.pop.record_trails()
        self.ship.record_trail()

        # 5. Optional chase camera.  The hole stays pinned to the world origin;
        #    it is the viewport that moves, so no physics is affected either way.
        if self.panel.t_follow.value:
            self.camera.center += (self.ship.position - self.camera.center) \
                * min(1.0, 5.0 * dt)
        elif float(norm(self.camera.center)) > 1e-9:
            self.camera.center *= max(0.0, 1.0 - 3.0 * dt)

    # ---------------------------------------------------------------- render
    def draw(self, surf: pygame.Surface) -> None:
        distortion = self.panel.t_distort.value or (self.shielded and not self.laser_on)

        # 1. The ray-traced field: this IS the background, not a backdrop.
        lensed = self.lens_renderer.trace(self.camera, self.cosmos, self.bubble,
                                          distortion)
        surf.blit(lensed, (0, 0))

        # 2. Detection statistics, measured on the traced pixels themselves.
        contrast = self.lens_renderer.measure_arc_contrast()
        self.lensing_snr = ContaminationField.snr(contrast, self.contamination_level)

        # 3. Electromagnetic contamination buries the arcs without touching the
        #    physics that produced them.
        if not distortion:
            self.contamination.draw(surf, self.contamination_level)

        o = self.overlay
        o.draw_boundaries(surf, self.hole, self.lens,
                          self.lens_renderer.einstein_radius)
        o.draw_bodies(surf, self.pop, self.hole, self.fog)
        o.draw_dark_matter(surf, self.dm, self.shielded, self.laser_on)
        o.draw_ship(surf, self.ship, self.bubble)

        # 4. Dust is electromagnetic: it exists only on the light path.
        if o.light_enabled and not distortion:
            self.fog.draw(surf)


# ==============================================================================
#  SECTION 23 -- ENGINE  (window, input, main loop)
# ==============================================================================

class LensingLab:
    def __init__(self, headless: bool = False, fits_path: str | None = None,
                 quiet: bool = False):
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        # Only the subsystems actually used: initializing the mixer on a machine
        # with no sound card emits a wall of ALSA warnings for no benefit.
        pygame.display.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((CFG.width, CFG.height),
                                              pygame.DOUBLEBUF)
        pygame.display.set_caption(CFG.caption)
        self.clock = pygame.time.Clock()
        self.fonts = Fonts()

        self.canvas_rect = pygame.Rect(0, 0, CFG.width - CFG.sidebar_w, CFG.height)
        self.panel_rect = pygame.Rect(self.canvas_rect.right, 0, CFG.sidebar_w,
                                      CFG.height)
        self.canvas = pygame.Surface(self.canvas_rect.size).convert(32, 0)
        self.camera = Camera(self.canvas_rect, scale=1.0)
        self.panel = ControlPanel(self.panel_rect, self.fonts)
        self.fog = DustFog(self.canvas_rect.width, self.canvas_rect.height)
        self.sky = SkySurvey(fits_path, quiet=quiet)
        self.sim = Simulation(self.camera, self.fonts, self.panel, self.fog,
                              self.sky, self.canvas_rect)

        self.running = True
        self.panning = False
        self.dragging_mass = False
        self.toast = ""
        self.toast_t = 0.0

    def say(self, msg: str) -> None:
        self.toast, self.toast_t = msg, 2.8

    # ----------------------------------------------------------------- input
    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                continue
            if self.panel.handle(event):
                self._panel_side_effects()
                continue
            if event.type == pygame.KEYDOWN:
                self._on_key(event)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._on_mouse_down(event)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.dragging_mass = False
                elif event.button == 3:
                    self.panning = False
            elif event.type == pygame.MOUSEMOTION and self.panning:
                d = np.array(event.rel, dtype=np.float64) / self.camera.scale
                self.camera.center += np.array([-d[0], d[1]])
            elif event.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                if self.canvas_rect.collidepoint(mx, my):
                    self.camera.zoom_at(mx, my, 1.12 ** event.y)

    def _panel_side_effects(self) -> None:
        if self.panel.b_pause.consume():
            self.sim.paused = not self.sim.paused
            self.panel.b_pause.label = "RESUME" if self.sim.paused else "PAUSE"
        if self.panel.b_reset.consume():
            self.sim.ship.reset()
            self.panel.r_mode.index = self.sim.ship.mode
            self.say("Ship reset to a circular orbit at 3.1 x ISCO")
        if self.panel.b_clear.consume():
            self.sim.pop.clear()
            self.sim.focus_index = -1

    def _on_key(self, event) -> None:
        k, p = event.key, self.panel
        if k == pygame.K_ESCAPE:
            self.running = False
        elif k == pygame.K_SPACE:
            self.sim.paused = not self.sim.paused
            p.b_pause.label = "RESUME" if self.sim.paused else "PAUSE"
        elif k == pygame.K_w:
            p.r_mode.index = ShipMode.WARP
            self.say("Alcubierre drive engaged: image contracts ahead, expands behind")
        elif k == pygame.K_s:
            p.r_mode.index = ShipMode.SLINGSHOT
            self.say("Slingshot / Halo Drive: thrust with the arrow keys")
        elif k == pygame.K_b:
            p.t_labels.value = not p.t_labels.value
        elif k == pygame.K_t:
            p.t_trails.value = not p.t_trails.value
        elif k == pygame.K_m:
            p.t_distort.value = not p.t_distort.value
            self.say("Magnification map: log10|mu|, cyan = critical curve"
                     if p.t_distort.value else "Returned to the survey image")
        elif k == pygame.K_z:
            p.t_fog.value = not p.t_fog.value
        elif k == pygame.K_f:
            p.t_follow.value = not p.t_follow.value
        elif k == pygame.K_d:
            p.t_dm.value = not p.t_dm.value
            if p.t_dm.value:
                self.sim.dm.seed_halo(500, self.sim.hole.r_ergosphere * 1.05,
                                      self.sim.hole.r_isco * 7.5)
        elif k == pygame.K_u:
            p.t_shield.value = not p.t_shield.value
            self.say("Deep shielding ON: light vectors off, dark sector measurable"
                     if p.t_shield.value else
                     "Shielding lifted: ambient photons contaminate the array")
        elif k == pygame.K_l:
            p.t_laser.value = not p.t_laser.value
            self.say("Laser sensor ON: photon flood burying the lensing signal"
                     if p.t_laser.value else "Laser sensor OFF")
        elif k == pygame.K_r:
            self.sim.ship.reset()
            p.r_mode.index = self.sim.ship.mode
        elif k == pygame.K_c:
            self.sim.pop.clear()
            self.sim.focus_index = -1
        elif k in (pygame.K_1, pygame.K_2, pygame.K_3):
            p.r_spawn.index = k - pygame.K_1

    def _on_mouse_down(self, event) -> None:
        if not self.canvas_rect.collidepoint(event.pos):
            return
        if event.button == 3:
            self.panning = True
            return
        if event.button != 1:
            return
        world = self.camera.to_world(*event.pos)
        idx = self.sim.pop.add_mass_at(world, 0.0, grab_radius=22.0)
        if idx >= 0:
            self.sim.focus_index = idx
            self.dragging_mass = True
            return
        self._spawn_at(world)

    def _spawn_at(self, world: np.ndarray) -> None:
        mode = self.panel.r_spawn.index
        hole = self.sim.hole
        r = max(float(norm(world)), hole.r_s * 1.4)
        if mode == 2:
            self.sim.dm.seed_halo(90, r * 0.86, r * 1.16)
            self.panel.t_dm.value = True
            self.say("Dark matter injected -- invisible unless deep-shielded")
            return
        v = min(float(hole.circular_speed(r)), 0.9)
        tangent = np.array([-world[1], world[0]]) / max(r, CFG.eps)
        self.sim.pop.spawn(world, tangent * v, 2.0 if mode == 0 else 320.0)
        self.sim.focus_index = self.sim.pop.count - 1

    def _continuous_input(self, dt_real: float) -> None:
        keys = pygame.key.get_pressed()
        ship = self.sim.ship
        if ship.mode == ShipMode.WARP:
            turn = 0.0
            if keys[pygame.K_LEFT]:
                turn += 1.9 * dt_real
            if keys[pygame.K_RIGHT]:
                turn -= 1.9 * dt_real
            if turn:
                ship.steer(turn)
            ship.set_thrust(np.zeros(2))
        else:
            d = np.zeros(2)
            if keys[pygame.K_LEFT]:
                d[0] -= 1
            if keys[pygame.K_RIGHT]:
                d[0] += 1
            if keys[pygame.K_UP]:
                d[1] += 1
            if keys[pygame.K_DOWN]:
                d[1] -= 1
            ship.set_thrust(d)

        if self.dragging_mass and pygame.mouse.get_pressed()[0]:
            i = self.sim.focus_index
            if 0 <= i < self.sim.pop.count:
                # Growth is proportional to current mass, so the whole ladder
                # from 1 MJ to 3 Msun is reachable in a few seconds of dragging.
                m = self.sim.pop.mass_mj[i]
                self.sim.pop.mass_mj[i] = min(m * (1.0 + 2.4 * dt_real) + 0.35,
                                              CFG.max_body_mj)
                self.sim.pop.refresh_structure()

    # ------------------------------------------------------------- main loop
    def frame(self, dt_real: float) -> None:
        self.handle_events()
        self._continuous_input(dt_real)
        self.sim.sync_controls()
        self.sim.fps = self.clock.get_fps()
        if not self.sim.paused:
            self.sim.step(self.panel.s_dt.value)

        self.sim.draw(self.canvas)
        self.screen.blit(self.canvas, (0, 0))
        self.panel.draw(self.screen, self.sim)
        self._overlay_text()
        if self.toast_t > 0.0:
            self.toast_t -= dt_real
            self.screen.blit(self.fonts.body.render(self.toast, True, Palette.WARN),
                             (16, CFG.height - 28))

    def _overlay_text(self) -> None:
        f = self.fonts
        sim = self.sim
        lines = [("PAUSED" if sim.paused else "RUNNING",
                  Palette.WARN if sim.paused else Palette.OK),
                 (f"t = {sim.time:8.1f}   a(t) = {sim.cosmos.scale_factor:6.3f}"
                  f"   src: {self.sky.source_name}", Palette.TEXT_DIM)]
        if sim.panel.t_distort.value or (sim.shielded and not sim.laser_on):
            lines.append(("MAGNIFICATION MAP -- log10|mu|, cyan = critical curve",
                          Palette.EINSTEIN))
        if sim.shielded and not sim.laser_on:
            lines.append(("DEEP UNDERGROUND SHIELDING -- all light vectors off",
                          Palette.DARK_MATTER))
        if sim.laser_on:
            lines.append((f"LASER SENSOR ACTIVE -- photon flood, arc SNR = "
                          f"{sim.lensing_snr:.2f}", Palette.DANGER))
        if sim.bubble.active:
            lines.append((f"WARP FIELD  v_s = {sim.ship.warp_speed:.2f} c "
                          f"(coordinate; local v = 0)", Palette.WARP))
        # A dark plate behind the status strip: the traced field can be bright,
        # and unreadable telemetry is worse than none.
        widest = max(f.small.render(t, True, c).get_width() for t, c in lines)
        plate = pygame.Surface((widest + 16, len(lines) * 16 + 8), pygame.SRCALPHA)
        plate.fill((3, 5, 12, 214))
        self.screen.blit(plate, (8, 6))
        y = 11
        for text, col in lines:
            self.screen.blit(f.small.render(text, True, col), (14, y))
            y += 16

    def run(self) -> None:
        while self.running:
            dt_real = self.clock.tick(CFG.target_fps) / 1000.0
            self.frame(min(dt_real, 0.1))
            pygame.display.flip()
        pygame.quit()


# ==============================================================================
#  SECTION 24 -- VERIFICATION
#
#  Not tests for their own sake: each one is the evidence for a claim made in the
#  comments above.  Every check compares against a closed-form result, an
#  independent numerical method, or an exactly-known limit.
# ==============================================================================

class SelfTest:
    def __init__(self):
        self.failures: list[str] = []
        self.lines: list[str] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name}"
                          + (f"   {detail}" if detail else ""))
        if not ok:
            self.failures.append(name)

    def near(self, name: str, got: float, want: float, tol: float) -> None:
        self.check(name, abs(got - want) <= tol,
                   f"got {got:.8g}, expected {want:.8g} (tol {tol:g})")

    # -------------------------------------------------------------- deflection
    def test_deflection(self) -> None:
        """
        The exact deflection is validated three ways: against Einstein's
        weak-field limit, against its known second-order coefficient, and
        against an INDEPENDENT numerical quadrature of the null geodesic.
        """
        self.lines.append("Exact light deflection (Darwin elliptic integrals)")
        from scipy.integrate import quad
        M = 1.0
        tab = DeflectionTable(M)

        self.near("b_c = 3 sqrt(3) M", tab.b_critical, 3 * math.sqrt(3), 1e-12)
        self.near("b(r0 = 3M) = b_c",
                  float(tab.impact_parameter(np.array([3.0 * (1 + 1e-12)]), M)[0]),
                  3 * math.sqrt(3), 1e-5)
        self.near("shadow radius = 2.598 r_s", tab.b_critical / (2 * M),
                  2.5980762, 1e-6)

        # Independent check: alpha = 2 * int_0^{1/r0} du/sqrt(1/b^2 - u^2 + 2Mu^3) - pi
        worst = 0.0
        for r0 in (3.5, 5.0, 10.0, 40.0, 300.0):
            b = float(tab.impact_parameter(np.array([r0]), M)[0])
            f = lambda uu: 1.0 / math.sqrt(max(1 / b ** 2 - uu * uu
                                               + 2 * M * uu ** 3, 1e-300))
            num = 2.0 * quad(f, 0, 1 / r0, limit=400, points=[1 / r0])[0] - math.pi
            exact = float(tab.deflection_exact(np.array([r0]), M)[0])
            worst = max(worst, abs(num - exact))
        self.check("matches independent numerical quadrature", worst < 1e-6,
                   f"max |difference| = {worst:.3e} rad over 5 radii")

        # Weak field: alpha -> 4M/b, with second-order coefficient 15 pi M/(4 b^2)
        for b, tol in ((1e4, 2e-6), (1e6, 2e-8)):
            ratio = float(tab(np.array([b]))[0]) / (4 * M / b)
            self.near(f"alpha/(4M/b) at b={b:g}M -> 1 + 15 pi M/16b", ratio,
                      1.0 + 15.0 * math.pi / 16.0 * M / b, tol)

        # Strong field: alpha ~ -ln(b/b_c - 1).  Because b has a MINIMUM at the
        # photon sphere, b/b_c - 1 grows quadratically in the offset d = r0/3M - 1,
        # so each decade of d must add exactly 2 ln 10 to the deflection.
        alphas = [float(tab.deflection_exact(np.array([3.0 * (1 + d)]), M)[0])
                  for d in (1e-4, 1e-5, 1e-6, 1e-7)]
        gaps = np.diff(alphas)
        self.check("diverges as -ln(b/b_c - 1) at the photon sphere",
                   alphas[0] > 2 * math.pi
                   and float(np.max(np.abs(gaps - 2 * math.log(10)))) < 0.02,
                   f"alpha {alphas[0]:.3f} -> {alphas[-1]:.3f}; per-decade gaps "
                   + ", ".join(f"{g:.4f}" for g in gaps)
                   + f" vs 2ln10 = {2 * math.log(10):.4f}")

        # The far-field fallback must agree with the table AT the crossover --
        # compared at the same b, since alpha has a genuine slope there.
        edge = float(tab.b[-1])
        pn = 4.0 * M / edge + 15.0 * math.pi * M * M / (4.0 * edge * edge)
        gap = abs(float(tab(np.array([edge]))[0]) - pn)
        self.check("weak-field fallback joins the table continuously",
                   gap < 1e-8, f"mismatch {gap:.2e} rad = {gap / pn:.1e} relative "
                               f"at b = {edge:.0f} M")
        self.check("deflection decays to zero at large b",
                   float(tab(np.array([1e12]))[0]) < 1e-11)
        self.check("interpolated table reproduces the closed form",
                   float(np.max(np.abs(
                       tab(tab.b) - tab.alpha))) < 1e-9)

    # ------------------------------------------------------------- lens model
    def test_lens_model(self) -> None:
        self.lines.append("Lens equation, Einstein ring and magnification")
        hole = BlackHole(mass=26.0, spin=0.0)
        lens = LensModel(hole, depth=190.0)
        rE = lens.einstein_radius()
        self.check("Einstein radius found", rE > 0, f"theta_E = {rE:.4f} px")
        self.near("r_source(theta_E) = 0",
                  float(lens.radial_map(np.array([rE]))[0]), 0.0, 1e-7)

        # Parity: the image inside the critical curve must be inverted.
        inside = float(lens.magnification(np.array([rE * 0.9]))[0])
        outside = float(lens.magnification(np.array([rE * 1.1]))[0])
        self.check("magnification flips parity across the critical curve",
                   inside < 0 < outside, f"mu = {inside:.3f} inside, "
                                         f"{outside:.3f} outside")
        near = abs(float(lens.magnification(np.array([rE * 1.0005]))[0]))
        far = abs(float(lens.magnification(np.array([rE * 12.0]))[0]))
        self.check("magnification diverges on the critical curve and -> 1 far away",
                   near > 100.0 and abs(far - 1.0) < 0.05,
                   f"|mu| = {near:.1f} at the ring, {far:.4f} far out")

        # Weak-field Einstein radius must approach sqrt(2 r_s D) when the ring
        # forms far outside the strong-field region.
        weak = LensModel(BlackHole(mass=0.05, spin=0.0), depth=4000.0)
        got = weak.einstein_radius()
        want = math.sqrt(2 * weak.hole.r_s * weak.depth)
        self.check("weak-field limit -> sqrt(2 r_s D_eff)",
                   abs(got - want) / want < 0.02,
                   f"{got:.3f} vs {want:.3f} ({abs(got - want) / want * 100:.2f}%)")

        # Frame dragging: azimuthal only, falls as 1/r^3, vanishes at zero spin.
        spun = LensModel(BlackHole(mass=26.0, spin=0.9), depth=190.0)
        t1 = float(spun._drag_twist(np.array([600.0]))[0])
        t2 = float(spun._drag_twist(np.array([1200.0]))[0])
        self.near("frame-drag twist scales as 1/r^3", t1 / t2, 8.0, 0.05)
        self.check("no twist for a non-spinning hole",
                   float(lens._drag_twist(np.array([400.0]))[0]) == 0.0)

        # Stretch factors must reproduce 1/|mu|.
        r = np.array([300.0, 500.0, 900.0])
        rad, tan = spun.stretch_factors(r)
        self.check("radial x tangential stretch = 1/|mu|",
                   float(np.max(np.abs(rad * tan
                                       - 1.0 / np.abs(spun.magnification(r))))) < 1e-6)

    def test_ray_tracer(self) -> None:
        """The traced field must be finite, shadowed inside b_c, and fast."""
        self.lines.append("Ray tracer")
        sky = SkySurvey(quiet=True)
        hole = BlackHole(26.0, 0.85)
        lens = LensModel(hole, 190.0)
        vp = pygame.Rect(0, 0, 1260, 900)
        cam = Camera(vp, 1.0)
        cos = QuintessenceField(0.0)
        lr = LensRenderer(sky, lens, vp)
        lr.ensure_map(cam)
        self.check("source coordinates all finite",
                   bool(np.isfinite(lr.src_x).all() and np.isfinite(lr.src_y).all()))
        self.check("captured region matches b < b_c", bool(lr.captured.any()))

        # Every ray inside b_c must be captured, and none outside it.
        wx = lr._px[:, None] / cam.scale
        wy = -lr._py[None, :] / cam.scale
        r = np.hypot(np.broadcast_to(wx, (lr.w, lr.h)),
                     np.broadcast_to(wy, (lr.w, lr.h)))
        self.check("source rasters have the raster shape",
                   lr.src_x.shape == (lr.w, lr.h) and lr.src_y.shape == (lr.w, lr.h))
        bc = lens.table.b_critical
        self.check("shadow is exactly the b < b_c disc",
                   bool(np.all(lr.captured == (r < bc))))

        lr.trace(cam, cos, None, False)
        self.check("shadow pixels render black",
                   int(lr._out[lr.captured].max()) == 0)
        self.check("mip levels selected inside the pyramid",
                   0 <= int(lr.mip_offset.min())
                   and int(lr.mip_offset.max())
                   <= (sky.n_mips - 1) * sky.level_stride)
        self.check("mip level rises toward the shadow where compression is worst",
                   int(lr.mip_offset[lr.w // 2, lr.h // 2]) >= 0)
        # Sources far from the lens must be essentially undisplaced.
        far = r > 3000.0
        if far.any():
            resid = np.hypot(lr.src_x[far] - np.broadcast_to(wx, (lr.w, lr.h))[far],
                             lr.src_y[far] - np.broadcast_to(wy, (lr.w, lr.h))[far])
            del far
            self.check("far field is undeflected", float(np.max(resid)) < 30.0,
                       f"max residual {float(np.max(resid)):.2f} px")

        import time
        lr.trace(cam, cos, None, False)
        t0 = time.perf_counter()
        for _ in range(40):
            lr.trace(cam, cos, None, False)
        ms = (time.perf_counter() - t0) / 40 * 1000
        self.check("trace within the frame budget", ms < 8.0, f"{ms:.2f} ms/frame")

    # ---------------------------------------------------------------- ingest
    def test_ingestion(self) -> None:
        self.lines.append("FITS ingestion pipeline (astropy)")
        sky = SkySurvey(quiet=True)
        self.check("image loaded", sky.data.size > 0 and np.isfinite(sky.data).all(),
                   f"{sky.data.shape} from {sky.source_name}")
        self.check("WCS parsed", sky.wcs is not None)
        self.check("sky coordinates resolve", "RA" in sky.sky_coord(100, 100))
        self.check("mip pyramid built",
                   sky.flat.size == sky.n_mips * sky.level_stride + 1)
        self.check("black sentinel is black", int(sky.flat[sky.black_index]) == 0)
        self.check("packed layout matches a 32-bit surface",
                   sky.packed_wh.shape == (sky.sw, sky.sh))

        # Round-trip an arbitrary FITS through the loader, including a cube.
        tmp = "._selftest_cube.fits"
        cube = np.random.default_rng(3).normal(50, 5, (4, 64, 80)).astype(np.float32)
        fits.PrimaryHDU(cube).writeto(tmp, overwrite=True)
        try:
            s2 = SkySurvey(tmp, quiet=True)
            self.check("collapses a 3-D cube to 2-D",
                       s2.data.ndim == 2 and s2.data.shape == (CFG.sky_h, CFG.sky_w))
        finally:
            os.remove(tmp)

        # Colour packing must be lossless.
        rgb = np.random.default_rng(5).integers(0, 256, (7, 11, 3), dtype=np.uint8)
        p = pack_rgb(rgb)
        back = np.stack([(p >> 16) & 255, (p >> 8) & 255, p & 255], axis=-1)
        self.check("pack_rgb round-trips exactly", bool(np.array_equal(rgb, back)))

    # ------------------------------------------------------------ integrator
    def test_integrator(self) -> None:
        self.lines.append("RK4 integrator and relativistic dynamics")
        hole = BlackHole(mass=10.0, spin=0.0)
        g = GravityField(hole, QuintessenceField(0.0))
        integ = RK4Integrator(g.make_derivative(False, False))
        r0 = 400.0
        v0 = math.sqrt(hole.mass * r0) / (r0 - hole.r_s)     # PW circular, Newtonian
        s0 = np.array([[r0, 0.0, 0.0, v0]])
        period = math.tau * r0 / v0
        errs = [abs(float(norm(integ.integrate(s0.copy(), 0.0, period, n)[0, 0:2]))
                    - r0) for n in (400, 800, 1600)]
        r1, r2 = errs[0] / max(errs[1], 1e-18), errs[1] / max(errs[2], 1e-18)
        self.check("error falls ~16x per step halving (4th order)",
                   8 < r1 < 40 and 8 < r2 < 40,
                   f"ratios {r1:.1f}, {r2:.1f}; {errs[0]:.2e} -> {errs[2]:.2e}")

        # Relativistic circular orbits: stacked, integrated together.
        hole2 = BlackHole(26.0, 0.0)
        g2 = GravityField(hole2, QuintessenceField(0.0))
        i2 = RK4Integrator(g2.make_derivative(False, True))
        radii = np.array([200.0, 300.0, 420.0])
        vs = np.array([float(hole2.circular_speed(r)) for r in radii])
        st = np.stack([radii, np.zeros(3), np.zeros(3), vs], axis=1)
        periods = math.tau * radii / vs
        worst = 0.0
        for i in range(int(6 * float(periods.max()))):
            st = i2.step(st, 0.0, 1.0)
            if i % 29 == 0:
                worst = max(worst, float(np.max(np.abs(norm(st[:, 0:2]) - radii)
                                                / radii)))
        laps = 6 * periods.max() / periods
        self.check(f"orbit closes over {laps.min():.0f}-{laps.max():.0f} full laps",
                   worst < 1e-6, f"max fractional excursion {worst:.2e}")

        # c is an asymptote, never reached, with no clamp.
        v = np.array([[0.0, 0.0]])
        acc = np.array([[0.05, 0.0]])
        for _ in range(200000):
            v = v + GravityField.proper_accel(v, acc) * 0.05
        self.check("speed asymptotes below c under huge sustained thrust",
                   0.999 < float(norm(v[0])) < 1.0, f"v = {float(norm(v[0])):.9f} c")
        vv = np.array([[0.9, 0.0]])
        gam = float(lorentz_gamma(vv)[0])
        self.near("longitudinal response = 1/gamma^3",
                  float(GravityField.proper_accel(vv, np.array([[1.0, 0.0]]))[0, 0]),
                  1 / gam ** 3, 1e-9)
        self.near("transverse response = 1/gamma",
                  float(GravityField.proper_accel(vv, np.array([[0.0, 1.0]]))[0, 1]),
                  1 / gam, 1e-9)

    def test_isco(self) -> None:
        """
        The defining property of Paczynski-Wiita: its marginally stable orbit is
        at exactly 6M.  For V(r) = L^2/2r^2 - M/(r - r_s) with the circular
        condition L^2 = M r^3/(r - r_s)^2,
            d2V/dr2 = M/(r - r_s)^2 [3/r - 2/(r - r_s)]
        which changes sign at 3(r - r_s) = 2r, i.e. r = 3 r_s = 6M.
        """
        self.lines.append("ISCO and marginal stability")
        M = 26.0
        hole = BlackHole(M, 0.0)
        rs = hole.r_s
        d2V = lambda r: M / (r - rs) ** 2 * (3.0 / r - 2.0 / (r - rs))
        self.check("d2V/dr2 > 0 outside 6M", d2V(6 * M + 1e-6) > 0)
        self.check("d2V/dr2 < 0 inside 6M", d2V(6 * M - 1e-6) < 0)
        self.near("marginally stable orbit = 6M", brentq(d2V, 2.2 * rs, 10 * rs,
                                                         xtol=1e-12), 6 * M, 1e-9)
        self.near("BlackHole.r_isco agrees at zero spin", hole.r_isco, 6 * M, 1e-9)

        g = GravityField(hole, QuintessenceField(0.0))
        integ = RK4Integrator(g.make_derivative(False, True))
        # Identical 0.1% inward nudge: unstable inside the ISCO, bound outside.
        for fac, should_plunge in ((0.92, True), (1.60, False)):
            r0 = hole.r_isco * fac
            st = np.array([[r0, 0.0, 0.0, float(hole.circular_speed(r0)) * 0.999]])
            rmin = r0
            for _ in range(9000):
                st = integ.step(st, 0.0, 0.5)
                rmin = min(rmin, float(norm(st[0, 0:2])))
                if rmin < rs:
                    break
            self.check(f"0.1% nudge at {fac:.2f}x ISCO "
                       f"{'plunges' if should_plunge else 'stays bound'}",
                       (rmin < rs) == should_plunge, f"min r = {rmin:.2f}")

    def test_kerr(self) -> None:
        self.lines.append("Kerr boundary radii")
        h = BlackHole(10.0, 0.0)
        self.near("Schwarzschild horizon = 2M", h.r_horizon, 20.0, 1e-9)
        self.near("photon sphere = 3M", h.r_photon, 30.0, 1e-6)
        self.near("ISCO = 6M", h.r_isco, 60.0, 1e-6)
        e = BlackHole(10.0, 0.999999)
        self.near("horizon = M + sqrt(M^2 - a^2)", e.r_horizon,
                  10 * (1 + math.sqrt(1 - e.spin ** 2)), 1e-9)
        self.near("retrograde photon orbit closed form", e.r_photon_retro,
                  20 * (1 + math.cos(2 / 3 * math.acos(e.spin))), 1e-9)
        spins = np.linspace(0.0, 0.99, 40)
        hs = [BlackHole(10.0, float(a)) for a in spins]
        self.check("prograde radii decrease monotonically with spin",
                   all(np.all(np.diff([getattr(x, k) for x in hs]) < 1e-12)
                       for k in ("r_horizon", "r_photon", "r_isco")))

    def test_time_dilation(self) -> None:
        self.lines.append("Schwarzschild time dilation")
        hole = BlackHole(20.0, 0.0)
        g = GravityField(hole, QuintessenceField(0.0))
        ship = Spacecraft(g, hole, WarpBubble())
        ship.mode = ShipMode.BALLISTIC
        for r, v in ((200.0, 0.0), (60.0, 0.0), (44.0, 0.6)):
            ship.state = np.array([[r, 0.0, 0.0, v]])
            self.near(f"r={r:g}, v={v:g}", ship.dilation_factor(),
                      math.sqrt(1 - hole.r_s / r) * math.sqrt(1 - v * v), 1e-9)
        ship.state = np.array([[hole.r_s * 1.0005, 0.0, 0.0, 0.0]])
        self.check("clock nearly frozen at the horizon",
                   ship.dilation_factor() < 0.03, f"{ship.dilation_factor():.5f}")
        s2 = Spacecraft(g, hole, WarpBubble())
        s2.set_mode(ShipMode.WARP)          # engaging the drive raises the bubble
        s2.state = np.array([[hole.r_s * 1.01, 0.0, 0.0, 0.0]])
        self.check("engaging warp raises the bubble", s2.bubble.active)
        self.near("flat bubble interior -> dtau/dt = 1", s2.dilation_factor(),
                  1.0, 1e-12)
        s2.set_mode(ShipMode.SLINGSHOT)
        self.check("dropping out of warp restores real dilation",
                   s2.dilation_factor() < 0.2 and not s2.bubble.active,
                   f"dtau/dt = {s2.dilation_factor():.5f}")

    def test_alcubierre(self) -> None:
        self.lines.append("Alcubierre metric")
        b = WarpBubble(radius=80.0, sigma=0.06)
        b.configure(np.zeros(2), np.array([1.0, 0.0]), 3.0)
        b.active = True
        self.near("f(0) = 1 (flat interior)", float(b.f(np.array([0.0]))[0]), 1.0, 2e-3)
        self.check("f -> 0 far outside", float(b.f(np.array([600.0]))[0]) < 1e-6)
        self.check("df/dr_s < 0 everywhere",
                   bool(np.all(b.df(np.linspace(1, 400, 500)) < 0)))
        ahead = float(b.expansion_scalar(np.array([[80.0, 0.0]]))[0])
        behind = float(b.expansion_scalar(np.array([[-80.0, 0.0]]))[0])
        self.check("theta < 0 ahead (contraction)", ahead < 0, f"{ahead:.3e}")
        self.check("theta > 0 behind (expansion)", behind > 0, f"{behind:.3e}")
        self.near("theta antisymmetric about the ship", ahead, -behind, 1e-12)
        self.check("tidal screening ~0 inside, ~1 outside",
                   float(b.exterior_mask(np.array([[0.0, 0.0]]))[0, 0]) < 5e-3 and
                   float(b.exterior_mask(np.array([[400.0, 0.0]]))[0, 0]) > 0.999)

    def test_smoothstep_and_halo(self) -> None:
        self.lines.append("Interpolation helper and halo-drive window")
        self.near("descending edges invert the ramp", float(smoothstep(8, 2, 8)),
                  0.0, 0)
        self.near("descending: at edge1 -> 1", float(smoothstep(8, 2, 2)), 1.0, 0)
        self.near("descending: beyond edge0 -> 0", float(smoothstep(8, 2, 20)),
                  0.0, 0)
        self.near("ascending midpoint -> 0.5", float(smoothstep(2, 8, 5)), 0.5, 1e-12)

        hole = BlackHole(26.0, 0.9)
        g = GravityField(hole, QuintessenceField(0.0))
        g.set_sources(np.zeros((0, 2)), np.zeros(0), np.zeros(0))
        ship = Spacecraft(g, hole, WarpBubble())
        ship.mode = ShipMode.SLINGSHOT
        r_ph = hole.r_photon

        def boost(r, prograde=True):
            v = float(hole.circular_speed(max(r, hole.r_s * 1.2)))
            ship.state = np.array([[r, 0.0, 0.0, v if prograde else -v]])
            return float(norm(ship._halo_drive_impulse()))

        near = boost(r_ph * 1.15)
        self.check("thrust available at the photon sphere", near > 1e-4,
                   f"{near:.5f}")
        self.check("no thrust far outside", boost(r_ph * 5) == 0.0
                   and boost(r_ph * 20) == 0.0)
        self.check("no thrust on a retrograde pass",
                   boost(r_ph * 1.15, prograde=False) == 0.0)
        slow = BlackHole(26.0, 0.0)
        g2 = GravityField(slow, QuintessenceField(0.0))
        g2.set_sources(np.zeros((0, 2)), np.zeros(0), np.zeros(0))
        s2 = Spacecraft(g2, slow, WarpBubble()); s2.mode = ShipMode.SLINGSHOT
        s2.state = np.array([[slow.r_photon * 1.15, 0, 0,
                              float(slow.circular_speed(slow.r_photon * 1.15))]])
        self.check("a non-spinning hole gives a weaker boost",
                   float(norm(s2._halo_drive_impulse())) < near)

    def test_structure(self) -> None:
        self.lines.append("Atomic equilibrium and the collapse ladder")
        S = StellarStructure
        ms = np.logspace(-2, 3.7, 8000)
        self.near("Zapolsky-Salpeter radius peaks at M_0/3^(3/4)",
                  float(ms[int(np.argmax(S.physical_radius(ms)))]),
                  S.M_ZS / 3 ** 0.75, 0.05)
        self.check("degenerate branch shrinks with mass",
                   float(S.physical_radius(np.array([1000.0]))[0])
                   < float(S.physical_radius(np.array([100.0]))[0]))
        for m, want in ((1.0, Phase.GAS_GIANT), (13.5, Phase.BROWN_DWARF),
                        (81.0, Phase.STAR), (1600.0, Phase.DEGENERATE),
                        (3200.0, Phase.COLLAPSED)):
            self.check(f"{m} MJ -> {Phase.NAMES[want]}",
                       int(S.phase_of(np.array([m]))[0]) == want)
        self.near("Chandrasekhar limit = 1.44 Msun", S.M_CHANDRA, 1.44 * 1047.57, 1e-6)
        g0, d0 = S.pressure_balance(1.0)
        g1, d1 = S.pressure_balance(2500.0)
        self.check("compression outruns EM resistance above M_Ch",
                   (g1 / d1) > (g0 / d0), f"{g0 / d0:.3e} -> {g1 / d1:.3e}")

    def test_dark_sector(self) -> None:
        self.lines.append("Dark matter rules and contamination")
        hole = BlackHole(26.0, 0.9)
        g = GravityField(hole, QuintessenceField(0.0))
        dm = DarkMatterField(g, hole)
        dm.seed_halo(200, 90.0, 400.0)
        self.check("halo seeded", dm.count > 0, f"{dm.count} particles")

        # Only gravity may deflect it -- there is no contact term to test against.
        base = dm.state.copy()
        g.set_sources(np.zeros((0, 2)), np.zeros(0), np.zeros(0))
        dm.step(0.0, 0.5, 2, False, False)
        free = dm.state.copy()
        dm.state = base.copy()
        g.set_sources(np.array([[200.0, 0.0]]), np.array([40.0]), np.array([8.0]))
        dm.step(0.0, 0.5, 2, False, False)
        moved = float(np.max(norm(dm.state[:, 0:2] - free[:, 0:2])))
        self.check("responds to mass gravitationally only", moved > 0.0,
                   f"deflection {moved:.4f} px")

        dm.coherence[:] = 1.0
        self.check("invisible when unshielded",
                   not dm.visible_mask(False, False).any())
        self.check("invisible when the laser is on",
                   not dm.visible_mask(True, True).any())
        self.check("visible only when shielded and dark",
                   dm.visible_mask(True, False).all())
        for _ in range(30):
            dm.step(0.0, 0.4, 1, True, False)
        self.check("decoheres under EM probing", float(np.max(dm.coherence)) < 0.06)
        self.check("still gravitating after decoherence", dm.count > 0)
        for _ in range(30):
            dm.step(0.0, 0.4, 1, False, False)
        self.check("re-coheres inside deep shielding",
                   float(np.max(dm.coherence)) > 0.9)

        # Superradiance must scale with spin and vanish for a static hole.
        for spin, expect in ((0.9, True), (0.0, False)):
            hh = BlackHole(26.0, spin)
            gg = GravityField(hh, QuintessenceField(0.0))
            gg.set_sources(np.zeros((0, 2)), np.zeros(0), np.zeros(0))
            d2 = DarkMatterField(gg, hh)
            n = 60
            v = float(hh.circular_speed(hh.r_ergosphere * 1.3))
            d2.state = np.tile(np.array([[hh.r_ergosphere * 0.92, 0.0, 0.0, v]]),
                               (n, 1))
            d2.coherence = np.ones(n)
            d2.superradiant = np.zeros(n, dtype=bool)
            spin0 = hh.spin
            for _ in range(8):
                d2.step(0.0, 0.25, 1, False, True)
            self.check(f"a*={spin}: Penrose extraction "
                       f"{'occurs' if expect else 'absent'}",
                       (d2.extracted_energy > 0) == expect,
                       f"extracted {d2.extracted_energy:.4f}")
            if expect:
                self.check("  hole spins down to pay for it", hh.spin < spin0,
                           f"a* {spin0:.6f} -> {hh.spin:.6f}")

        # Contamination must destroy detectability without touching the physics.
        clean = ContaminationField.snr(28.0, 0.0)
        lit = ContaminationField.snr(28.0, 0.85)
        self.check("laser flood drives arc SNR below 1", clean > 5 > 1 > lit,
                   f"SNR {clean:.2f} shielded -> {lit:.2f} contaminated")

    def test_cosmology(self) -> None:
        self.lines.append("Dark energy background")
        c = QuintessenceField(0.01)
        c.advance(100.0)
        self.near("a(t) = exp(H t)", c.scale_factor, math.exp(1.0), 1e-9)
        hole = BlackHole(26.0, 0.0)
        g = GravityField(hole, c)
        r_ta = c.turnaround_radius(hole.mass, hole.r_s)
        a = float(g.central_acceleration(np.array([[r_ta, 0.0]]))[0, 0])
        self.check("gravity exactly cancels dark energy at r_ta",
                   abs(a) < 1e-9 * hole.mass / (r_ta - hole.r_s) ** 2,
                   f"net a = {a:.3e} at r_ta = {r_ta:.1f}")
        self.check("bound inside r_ta, unbound outside",
                   float(g.central_acceleration(np.array([[r_ta * .9, 0]]))[0, 0]) < 0
                   and float(g.central_acceleration(
                       np.array([[r_ta * 1.1, 0]]))[0, 0]) > 0)
        self.near("Newtonian limit (GM/H^2)^(1/3)", c.turnaround_radius(hole.mass),
                  (26.0 / 1e-4) ** (1 / 3), 1e-9)

    def test_spaghettification(self) -> None:
        self.lines.append("Tidal disruption")
        hole = BlackHole(26.0, 0.0)
        g = GravityField(hole, QuintessenceField(0.0))
        pop = BodyPopulation(g)
        pop.spawn([hole.r_horizon * 0.55, 0.0], [0.0, 0.0], 3.0)
        self.check("body starts intact", float(pop.strain[0]) == 0.0)
        n = 0
        while pop.count and n < 4000:
            pop.update_spaghettification(hole, 0.05, None)
            n += 1
        self.check("horizon crossing -> strain -> deletion", pop.count == 0,
                   f"destroyed after {n} ticks")
        for dt_test, label in ((CFG.dt, "default dt"), (CFG.dt * 0.25, "slow-mo")):
            p3 = BodyPopulation(g)
            p3.spawn([hole.r_horizon * 0.9, 0.0], [0.0, 0.0], 3.0)
            k = 0
            while p3.count and k < 20000:
                p3.update_spaghettification(hole, dt_test, None)
                k += 1
            secs = k * dt_test / CFG.dt / 60.0
            self.check(f"disruption lasts {k} steps at {label}",
                       50 <= k <= 4000 and 0.5 < secs < 4.0,
                       f"~{secs:.2f} s of wall clock at 60 FPS")
        b = WarpBubble(radius=90.0)
        b.configure(np.array([hole.r_horizon * 0.55, 0.0]), np.array([1.0, 0.0]), 2.0)
        b.active = True
        p2 = BodyPopulation(g)
        p2.spawn([hole.r_horizon * 0.55, 0.0], [0.0, 0.0], 3.0)
        for _ in range(200):
            p2.update_spaghettification(hole, 0.05, b)
        self.check("warp bubble shelters matter from tidal death",
                   p2.count == 1 and float(p2.strain[0]) == 0.0)

    def test_physical_scale(self) -> None:
        self.lines.append("Physical unit anchoring (astropy)")
        ps = PhysicalScale(mass_msun=1.0)
        # The Sun's Schwarzschild radius is 2.953 km, a standard reference value.
        self.near("r_s(1 Msun) = 2.953 km", ps.schwarzschild_radius_m() / 1e3,
                  2.953, 1e-3)
        ps2 = PhysicalScale(mass_msun=4.297e6)
        self.near("r_s(Sgr A*) = 0.0849 AU",
                  ps2.schwarzschild_radius_m() / 1.495978707e11, 0.0849, 5e-4)
        self.check("metres per pixel scales inversely with pixel radius",
                   abs(ps2.metres_per_pixel(52.0) * 52.0
                       - ps2.schwarzschild_radius_m()) < 1.0)
        th = ps2.einstein_angle_arcsec()
        self.check("true theta_E computed from a Planck18 cosmology",
                   th is None or 1e-8 < th < 1.0,
                   "unavailable" if th is None else f"{th * 1e6:.3f} uas")
        self.check("format_length picks sensible units",
                   PhysicalScale.format_length(1.5e11).endswith("AU")
                   and PhysicalScale.format_length(2953.0).endswith("km"))

    # ------------------------------------------------------------- rendering
    def test_render(self, frames: int = 200) -> None:
        self.lines.append(f"Headless render benchmark ({frames} frames)")
        base = LensingLab(headless=True, quiet=True)
        for _ in range(25):
            base.frame(1 / 60)
        t0 = pygame.time.get_ticks()
        for _ in range(frames):
            base.frame(1 / 60)
        fps = frames / max((pygame.time.get_ticks() - t0) / 1000.0, 1e-6)
        self.check("default scene holds 60 FPS", fps >= 60.0,
                   f"{fps:.1f} FPS ({base.sim.pop.count} bodies, "
                   f"{base.sim.dm.count} DM, ray tracing every frame)")

        # Exercise the expensive paths together.
        eng = LensingLab(headless=True, quiet=True)
        eng.panel.t_fog.value = True
        eng.panel.r_mode.index = ShipMode.WARP
        for i in range(18):
            eng.sim.pop.spawn_orbiting(eng.sim.hole,
                                       eng.sim.hole.r_isco * (1.7 + 0.26 * i),
                                       2.0 + 40.0 * i, angle=i * 0.51)
        for _ in range(25):
            eng.frame(1 / 60)
        t0 = pygame.time.get_ticks()
        for _ in range(frames):
            eng.frame(1 / 60)
        fps2 = frames / max((pygame.time.get_ticks() - t0) / 1000.0, 1e-6)
        self.check("worst case holds the 16.7 ms budget", fps2 >= 60.0,
                   f"{fps2:.1f} FPS = {1000 / fps2:.2f} ms/frame "
                   f"({eng.sim.pop.count} bodies, warp + fog)")

        # Every research mode must render without error.
        for shield, laser, dist, name in ((True, False, False, "deep shielding"),
                                          (False, True, False, "laser flood"),
                                          (False, False, True, "magnification map")):
            eng.panel.t_shield.value = shield
            eng.panel.t_laser.value = laser
            eng.panel.t_distort.value = dist
            for _ in range(12):
                eng.frame(1 / 60)
            self.check(f"{name} render path stable", True,
                       f"arc SNR = {eng.sim.lensing_snr:.2f}")
        pygame.quit()

    # ------------------------------------------------------------------ run
    def run(self) -> int:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.display.init()
        pygame.font.init()
        pygame.display.set_mode((64, 64))
        np.random.seed(7)
        for fn in (self.test_deflection, self.test_lens_model, self.test_ray_tracer,
                   self.test_ingestion, self.test_integrator, self.test_isco,
                   self.test_kerr, self.test_time_dilation, self.test_alcubierre,
                   self.test_smoothstep_and_halo, self.test_structure,
                   self.test_dark_sector, self.test_cosmology,
                   self.test_spaghettification, self.test_physical_scale,
                   self.test_render):
            fn()
            self.lines.append("")
        print("\n".join(self.lines))
        if self.failures:
            print(f"{len(self.failures)} FAILURE(S): " + ", ".join(self.failures))
            return 1
        print(f"All {sum(l.strip().startswith('[PASS]') for l in self.lines)} "
              f"physics checks passed.")
        return 0


# ==============================================================================
#  SECTION 25 -- ENTRY POINT
# ==============================================================================

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Roman Lensing Lab: FITS ray-tracing and relativistic sandbox.")
    ap.add_argument("--fits", metavar="PATH", default=None,
                    help="FITS file to use as the source plane "
                         "(default: autodiscover, else synthesize)")
    ap.add_argument("--selftest", action="store_true",
                    help="run the analytic physics checks and benchmark")
    ap.add_argument("--headless", type=int, metavar="N", default=0,
                    help="run N frames with no window, then exit")
    args = ap.parse_args(argv)

    if args.selftest:
        return SelfTest().run()
    if args.headless:
        lab = LensingLab(headless=True, fits_path=args.fits)
        for _ in range(args.headless):
            lab.frame(1 / 60)
        print(f"Completed {args.headless} headless frames.")
        pygame.quit()
        return 0

    LensingLab(fits_path=args.fits).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
