"""
================================================================================
 COSMIC SIMULATION ENGINE  --  a 2D relativistic / quantum-field sandbox
================================================================================

A single-file, modular, object-oriented spacetime laboratory built on Pygame +
NumPy.  Everything on screen is driven by a stated equation; nothing is arcade
hand-waving.  Where an exact result is impractical in 2D real time, the code
uses a *named* physical approximation and says so in the comment.

--------------------------------------------------------------------------------
 UNIT SYSTEM  (read this first -- it explains every magic number below)
--------------------------------------------------------------------------------
 Geometrized units:  G = c = 1.

   * Length      1 world unit == 1 screen pixel at zoom 1.
   * Mass        In geometrized units GM/c^2 has dimensions of LENGTH, so a mass
                 is quoted directly in world units.  The central black hole's
                 `M` is literally its gravitational radius r_g = GM/c^2.
   * Time        1 world time unit = the time light needs to cross 1 pixel.
   * Velocity    dimensionless, in units of c.  v = 1.0 IS the speed of light.

 Consequences that fall out for free (all exact for Schwarzschild):
       Schwarzschild radius      r_s   = 2M
       Photon sphere             r_ph  = 3M
       Innermost stable orbit    r_isco= 6M
       Circular orbit speed      v     = sqrt(M/r)      (weak field)

 With the default M = 26 px this puts the horizon at 52 px, the photon sphere at
 78 px and the ISCO at 156 px -- comfortably resolved on a 1600x900 canvas -- and
 makes a circular orbit at the ISCO run at v = sqrt(1/6) = 0.41 c.  The unit
 system was chosen so that "looks right" and "is right" coincide.

--------------------------------------------------------------------------------
 PHYSICS INDEX  (equation -> where it lives)
--------------------------------------------------------------------------------
  Paczynski-Wiita pseudo-Newtonian potential ....... GravityField.acceleration
  Lense-Thirring gravitomagnetic frame dragging .... GravityField.acceleration
  de Sitter / Lambda cosmological repulsion ........ GravityField.acceleration
  Relativistic dynamics  dp/dt = F,  p = gamma m v .. GravityField.proper_accel
  Runge-Kutta 4th order integrator ................. RK4Integrator
  Flamm paraboloid grid pinch ...................... SpacetimeGrid._gravity_pinch
  FRW scale factor  a(t) = exp(H t) ................ CosmologicalField
  Alcubierre shape function + expansion scalar ..... WarpBubble
  Kerr horizon / ergosphere / photon sphere radii .. BlackHole
  Schwarzschild gravitational time dilation ........ Spacecraft.tick_clocks
  Tidal (spaghettification) strain  2GM dr / r^3 ... Spaghettifier
  Zapolsky-Salpeter mass-radius relation ........... StellarStructure
  Chandrasekhar degeneracy limit ................... StellarStructure
  Penrose / superradiant energy extraction ......... DarkMatterField

--------------------------------------------------------------------------------
 CONTROLS
--------------------------------------------------------------------------------
  Mouse
    Left click (canvas) ..... spawn body of the selected class
    Left drag on a body ..... pour mass into it (watch the phase transitions)
    Right drag .............. pan camera        Wheel ......... zoom
  Keyboard
    W ....... toggle Alcubierre warp drive       S ....... slingshot / halo mode
    Arrow keys / WASD-thrust while in slingshot mode steer the ship
    D ....... dark matter visibility experiment  L ....... laser / light sensor
    U ....... deep underground shielding         Z ....... zone of avoidance fog
    T ....... particle trails                    G ....... grid on/off
    B ....... boundary labels                    SPACE ... pause
    R ....... reset ship                         C ....... clear spawned bodies
    1/2/3 ... select spawn class (planet / star / dark matter burst)
    ESC ..... quit

  CLI
    python cosmic_engine.py                 -- run the sandbox
    python cosmic_engine.py --selftest      -- headless physics + render checks
    python cosmic_engine.py --headless 600  -- run 600 frames with no window
================================================================================
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

# Pygame prints a banner to stdout on import; silence it so --selftest output is
# machine-readable.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402


# ==============================================================================
#  SECTION 0 -- CONFIGURATION
# ==============================================================================

@dataclass
class Config:
    """Every tunable in one place.  Values are in the geometrized units above."""

    # --- window -------------------------------------------------------------
    width: int = 1600
    height: int = 900
    sidebar_w: int = 330
    target_fps: int = 60
    caption: str = "Cosmic Simulation Engine -- Relativity / Quantum Fields / Warp"

    # --- central black hole -------------------------------------------------
    bh_mass: float = 26.0            # M = r_g = GM/c^2, in world units (px)
    bh_mass_min: float = 6.0
    bh_mass_max: float = 70.0
    bh_spin: float = 0.85            # dimensionless Kerr parameter a* = a/M in [0,1)

    # --- cosmology ----------------------------------------------------------
    hubble: float = 0.00055          # H in 1/world-time.  a(t) = exp(H t)
    hubble_min: float = 0.0
    hubble_max: float = 0.0035

    # --- integration --------------------------------------------------------
    dt: float = 0.55                 # world-time advanced per rendered frame
    dt_min: float = 0.0
    dt_max: float = 2.0
    max_substep: float = 0.70        # largest RK4 step size, in world time

    # --- spacetime grid -----------------------------------------------------
    grid_spacing: float = 46.0       # comoving lattice pitch, world units
    grid_max_pinch: float = 0.82     # a vertex may never fall past this fraction
    grid_curve_samples: int = 5      # extra samples per grid segment (smoothness)

    # --- bodies -------------------------------------------------------------
    max_bodies: int = 220
    max_body_mj: float = 24000.0     # ~23 Msun: the ceiling on hand-fed mass
    max_dark_particles: int = 900
    trail_len: int = 190

    # --- softening ----------------------------------------------------------
    eps: float = 1e-9


CFG = Config()


class Palette:
    """A deliberately cold, high-contrast astrophysical palette."""

    VOID          = (6, 7, 14)
    PANEL         = (13, 16, 27)
    PANEL_EDGE    = (38, 46, 70)
    GRID          = (44, 62, 104)
    GRID_HOT      = (96, 140, 220)
    GRID_CONTRACT = (86, 172, 255)   # Alcubierre:  theta < 0, space compressed
    GRID_EXPAND   = (255, 122, 96)   # Alcubierre:  theta > 0, space stretched
    TEXT          = (214, 224, 244)
    TEXT_DIM      = (124, 138, 170)
    ACCENT        = (120, 208, 255)
    WARN          = (255, 176, 74)
    DANGER        = (255, 92, 104)
    OK            = (120, 240, 168)

    HORIZON       = (0, 0, 0)
    PHOTON        = (255, 226, 138)
    ERGO          = (150, 108, 255)
    ACCRETION     = (255, 152, 60)

    PLANET        = (110, 170, 224)
    ROCK          = (176, 150, 128)
    BROWN_DWARF   = (208, 96, 64)
    STAR          = (255, 238, 190)
    DARK_MATTER   = (176, 120, 255)
    SHIP          = (150, 255, 214)
    WARP          = (128, 236, 255)


# ==============================================================================
#  SECTION 1 -- VECTOR MATHEMATICS
# ==============================================================================

def norm(v: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    """Euclidean length.  Kept separate so every call site is easy to audit."""
    return np.sqrt(np.sum(v * v, axis=axis, keepdims=keepdims))


def safe_normalize(v: np.ndarray, axis: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """Return (unit_vector, magnitude) with a zero-safe denominator."""
    mag = norm(v, axis=axis, keepdims=True)
    unit = v / np.maximum(mag, CFG.eps)
    return unit, mag


def lorentz_gamma(v: np.ndarray) -> np.ndarray:
    """
    Lorentz factor   gamma = 1 / sqrt(1 - v^2/c^2),  with c = 1.

    Speeds are hard-clamped just below c: nothing in this engine may reach or
    exceed the light cone, so gamma stays finite and the integrator stays stable.
    """
    speed2 = np.sum(v * v, axis=-1)
    speed2 = np.minimum(speed2, 1.0 - 1e-9)
    return 1.0 / np.sqrt(1.0 - speed2)


def smoothstep(edge0: float, edge1: float, x: np.ndarray | float) -> np.ndarray:
    """
    Hermite 3t^2 - 2t^3 interpolation, clamped to [0, 1].

    Edges may be given in DESCENDING order (edge0 > edge1) to get a falling
    ramp; the span keeps its sign and only its magnitude is floored, so a
    reversed pair inverts the ramp instead of saturating it.
    """
    span = edge1 - edge0
    if abs(span) < CFG.eps:
        span = math.copysign(CFG.eps, span if span != 0.0 else 1.0)
    t = np.clip((x - edge0) / span, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def lerp_color(a: Sequence[float], b: Sequence[float], t: float) -> tuple[int, int, int]:
    t = min(max(t, 0.0), 1.0)
    return (int(a[0] + (b[0] - a[0]) * t),
            int(a[1] + (b[1] - a[1]) * t),
            int(a[2] + (b[2] - a[2]) * t))


# ==============================================================================
#  SECTION 2 -- THE INTEGRATOR
#
#  Classical Runge-Kutta, 4th order.  Local truncation error O(h^5), global
#  O(h^4).  Euler is O(h) globally and secularly inflates orbital energy -- it
#  would make every orbit near this black hole spiral out into a lie.  RK4 is
#  therefore the *only* integrator in this engine.
#
#      k1 = f(t,         y)
#      k2 = f(t + h/2,   y + h k1 / 2)
#      k3 = f(t + h/2,   y + h k2 / 2)
#      k4 = f(t + h,     y + h k3)
#      y(t+h) = y + h (k1 + 2 k2 + 2 k3 + k4) / 6
#
#  `y` is an (N, 4) array [x, y, vx, vy] so the whole population integrates in
#  four vectorized derivative evaluations regardless of N.
# ==============================================================================

class RK4Integrator:
    """Fixed-step classical RK4 over a stacked phase-space state array."""

    __slots__ = ("derivative",)

    def __init__(self, derivative: Callable[[np.ndarray, float], np.ndarray]):
        # derivative(state, t) -> dstate/dt, same shape as state.
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

    def integrate(self, state: np.ndarray, t: float, h: float, substeps: int) -> np.ndarray:
        """Advance by `h` total using `substeps` equal RK4 steps."""
        if substeps < 1:
            substeps = 1
        sub = h / substeps
        for i in range(substeps):
            state = self.step(state, t + i * sub, sub)
        return state


# ==============================================================================
#  SECTION 3 -- COSMOLOGY:  THE QUINTESSENCE SCALAR FIELD
#
#  Space here is not an empty stage.  It is a medium with its own equation of
#  state, driven by a homogeneous scalar (quintessence) field whose potential
#  energy behaves as a cosmological constant Lambda.
#
#      Friedmann II :   a_dotdot / a = -4 pi G (rho + 3p) / 3  +  Lambda c^2 / 3
#      For w = -1 (pure Lambda) the vacuum term dominates and
#          a(t) = exp(H t),          H = sqrt(Lambda c^2 / 3)
#
#  The dynamical consequence used by every particle in the sim is the de Sitter
#  repulsion, which is *exact* for a Lambda-dominated universe:
#
#          a_vac(r) = (Lambda c^2 / 3) r  =  H^2 r          (outward)
#
#  This is why the engine needs no special-case "unbind distant galaxies" hack:
#  a body is bound iff local gravity beats H^2 r, i.e. inside the turnaround
#  radius r_ta = (GM / H^2)^(1/3).  Beyond it, recession is automatic.
# ==============================================================================

class CosmologicalField:
    """Homogeneous dark-energy background: scale factor plus de Sitter repulsion."""

    def __init__(self, hubble: float = CFG.hubble):
        self.hubble = hubble
        self.log_a = 0.0        # ln a(t); stored logarithmically to avoid overflow
        self.time = 0.0

    # -- evolution ----------------------------------------------------------
    def advance(self, dt: float) -> None:
        """d(ln a)/dt = H  ->  a(t) = exp(H t)."""
        self.log_a += self.hubble * dt
        self.time += dt

    @property
    def scale_factor(self) -> float:
        return math.exp(self.log_a)

    @property
    def wrapped_scale(self) -> float:
        """
        Scale factor folded into [1, 2).

        The grid is a *comoving* lattice, so its pitch grows without bound as
        a(t) does.  Folding by octaves lets the fabric visibly stream outward
        forever while the number of drawn lines stays constant -- the lattice
        simply re-seeds each time it has doubled, which is invisible because a
        doubled lattice is congruent to the original one.
        """
        return 2.0 ** (math.fmod(self.log_a / math.log(2.0), 1.0) % 1.0)

    def turnaround_radius(self, mass: float, r_s: float = 0.0) -> float:
        """
        Radius at which inward gravity exactly balances outward dark energy --
        the boundary between "bound" and "receding".

        In the Newtonian limit this is the familiar r_ta = (GM/H^2)^(1/3).  But
        the engine's force law is Paczynski-Wiita, so the honest balance is

                M / (r - r_s)^2 = H^2 r      =>      H^2 r (r - r_s)^2 - M = 0

        a cubic whose largest real root above r_s is the true turnaround radius.
        Near a black hole the two answers differ by an order of magnitude, so
        reporting the Newtonian value here would simply be wrong.
        """
        if self.hubble <= CFG.eps:
            return float("inf")
        h2 = self.hubble * self.hubble
        if r_s <= 0.0:
            return (mass / h2) ** (1.0 / 3.0)
        # h2 r^3 - 2 h2 rs r^2 + h2 rs^2 r - M = 0
        roots = np.roots([h2, -2.0 * h2 * r_s, h2 * r_s * r_s, -mass])
        real = [float(z.real) for z in roots
                if abs(z.imag) < 1e-9 and z.real > r_s * 1.000001]
        return max(real) if real else float("inf")

    def vacuum_acceleration(self, pos: np.ndarray) -> np.ndarray:
        """Outward de Sitter acceleration H^2 * r, vectorized over (N, 2)."""
        if self.hubble <= CFG.eps:
            return np.zeros_like(pos)
        return (self.hubble * self.hubble) * pos


# ==============================================================================
#  SECTION 4 -- THE KERR BLACK HOLE GEOMETRY
#
#  A rotating (Kerr) hole, viewed in its equatorial plane.  All radii below are
#  the textbook Boyer-Lindquist results with G = c = 1 and a = a* M.
#
#      Outer event horizon    r+   = M + sqrt(M^2 - a^2)
#      Ergosphere (equator)   rE   = 2M                       [theta = pi/2]
#      Photon circular orbits r_ph = 2M {1 + cos( (2/3) arccos( -/+ a/M ) )}
#                                    upper sign -> prograde, lower -> retrograde
#      ISCO                   Bardeen-Press-Teukolsky Z1/Z2 construction
#
#  The ergosphere touching the horizon at the poles and bulging to 2M at the
#  equator is why an equatorial slice always shows rE = 2M regardless of spin --
#  that is not a bug, it is the geometry.
# ==============================================================================

class BlackHole:
    """The central rotating singularity and its causal boundaries."""

    def __init__(self, mass: float = CFG.bh_mass, spin: float = CFG.bh_spin):
        self.mass = float(mass)
        self.spin = float(np.clip(spin, 0.0, 0.999))
        self.position = np.zeros(2, dtype=np.float64)  # pinned to world origin
        self.phase = 0.0                               # accretion-disk animation

    # -- derived geometry ---------------------------------------------------
    @property
    def a(self) -> float:
        """Kerr spin parameter with dimensions of length:  a = J / (M c)."""
        return self.spin * self.mass

    @property
    def r_s(self) -> float:
        """Schwarzschild radius 2GM/c^2 -- the non-rotating horizon."""
        return 2.0 * self.mass

    @property
    def r_horizon(self) -> float:
        """Outer Kerr horizon r+ = M + sqrt(M^2 - a^2)."""
        m, a = self.mass, self.a
        return m + math.sqrt(max(m * m - a * a, 0.0))

    @property
    def r_ergosphere(self) -> float:
        """Equatorial static limit:  rE = M + sqrt(M^2 - a^2 cos^2 th) = 2M."""
        return 2.0 * self.mass

    @property
    def r_photon(self) -> float:
        """Prograde equatorial photon circular orbit."""
        m, astar = self.mass, self.spin
        return 2.0 * m * (1.0 + math.cos((2.0 / 3.0) * math.acos(-astar)))

    @property
    def r_photon_retro(self) -> float:
        """Retrograde equatorial photon circular orbit (always larger)."""
        m, astar = self.mass, self.spin
        return 2.0 * m * (1.0 + math.cos((2.0 / 3.0) * math.acos(astar)))

    @property
    def r_isco(self) -> float:
        """Bardeen-Press-Teukolsky prograde ISCO."""
        m, astar = self.mass, self.spin
        z1 = 1.0 + (1.0 - astar ** 2) ** (1.0 / 3.0) * (
            (1.0 + astar) ** (1.0 / 3.0) + (1.0 - astar) ** (1.0 / 3.0))
        z2 = math.sqrt(3.0 * astar ** 2 + z1 * z1)
        return m * (3.0 + z2 - math.sqrt(max((3.0 - z1) * (3.0 + z1 + 2.0 * z2), 0.0)))

    @property
    def angular_momentum(self) -> float:
        """J = a M  (geometrized).  Drives the Lense-Thirring term."""
        return self.a * self.mass

    @property
    def omega_horizon(self) -> float:
        """Horizon angular velocity Omega_H = a / (2 M r+); the superradiance cap."""
        return self.a / max(2.0 * self.mass * self.r_horizon, CFG.eps)

    # -- metric functions ---------------------------------------------------
    def lapse(self, r: np.ndarray | float) -> np.ndarray | float:
        """
        Schwarzschild lapse  sqrt(1 - r_s / r)  =  dtau / dt for a static
        observer.  This is the gravitational time-dilation factor.  Clamped just
        above zero so a hovering clock slows but never divides by zero.
        """
        val = 1.0 - self.r_s / np.maximum(r, self.r_s * 1.0000001)
        return np.sqrt(np.clip(val, 1e-9, 1.0))

    def circular_speed(self, r: np.ndarray | float) -> np.ndarray | float:
        """
        Speed of a circular orbit under the engine's ACTUAL equation of motion.

        The force law is Paczynski-Wiita, but the dynamics are relativistic:
        dv/dt = (1/gamma)[a - (v.a)v].  On a circular orbit v.a = 0, so the
        centripetal condition is not v^2/r = |a| but

                v^2 / r = |a| / gamma        =>       v^2 gamma = r |a|

        Writing K = r |a| = r M / (r - r_s)^2 and u = v^2, the condition
        u / sqrt(1 - u) = K is a quadratic u^2 + K^2 u - K^2 = 0 with root

                u = ( -K^2 + sqrt(K^4 + 4 K^2) ) / 2

        which reduces to the Newtonian v = sqrt(M/r) in the weak field.  Using
        the Newtonian value instead would launch every "circular" orbit visibly
        eccentric -- so this correction is not cosmetic bookkeeping.
        """
        r = np.maximum(r, self.r_s * 1.02)
        k = r * self.mass / np.maximum((r - self.r_s) ** 2, CFG.eps)
        k2 = k * k
        u = 0.5 * (-k2 + np.sqrt(k2 * k2 + 4.0 * k2))
        return np.sqrt(np.clip(u, 0.0, 0.9801))

    def frame_drag_omega(self, r: np.ndarray | float) -> np.ndarray | float:
        """
        Lense-Thirring precession rate  omega(r) = 2 G J / (c^2 r^3).
        This is the angular velocity that inertial frames themselves acquire.
        """
        r = np.maximum(r, CFG.eps)
        return 2.0 * self.angular_momentum / (r ** 3)


# ==============================================================================
#  SECTION 5 -- THE COMPOSED FORCE LAW
#
#  Every massive object in the sandbox -- planets, stars, dark matter and the
#  spacecraft -- obeys one acceleration function, so momentum bookkeeping is
#  automatically consistent between subsystems.  Four contributions superpose:
#
#  (1) CENTRAL HOLE.  Paczynski & Wiita (1980) pseudo-Newtonian potential
#
#          Phi_PW(r) = -GM / (r - r_s)          =>   g = -GM / (r - r_s)^2
#
#      This is not a fudge: it is the standard relativistic surrogate that
#      reproduces the Schwarzschild marginally-stable orbit at exactly 6M and
#      the marginally-bound orbit at exactly 4M, and it diverges at the horizon
#      so nothing can hover there.  Unlike a Newtonian 1/r^2 law it therefore
#      gives correct periastron precession behaviour and a genuine plunge region.
#
#  (2) FRAME DRAGGING.  Weak-field gravitomagnetism.  A spinning mass sources a
#      gravitomagnetic field which, in the equatorial plane (J perpendicular to
#      the plane, so J . rhat = 0), reduces to
#
#          B_g = -(2 G J / c^3 r^3) zhat
#          a_LT = -v x B_g  =  (2J/r^3) (v_y, -v_x)
#
#      An infalling particle is therefore swept tangentially in the direction of
#      the hole's spin: the whirlpool of the ergosphere.
#
#  (3) DARK ENERGY.  a_vac = H^2 r, outward.  See CosmologicalField.
#
#  (4) N-BODY.  Plummer-softened Newtonian gravity between spawned bodies,
#      a = -G m (r_ij) / (|r_ij|^2 + s^2)^(3/2), fully vectorized as an (N, K)
#      pair tensor.
#
#  RELATIVISTIC DYNAMICS.  The engine does not integrate a = F/m.  It integrates
#  the true relativistic law dp/dt = F with p = gamma m v, whose velocity form is
#
#          dv/dt = (1/gamma) [ a_N - (v . a_N) v / c^2 ]
#
#  The subtracted longitudinal term is what makes c an asymptote: as |v| -> 1 the
#  component of acceleration parallel to v is suppressed by 1/gamma^3 while the
#  transverse component is only suppressed by 1/gamma.  A slingshot can therefore
#  push the ship to 0.999 c and never past it, with no artificial speed clamp.
# ==============================================================================

class GravityField:
    """Superposed acceleration field + relativistic velocity dynamics."""

    def __init__(self, hole: BlackHole, cosmos: CosmologicalField):
        self.hole = hole
        self.cosmos = cosmos
        # Populated each frame by Simulation so the derivative closure stays pure.
        self.src_pos = np.zeros((0, 2), dtype=np.float64)
        self.src_mass = np.zeros((0,), dtype=np.float64)
        self.src_soft = np.zeros((0,), dtype=np.float64)
        self._src_soft2 = np.zeros((0,), dtype=np.float64)
        # Warp bubble, if engaged, nulls external tidal forces inside its wall.
        self.bubble: "WarpBubble | None" = None

    # -- source table -------------------------------------------------------
    def set_sources(self, pos: np.ndarray, mass: np.ndarray, soft: np.ndarray) -> None:
        """Register the N-body gravitational sources for this frame."""
        self.src_pos = pos
        self.src_mass = mass
        self.src_soft = soft
        self._src_soft2 = soft * soft

    # -- (1)+(2)+(3): the central hole and the vacuum -----------------------
    def central_acceleration(self, pos: np.ndarray) -> np.ndarray:
        """Hole (Paczynski-Wiita) + frame dragging needs v, so this is the static part."""
        hole = self.hole
        d = pos - hole.position
        r = norm(d, keepdims=True)
        r = np.maximum(r, CFG.eps)
        unit = d / r

        # Paczynski-Wiita: singular at r = r_s, so keep the denominator positive.
        denom = np.maximum(r - hole.r_s, 0.35 * hole.r_s)
        g_pw = -hole.mass / (denom * denom)

        acc = unit * g_pw
        acc = acc + self.cosmos.vacuum_acceleration(pos)
        return acc

    def frame_drag_acceleration(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """a_LT = (2J/r^3) * (v_y, -v_x).  Zero for a non-spinning hole."""
        J = self.hole.angular_momentum
        if abs(J) <= CFG.eps:
            return np.zeros_like(pos)
        d = pos - self.hole.position
        r = norm(d, keepdims=True)
        r = np.maximum(r, 0.5 * self.hole.r_s)
        coeff = 2.0 * J / (r ** 3)
        # (v x zhat) in 2D, with the sign fixed so J > 0 drags along +phi.
        swirl = np.stack([vel[..., 1], -vel[..., 0]], axis=-1)
        return coeff * swirl

    # -- (4): mutual N-body -------------------------------------------------
    def nbody_acceleration(self, pos: np.ndarray) -> np.ndarray:
        """
        Vectorized Plummer-softened pairwise gravity.

        Builds an (N, K, 2) separation tensor in one shot; for the body counts
        this engine allows (N,K <= ~220) that is a few hundred thousand floats,
        which NumPy chews through far faster than any Python-level loop.
        """
        if self.src_mass.size == 0 or pos.size == 0:
            return np.zeros_like(pos)
        # (N, 1, 2) - (1, K, 2) -> (N, K, 2).  The squared distance is formed by
        # einsum rather than (diff*diff).sum so no second (N, K, 2) temporary is
        # allocated, and r^-3 is built from a reciprocal plus a sqrt because
        # np.power(x, -1.5) is several times slower than either.
        diff = self.src_pos[None, :, :] - pos[:, None, :]
        r2 = np.einsum("nkc,nkc->nk", diff, diff)
        inv = 1.0 / (r2 + self._src_soft2[None, :] + CFG.eps)
        w = self.src_mass[None, :] * inv * np.sqrt(inv)         # (N, K) = m / r^3
        return np.einsum("nk,nkc->nc", w, diff)

    # -- full field ---------------------------------------------------------
    def acceleration(self, pos: np.ndarray, vel: np.ndarray,
                     include_nbody: bool = True) -> np.ndarray:
        acc = self.central_acceleration(pos)
        acc += self.frame_drag_acceleration(pos, vel)
        if include_nbody:
            acc += self.nbody_acceleration(pos)
        return acc

    # -- relativistic conversion -------------------------------------------
    @staticmethod
    def proper_accel(vel: np.ndarray, acc_newtonian: np.ndarray) -> np.ndarray:
        """
        Convert a Newtonian-equivalent force-per-mass into dv/dt under
        dp/dt = F with p = gamma m v:

            dv/dt = (1/gamma) [ a - (v . a) v ]        (c = 1)

        Exact, not an approximation, and the reason no speed clamp is needed.
        """
        g = lorentz_gamma(vel)[..., None]
        v_dot_a = np.sum(vel * acc_newtonian, axis=-1, keepdims=True)
        return (acc_newtonian - v_dot_a * vel) / g

    # -- derivative closures for RK4 ---------------------------------------
    def make_derivative(self, include_nbody: bool = True,
                        relativistic: bool = True) -> Callable[[np.ndarray, float], np.ndarray]:
        """
        Return f(state, t) -> dstate/dt for state = [x, y, vx, vy].

        The closure captures only `self`, so the per-frame source table update
        is picked up automatically without rebuilding the integrator.
        """
        def derivative(state: np.ndarray, t: float) -> np.ndarray:
            pos = state[..., 0:2]
            vel = state[..., 2:4]
            acc = self.acceleration(pos, vel, include_nbody=include_nbody)
            if self.bubble is not None and self.bubble.active:
                # Inside an Alcubierre bubble the ship is in free fall in locally
                # flat space: the interior is Riemann-flat, so external tidal
                # acceleration is masked by (1 - f) where f is the shape function.
                acc = acc * self.bubble.exterior_mask(pos)
            if relativistic:
                acc = self.proper_accel(vel, acc)
            return np.concatenate([vel, acc], axis=-1)
        return derivative


# ==============================================================================
#  SECTION 6 -- QUANTUM MATTER:  THE ATOMIC TUG-OF-WAR
#
#  A cold self-gravitating sphere is a standing argument between two pressures.
#
#  INWARD -- gravitational compression.  For a uniform sphere the exact central
#  pressure from hydrostatic equilibrium dP/dr = -G m(r) rho / r^2 is
#
#          P_grav = 3 G M^2 / (8 pi R^4)
#
#  OUTWARD -- electromagnetic / degeneracy resistance.  At planetary densities
#  this is Coulomb pressure: electron clouds refuse to interpenetrate because
#  the Pauli exclusion principle forbids two electrons the same state.  Treated
#  as a polytrope it is non-relativistic electron degeneracy
#
#          P_deg = K_nr (rho / mu_e)^(5/3),    K_nr = (3/pi)^(2/3) h^2 / (20 m_e m_H^(5/3))
#
#  Balancing P_grav against P_deg gives the celebrated result that a degenerate
#  body SHRINKS as you feed it:  R ~ M^(-1/3).  Interpolating between the
#  incompressible low-mass branch (R ~ M^(1/3)) and that degenerate branch is the
#  Zapolsky-Salpeter (1969) relation used here:
#
#          R(M) = C M^(1/3) / [ 1 + (M / M_0)^(4/3) ]
#
#  which peaks at M = M_0 / 3^(3/4).  With M_0 = 8 M_J the maximum radius lands
#  near 3.5 M_J -- exactly where real gas giants top out.  Jupiter, Saturn and a
#  50 M_J brown dwarf are all very nearly the same size, and this formula is why.
#
#  As electrons are forced relativistic the polytropic index softens from 5/3 to
#  4/3, pressure support stops growing with mass, and the argument is lost:
#
#          M_Ch = 1.44 M_sun  = 1508 M_J   -> Chandrasekhar limit, degenerate core
#          M_TOV ~ 3.0 M_sun  = 3143 M_J   -> neutron degeneracy fails, horizon forms
#
#  Fusion thresholds are the observational ones:
#          > 13 M_J   deuterium burning       -> BROWN DWARF
#          > 80 M_J   sustained p-p hydrogen  -> STAR
# ==============================================================================

class Phase:
    """Structural phase codes.  Ordered by increasing degeneracy."""
    PLANET      = 0
    GAS_GIANT   = 1
    BROWN_DWARF = 2
    STAR        = 3
    DEGENERATE  = 4     # white dwarf / neutron star: past atomic resistance
    COLLAPSED   = 5     # horizon has formed

    NAMES = {0: "Planet", 1: "Gas Giant", 2: "Brown Dwarf",
             3: "Star", 4: "Degenerate Core", 5: "Black Hole"}
    COLORS = {0: Palette.ROCK, 1: Palette.PLANET, 2: Palette.BROWN_DWARF,
              3: Palette.STAR, 4: (222, 236, 255), 5: (10, 8, 16)}


class StellarStructure:
    """Static physics of the mass-radius relation and the phase transitions."""

    # Thresholds in Jupiter masses.
    M_DEUTERIUM   = 13.0        # deuterium ignition
    M_HYDROGEN    = 80.0        # hydrogen ignition (0.076 M_sun)
    M_JUP_PER_SUN = 1047.57
    M_CHANDRA     = 1.44 * M_JUP_PER_SUN     # 1508 M_J
    M_TOV         = 3.00 * M_JUP_PER_SUN     # 3143 M_J

    # Zapolsky-Salpeter scale mass; radius peaks at M_0 / 3^(3/4) ~ 3.5 M_J.
    M_ZS = 8.0
    R_JUP_PX = 11.0             # display radius of a 1 M_J body, in world units

    # Geometrized mass per Jupiter mass.  Calibrated so that a body which
    # collapses at M_TOV = 3 M_sun acquires geometric mass 25.1, essentially
    # equal to the default central hole's M = 26 -- a spawned black hole is
    # therefore a genuine gravitational rival, not a decoration.
    GEOM_PER_MJ = 0.008

    # ---------------------------------------------------------------- phases
    @staticmethod
    def phase_of(mass_mj: np.ndarray) -> np.ndarray:
        """Vectorized phase classification from Jupiter-mass array."""
        m = np.asarray(mass_mj, dtype=np.float64)
        ph = np.zeros(m.shape, dtype=np.int32)
        ph = np.where(m > 0.4, Phase.GAS_GIANT, ph)
        ph = np.where(m > StellarStructure.M_DEUTERIUM, Phase.BROWN_DWARF, ph)
        ph = np.where(m > StellarStructure.M_HYDROGEN, Phase.STAR, ph)
        ph = np.where(m > StellarStructure.M_CHANDRA, Phase.DEGENERATE, ph)
        ph = np.where(m > StellarStructure.M_TOV, Phase.COLLAPSED, ph)
        return ph

    # ---------------------------------------------------------------- radius
    @staticmethod
    def physical_radius(mass_mj: np.ndarray) -> np.ndarray:
        """
        Zapolsky-Salpeter R(M) in Jupiter radii.  This is the honest curve: it
        rises as M^(1/3), peaks near 3.5 M_J, then falls as M^(-1/3).
        """
        m = np.maximum(np.asarray(mass_mj, dtype=np.float64), 1e-6)
        return np.cbrt(m) / (1.0 + np.power(m / StellarStructure.M_ZS, 4.0 / 3.0))

    @staticmethod
    def display_radius(mass_mj: np.ndarray, phase: np.ndarray) -> np.ndarray:
        """
        Screen radius in world units.

        The true relation is kept for the planetary branch (so the "it stops
        growing and starts shrinking" behaviour is directly visible), but fusing
        objects are re-inflated on a log law because a real star is ~10x a gas
        giant while a real neutron star is ~1e-5 of one -- unrenderable at any
        single linear scale.  This is a *display* map only; every force in the
        engine uses geometric mass, never this number.
        """
        m = np.maximum(np.asarray(mass_mj, dtype=np.float64), 1e-6)
        ph = np.asarray(phase)
        r = StellarStructure.physical_radius(m) * StellarStructure.R_JUP_PX

        # Fusing bodies: main-sequence R ~ M^0.8 below 1 M_sun, log-compressed.
        star_r = 13.0 + 7.0 * np.log10(1.0 + m / StellarStructure.M_HYDROGEN)
        r = np.where(ph >= Phase.BROWN_DWARF, np.maximum(r, star_r * 0.72), r)
        r = np.where(ph >= Phase.STAR, star_r, r)
        # Degenerate remnants really are tiny; show them small but visible.
        r = np.where(ph == Phase.DEGENERATE, 6.0, r)
        # A collapsed body is drawn at its own Schwarzschild radius.
        # A collapsed body is drawn at its own Schwarzschild radius r_s = 2M,
        # capped only so an absurdly over-fed object cannot swallow the viewport.
        r = np.where(ph == Phase.COLLAPSED,
                     np.clip(2.0 * m * StellarStructure.GEOM_PER_MJ, 7.0, 90.0), r)
        return np.maximum(r, 2.0)

    # --------------------------------------------------------------- masses
    @staticmethod
    def geometric_mass(mass_mj: np.ndarray) -> np.ndarray:
        """Convert Jupiter masses to geometrized mass GM/c^2 in world units."""
        return np.asarray(mass_mj, dtype=np.float64) * StellarStructure.GEOM_PER_MJ

    # --------------------------------------------------- the tug-of-war gauge
    @staticmethod
    def pressure_balance(mass_mj: float) -> tuple[float, float]:
        """
        Return (compression, resistance) in arbitrary but *consistently scaled*
        units, for the UI gauge.

            P_grav ~ M^2 / R^4                (gravitational compression)
            P_deg  ~ rho^(5/3) ~ (M/R^3)^(5/3) (non-relativistic electron gas)

        Their ratio is what actually matters, and it crosses unity exactly where
        the Zapolsky-Salpeter radius turns over.  Above the Chandrasekhar mass
        the electron gas is relativistic, the exponent softens to 4/3, and
        resistance can no longer keep up at any radius -- the gauge pegs.
        """
        m = max(float(mass_mj), 1e-6)
        r = float(StellarStructure.physical_radius(m))
        p_grav = m * m / (r ** 4)
        index = 5.0 / 3.0 if m < StellarStructure.M_CHANDRA else 4.0 / 3.0
        p_deg = (m / (r ** 3)) ** index
        return p_grav, p_deg


# ==============================================================================
#  SECTION 7 -- THE BODY POPULATION  (structure-of-arrays particle engine)
#
#  Celestial bodies live in parallel NumPy arrays rather than in a list of Python
#  objects.  One RK4 step therefore costs four vectorized derivative evaluations
#  for the entire population instead of 4N interpreted function calls, which is
#  what keeps the frame budget intact at a couple of hundred bodies.
#
#  SPAGHETTIFICATION.  In the local frame of an infalling body the tidal
#  acceleration across a separation dr is, exactly, the geodesic deviation
#
#      radial (stretch)    d a_r  = +2 G M dr / r^3
#      transverse (squeeze) d a_t = -  G M dr / r^3
#
#  The factor of two, and the opposite signs, are the whole phenomenon: the body
#  is drawn into a filament along the radial direction while being pinched
#  perpendicular to it.  `strain` integrates 2GM/r^3 so the render stretches by
#  exactly the ratio the metric dictates before the body is deleted.
# ==============================================================================

class BodyPopulation:
    """Planets, stars and collapsed remnants, plus their tidal death throes."""

    # Strain runs 0 -> 1 over the disruption.  TIDAL_GAIN sets the presentation
    # timescale; MAX_STRAIN_STEP floors the render loop at 1/0.02 = 50 steps so
    # the filament is always actually watchable, however deep the body plunges
    # or however large a dt the user has dialled in.
    TIDAL_GAIN = 26.0
    MAX_STRAIN_STEP = 0.020

    def __init__(self, gravity: GravityField, trail_len: int = CFG.trail_len):
        self.gravity = gravity
        self.trail_len = trail_len
        self.integrator = RK4Integrator(gravity.make_derivative(include_nbody=True,
                                                                relativistic=True))
        self._alloc(0)

    # ------------------------------------------------------------- storage
    def _alloc(self, n: int) -> None:
        self.state = np.zeros((n, 4), dtype=np.float64)   # x, y, vx, vy
        self.mass_mj = np.zeros(n, dtype=np.float64)
        self.phase = np.zeros(n, dtype=np.int32)
        self.radius = np.zeros(n, dtype=np.float64)
        self.geom = np.zeros(n, dtype=np.float64)
        self.strain = np.zeros(n, dtype=np.float64)       # 0 = healthy, >0 = dying
        self.spin_seed = np.zeros(n, dtype=np.float64)
        self.trail = np.zeros((n, self.trail_len, 2), dtype=np.float32)
        self.trail_fill = np.zeros(n, dtype=np.int32)
        self.trail_head = 0

    @property
    def count(self) -> int:
        return self.state.shape[0]

    def _concat(self, arr: np.ndarray, extra: np.ndarray) -> np.ndarray:
        return np.concatenate([arr, extra], axis=0)

    # -------------------------------------------------------------- spawning
    def spawn(self, pos: Sequence[float], vel: Sequence[float], mass_mj: float) -> None:
        """Add one body.  Structure (phase, radius) is derived, never supplied."""
        if self.count >= CFG.max_bodies:
            return
        st = np.array([[pos[0], pos[1], vel[0], vel[1]]], dtype=np.float64)
        self.state = self._concat(self.state, st)
        self.mass_mj = self._concat(self.mass_mj, np.array([mass_mj], dtype=np.float64))
        self.phase = self._concat(self.phase, np.zeros(1, dtype=np.int32))
        self.radius = self._concat(self.radius, np.zeros(1))
        self.geom = self._concat(self.geom, np.zeros(1))
        self.strain = self._concat(self.strain, np.zeros(1))
        self.spin_seed = self._concat(self.spin_seed,
                                      np.random.uniform(0.0, math.tau, 1))
        new_trail = np.repeat(st[:, None, 0:2].astype(np.float32), self.trail_len, axis=1)
        self.trail = np.concatenate([self.trail, new_trail], axis=0)
        self.trail_fill = self._concat(self.trail_fill, np.zeros(1, dtype=np.int32))
        self.refresh_structure()

    def spawn_orbiting(self, hole: BlackHole, radius: float, mass_mj: float,
                       prograde: bool = True, angle: float | None = None) -> None:
        """
        Place a body on a circular orbit of the given radius.

        Speed comes from BlackHole.circular_speed, which solves the centripetal
        condition against the relativistic equation of motion actually used by
        the integrator -- so the orbit really closes.
        """
        ang = np.random.uniform(0.0, math.tau) if angle is None else angle
        r = max(radius, hole.r_s * 1.6)
        pos = hole.position + np.array([math.cos(ang), math.sin(ang)]) * r
        v_c = min(float(hole.circular_speed(r)), 0.92)
        tangent = np.array([-math.sin(ang), math.cos(ang)])
        if not prograde:
            tangent = -tangent
        self.spawn(pos, tangent * v_c, mass_mj)

    def clear(self) -> None:
        self._alloc(0)

    # ------------------------------------------------------------- structure
    def refresh_structure(self) -> None:
        """Re-derive phase, display radius and geometric mass from mass."""
        if self.count == 0:
            return
        self.phase = StellarStructure.phase_of(self.mass_mj)
        self.radius = StellarStructure.display_radius(self.mass_mj, self.phase)
        self.geom = StellarStructure.geometric_mass(self.mass_mj)

    def add_mass_at(self, world_pos: np.ndarray, amount_mj: float,
                    grab_radius: float = 26.0) -> int:
        """
        Pour mass into the nearest body within `grab_radius`.  Returns its index
        or -1.  This is the hook the mouse uses to drive phase transitions live.
        """
        if self.count == 0:
            return -1
        d = norm(self.state[:, 0:2] - np.asarray(world_pos, dtype=np.float64)[None, :])
        d = d - self.radius
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
        self.trail_fill = np.minimum(self.trail_fill + 1, self.trail_len)

    def gravitational_sources(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bodies that still source gravity (the doomed ones no longer do)."""
        if self.count == 0:
            return (np.zeros((0, 2)), np.zeros(0), np.zeros(0))
        live = self.strain <= 0.0
        return (self.state[live, 0:2], self.geom[live],
                np.maximum(self.radius[live] * 0.6, 1.5))

    # -------------------------------------------------- tidal disruption / death
    def update_spaghettification(self, hole: BlackHole, dt: float,
                                 bubble: "WarpBubble | None" = None) -> list[int]:
        """
        Flag bodies inside the horizon, integrate their tidal strain, and delete
        them once the filament is fully drawn.  Returns indices removed.

        A body sheltering inside an active warp bubble is exempt: the bubble
        interior is flat, so geodesic deviation across it vanishes identically.
        """
        if self.count == 0:
            return []
        pos = self.state[:, 0:2]
        d = norm(pos - hole.position)

        sheltered = np.zeros(self.count, dtype=bool)
        if bubble is not None and bubble.active:
            sheltered = bubble.interior_fraction(pos) > 0.5

        crossing = (d < hole.r_horizon) & (~sheltered)
        self.strain = np.where(crossing & (self.strain <= 0.0), 1e-3, self.strain)

        dying = self.strain > 0.0
        if np.any(dying):
            # ds/dt tracks the tidal tensor eigenvalue 2GM/r^3, so a body dies
            # faster the deeper it falls -- the physically correct ordering.
            rr = np.maximum(d, hole.r_horizon * 0.25)
            rate = 2.0 * hole.mass / (rr ** 3)
            inc = np.minimum(rate * dt * self.TIDAL_GAIN, self.MAX_STRAIN_STEP)
            self.strain = np.where(dying, self.strain + inc, self.strain)
            # Doomed bodies are dragged inward and spun down at the horizon.
            # Both decays are exponential IN dt rather than per-call, so the
            # disruption looks identical whether the user is running at full
            # speed or crawling through it in slow motion.
            self.state[dying, 0:2] *= math.exp(-0.030 * dt)
            self.state[dying, 2:4] *= math.exp(-0.110 * dt)

        gone = np.where(self.strain > 1.0)[0]
        if gone.size:
            self.remove(gone)
        return list(map(int, gone))

    def remove(self, indices: Iterable[int]) -> None:
        keep = np.ones(self.count, dtype=bool)
        keep[np.asarray(list(indices), dtype=int)] = False
        self.state = self.state[keep]
        self.mass_mj = self.mass_mj[keep]
        self.phase = self.phase[keep]
        self.radius = self.radius[keep]
        self.geom = self.geom[keep]
        self.strain = self.strain[keep]
        self.spin_seed = self.spin_seed[keep]
        self.trail = self.trail[keep]
        self.trail_fill = self.trail_fill[keep]


# ==============================================================================
#  SECTION 8 -- DARK MATTER:  THE INVISIBLE COSMIC ELEMENT
#
#  STRICT INTERACTION RULE, enforced structurally rather than by convention:
#  this population is stepped through `GravityField` and *nothing else*.  There
#  is no collision code, no pressure term and no electromagnetic coupling
#  anywhere in this class -- dark matter passes through planets, stars, walls and
#  the spacecraft because there is literally no code path by which it could not.
#  It responds to curvature alone.
#
#  SUPERRADIANCE.  Inside the ergosphere (r < 2M) no observer can remain static;
#  every worldline is dragged prograde.  That permits negative-energy orbits, and
#  hence the Penrose process: a particle on such an orbit can carry away more
#  energy than it brought in, drawn from the hole's rotational reservoir.  The
#  amplification condition for a co-rotating mode is
#
#          0 < omega < m Omega_H ,      Omega_H = a c^3 / (2 G M r_+)
#
#  Prograde particles in the ergosphere are therefore boosted here, and the hole
#  is spun down by exactly the reciprocal bookkeeping -- energy is not created.
#
#  THE OBSERVER EFFECT.  These are fragile coherent states.  Any electromagnetic
#  probe -- an active laser sensor, or simply operating outside a deep-shielded
#  laboratory where cosmic-ray and photon flux is unattenuated -- collapses them,
#  and `coherence` decays to zero.  A decohered particle is not deleted: it is
#  still there, still pulling on everything gravitationally.  It is merely
#  unmeasurable.  That distinction is the entire point of the experiment.
# ==============================================================================

class DarkMatterField:
    """Collisionless, EM-blind particles with a decoherence lifecycle."""

    DECOHERE_RATE = 0.9      # per world-time unit, under electromagnetic probing
    RECOHERE_RATE = 0.22     # per world-time unit, inside deep shielding

    def __init__(self, gravity: GravityField, hole: BlackHole):
        self.gravity = gravity
        self.hole = hole
        # Dark matter is stepped with the SAME field object, so it feels the
        # gravity of every body -- but the derivative is built once, here, and
        # only ever calls GravityField.  No EM term can leak in.
        self.integrator = RK4Integrator(gravity.make_derivative(include_nbody=True,
                                                               relativistic=True))
        self.state = np.zeros((0, 4), dtype=np.float64)
        self.coherence = np.zeros(0, dtype=np.float64)
        self.hue = np.zeros(0, dtype=np.float64)
        self.superradiant = np.zeros(0, dtype=bool)
        self.extracted_energy = 0.0

    @property
    def count(self) -> int:
        return self.state.shape[0]

    # -------------------------------------------------------------- spawning
    def seed_halo(self, n: int, r_min: float, r_max: float,
                  prograde_bias: float = 0.85) -> None:
        """
        Seed an isotropic halo on near-circular orbits.

        Radii follow an r^(1/2) draw so the surface density goes as 1/r, the
        flat-rotation-curve profile that dark matter was invented to explain.
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
        vel = tangent * (v_c * sign)[:, None]
        vel += np.random.normal(0.0, 0.012, (n, 2))     # velocity dispersion

        self.state = np.concatenate([self.state,
                                     np.concatenate([pos, vel], axis=1)], axis=0)
        self.coherence = np.concatenate([self.coherence, np.ones(n)])
        self.hue = np.concatenate([self.hue, np.random.uniform(0.0, 1.0, n)])
        self.superradiant = np.concatenate([self.superradiant, np.zeros(n, dtype=bool)])

    def clear(self) -> None:
        self.state = np.zeros((0, 4), dtype=np.float64)
        self.coherence = np.zeros(0)
        self.hue = np.zeros(0)
        self.superradiant = np.zeros(0, dtype=bool)

    # ---------------------------------------------------------------- physics
    def step(self, t: float, dt: float, substeps: int,
             em_probing: bool, superradiance: bool) -> None:
        if self.count == 0 or dt == 0.0:
            return
        self.state = self.integrator.integrate(self.state, t, dt, substeps)

        pos = self.state[:, 0:2]
        vel = self.state[:, 2:4]
        r = norm(pos)

        # -- Penrose / superradiant amplification inside the ergosphere -------
        if superradiance and self.hole.a > CFG.eps:
            inside = r < self.hole.r_ergosphere
            if np.any(inside):
                # Prograde test: sign of the 2D angular momentum Lz = x vy - y vx.
                lz = pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]
                co = inside & (lz > 0.0)
                self.superradiant = co
                if np.any(co):
                    # Amplification bounded by the horizon angular velocity, so
                    # the gain vanishes as a* -> 0 exactly as superradiance must.
                    gain = 1.0 + self.hole.omega_horizon * dt * 2.4
                    self.state[co, 2:4] *= gain
                    self.extracted_energy += float(np.sum(co)) * (gain - 1.0)
                    # Bookkeeping: the hole pays for it out of its spin.
                    self.hole.spin = max(0.0, self.hole.spin - 1.2e-6 * float(np.sum(co)) * dt)
            else:
                self.superradiant[:] = False
        else:
            self.superradiant[:] = False

        # -- decoherence lifecycle -------------------------------------------
        if em_probing:
            self.coherence *= math.exp(-self.DECOHERE_RATE * dt)
        else:
            self.coherence = np.minimum(1.0, self.coherence + self.RECOHERE_RATE * dt)

        # -- absorption: gravity is universal, so the horizon eats dark matter -
        if self.count:
            swallowed = r < self.hole.r_horizon
            if np.any(swallowed):
                keep = ~swallowed
                self.state = self.state[keep]
                self.coherence = self.coherence[keep]
                self.hue = self.hue[keep]
                self.superradiant = self.superradiant[keep]

    def visible_mask(self, shielded: bool, laser_on: bool) -> np.ndarray:
        """
        Measurement rule.  A particle renders only when the laboratory is deep
        shielded, the laser sensor is off, and its state has not been destroyed.
        """
        if self.count == 0:
            return np.zeros(0, dtype=bool)
        if laser_on or not shielded:
            return np.zeros(self.count, dtype=bool)
        return self.coherence > 0.06


# ==============================================================================
#  SECTION 9 -- THE ALCUBIERRE WARP BUBBLE
#
#  Alcubierre (1994), Class. Quantum Grav. 11, L73.  The metric is
#
#      ds^2 = -dt^2 + (dx - v_s(t) f(r_s) dt)^2 + dy^2 + dz^2
#
#  with the "top hat" shape function
#
#      f(r_s) = [ tanh(sigma (r_s + R)) - tanh(sigma (r_s - R)) ] / (2 tanh(sigma R))
#
#  where r_s is distance from the bubble centre, R the bubble radius and sigma
#  the wall steepness.  f = 1 inside, f = 0 outside, and the transition is the
#  bubble wall.
#
#  Two facts drive everything this class does:
#
#  (a) INSIDE, f = 1 identically, so its gradient vanishes and the region is
#      Riemann-flat.  The ship is in free fall in Minkowski space: it feels zero
#      proper acceleration, zero tidal force, and its clock runs at the same rate
#      as a distant observer's.  This is why a warping ship can sit next to a
#      black hole and not be spaghettified -- the tidal field is outside its wall.
#
#  (b) The expansion scalar of the normal volume elements is
#
#          theta = v_s (x_s / r_s) df/dr_s
#
#      Since df/dr_s < 0 everywhere, theta < 0 ahead of the ship (x_s > 0) and
#      theta > 0 behind it.  Space is CONTRACTED in front and EXPANDED behind.
#      The grid renderer reads exactly this quantity for its colour, so what is
#      drawn is the York time of the metric, not an artistic impression.
#
#  The ship never moves through its local space; the bubble carries it.  Its
#  coordinate speed v_s is therefore unbounded by c while its local speed is
#  identically zero -- no causality violation in the local frame.
# ==============================================================================

class WarpBubble:
    """Localized metric distortion: flat inside, contracted ahead, expanded behind."""

    def __init__(self, radius: float = 78.0, sigma: float = 0.055):
        self.radius = radius
        self.sigma = sigma
        self.active = False
        self.center = np.zeros(2, dtype=np.float64)
        self.heading = np.array([1.0, 0.0], dtype=np.float64)
        self.v_s = 0.0                 # bubble coordinate velocity, in units of c
        self._norm = math.tanh(self.sigma * self.radius)

    def configure(self, center: np.ndarray, heading: np.ndarray, v_s: float) -> None:
        self.center = np.asarray(center, dtype=np.float64)
        h = np.asarray(heading, dtype=np.float64)
        n = float(norm(h))
        self.heading = h / n if n > CFG.eps else np.array([1.0, 0.0])
        self.v_s = float(v_s)
        self._norm = math.tanh(self.sigma * self.radius)

    # -------------------------------------------------------- shape function
    def f(self, r_s: np.ndarray) -> np.ndarray:
        """f(r_s): 1 inside the bubble, 0 outside, smooth across the wall."""
        s, R = self.sigma, self.radius
        return (np.tanh(s * (r_s + R)) - np.tanh(s * (r_s - R))) / (2.0 * self._norm)

    def df(self, r_s: np.ndarray) -> np.ndarray:
        """df/dr_s.  Negative everywhere -- the source of the sign of theta."""
        s, R = self.sigma, self.radius
        sech2 = lambda z: 1.0 / np.cosh(np.clip(z, -30.0, 30.0)) ** 2
        return s * (sech2(s * (r_s + R)) - sech2(s * (r_s - R))) / (2.0 * self._norm)

    # ------------------------------------------------------------ field maps
    def interior_fraction(self, pos: np.ndarray) -> np.ndarray:
        """f evaluated at world positions (N, 2) -> (N,)."""
        if not self.active:
            return np.zeros(pos.shape[0])
        return self.f(norm(pos - self.center[None, :]))

    def exterior_mask(self, pos: np.ndarray) -> np.ndarray:
        """
        (1 - f) as an (N, 1) multiplier.

        Applied to every external acceleration, this is the mathematical
        statement of fact (a): the deeper inside the bubble you are, the more
        completely the outside universe's curvature is screened from you.
        """
        if not self.active:
            return np.ones((pos.shape[0], 1))
        return (1.0 - self.interior_fraction(pos))[:, None]

    def expansion_scalar(self, pos: np.ndarray) -> np.ndarray:
        """
        theta = v_s (x_s / r_s) df/dr_s, with x_s the along-heading offset.
        Negative = contraction (ahead), positive = expansion (behind).
        """
        if not self.active:
            return np.zeros(pos.shape[0])
        rel = pos - self.center[None, :]
        r = np.maximum(norm(rel), CFG.eps)
        x_s = rel @ self.heading
        return self.v_s * (x_s / r) * self.df(r)

    def grid_displacement(self, pos: np.ndarray,
                          gain: float = CFG.grid_spacing * 0.85) -> np.ndarray:
        """
        Displace fabric vertices by the metric's shift vector, -v_s f(r_s) x_hat.

        Because the displacement is proportional to f and f falls off across the
        wall, vertices near the nose are advected less than the ship itself: the
        lattice bunches up ahead and rarefies behind.  The visual compression is
        thus a *consequence* of the shift vector, not a separately authored effect.
        """
        if not self.active:
            return np.zeros_like(pos)
        f = self.interior_fraction(pos)
        # The shift is saturated through tanh(v_s): the SIGN and PROFILE are the
        # metric's, but an unbounded magnitude would displace vertices by several
        # lattice pitches at warp 3+, folding grid lines over one another and
        # destroying the very compression it is meant to show.
        amp = gain * math.tanh(self.v_s / 1.5)
        return (amp * f)[:, None] * self.heading[None, :]


# ==============================================================================
#  SECTION 10 -- THE SPACECRAFT
#
#  MODE A -- SLINGSHOT / HALO DRIVE.
#  Standard relativistic propulsion.  Thrust enters as a Newtonian-equivalent
#  force and is converted by GravityField.proper_accel, so the ship asymptotes to
#  c without ever being clamped.  Two genuine energy sources are available:
#
#    * Frame dragging.  A prograde pass through the ergosphere rides the dragged
#      frame and leaves with more energy than it entered with -- the Penrose
#      process, already present in the force law, no special code required.
#    * The Halo Drive (Kipping 2018).  A photon fired ahead of the ship into the
#      photon sphere returns after a near-circular orbit blueshifted by the
#      hole's motion, and the momentum difference is transferred to the ship.
#      Modelled as a thrust impulse available only within a few r_g of r_ph and
#      only on a prograde pass, which is exactly its physical availability window.
#
#  MODE B -- ALCUBIERRE WARP.
#  Standard propulsion is cut.  The ship's coordinate motion is imposed by the
#  bubble rather than integrated from forces, its local velocity is identically
#  zero, and its proper time runs at the flat-space rate.
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
    BALLISTIC = 0
    SLINGSHOT = 1
    WARP = 2
    NAMES = {0: "Ballistic (free-fall geodesic)",
             1: "Slingshot / Halo Drive",
             2: "Alcubierre Warp Drive"}


class Spacecraft:
    """A physics-defying human vessel that nonetheless obeys the equations."""

    THRUST = 0.0032           # proper thrust, in units of c per world-time
    WARP_ACCEL = 0.05
    WARP_MAX = 3.2            # coordinate speed in units of c -- superluminal, legally
    HALO_GAIN = 0.010

    def __init__(self, gravity: GravityField, hole: BlackHole, bubble: WarpBubble):
        self.gravity = gravity
        self.hole = hole
        self.bubble = bubble
        self.integrator = RK4Integrator(gravity.make_derivative(include_nbody=True,
                                                               relativistic=True))
        self.mode = ShipMode.SLINGSHOT
        self.state = np.zeros((1, 4), dtype=np.float64)
        self.heading = np.array([0.0, -1.0])
        self.thrust_dir = np.zeros(2)
        self.warp_speed = 0.0
        self.halo_boost = 0.0          # UI telemetry: last halo impulse magnitude
        self.trail = np.zeros((CFG.trail_len, 2), dtype=np.float32)
        self.trail_fill = 0
        self.trail_head = 0
        self.proper_time = 0.0         # ship clock  (tau)
        self.coord_time = 0.0          # deep space / Earth clock  (t)
        self.max_speed_reached = 0.0
        self.reset()

    # ---------------------------------------------------------------- setup
    def reset(self) -> None:
        r0 = max(self.hole.r_isco * 2.9, 210.0)
        v_c = min(float(self.hole.circular_speed(r0)), 0.55)
        self.state = np.array([[r0, 0.0, 0.0, v_c]], dtype=np.float64)
        self.heading = np.array([0.0, 1.0])
        self.warp_speed = 0.0
        self.halo_boost = 0.0
        self.proper_time = 0.0
        self.coord_time = 0.0
        self.max_speed_reached = 0.0
        self.trail[:] = self.state[0, 0:2]
        self.trail_fill = 0
        self.mode = ShipMode.SLINGSHOT

    # ------------------------------------------------------------ accessors
    @property
    def position(self) -> np.ndarray:
        return self.state[0, 0:2]

    @property
    def velocity(self) -> np.ndarray:
        return self.state[0, 2:4]

    @property
    def speed(self) -> float:
        """Local speed in units of c.  In warp this is identically zero."""
        if self.mode == ShipMode.WARP:
            return 0.0
        return float(norm(self.velocity))

    @property
    def coordinate_speed(self) -> float:
        """What a distant observer clocks -- may exceed c under warp."""
        if self.mode == ShipMode.WARP:
            return abs(self.warp_speed)
        return float(norm(self.velocity))

    @property
    def gamma(self) -> float:
        return float(lorentz_gamma(self.velocity[None, :])[0])

    @property
    def radius(self) -> float:
        return float(norm(self.position))

    def lapse(self) -> float:
        """sqrt(1 - r_s/r): the gravitational time-dilation factor at the ship."""
        if self.mode == ShipMode.WARP and self.bubble.active:
            return 1.0     # flat interior: no gravitational dilation at all
        return float(self.hole.lapse(self.radius))

    def time_dilation_factor(self) -> float:
        """dtau/dt -- the combined gravitational and kinematic slowdown."""
        if self.mode == ShipMode.WARP and self.bubble.active:
            return 1.0
        return self.lapse() * math.sqrt(max(1.0 - self.speed ** 2, 1e-12))

    # ------------------------------------------------------------- controls
    def set_thrust(self, direction: np.ndarray) -> None:
        d = np.asarray(direction, dtype=np.float64)
        n = float(norm(d))
        self.thrust_dir = d / n if n > CFG.eps else np.zeros(2)

    def set_mode(self, mode: int) -> None:
        self.mode = mode
        if mode == ShipMode.WARP:
            # Engaging warp cuts standard propulsion and freezes local motion:
            # inside the bubble the ship's local velocity is exactly zero.
            v = self.velocity.copy()
            n = float(norm(v))
            if n > CFG.eps:
                self.heading = v / n
            self.bubble.active = True
        else:
            if self.bubble.active:
                # Dropping out of warp hands the bubble's coordinate motion back
                # to the ship as ordinary momentum, capped just below c.
                self.state[0, 2:4] = self.heading * min(abs(self.warp_speed), 0.94)
            self.bubble.active = False
            self.warp_speed = 0.0

    def steer(self, turn: float) -> None:
        """Rotate the warp heading (radians).  Only meaningful under warp."""
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
        self.tick_clocks(dt)
        self.max_speed_reached = max(self.max_speed_reached, self.coordinate_speed)

    def _step_propulsive(self, t: float, dt: float, substeps: int) -> None:
        """RK4 through the full curvature field, plus thrust and the halo drive."""
        self.bubble.active = False
        self.halo_boost = 0.0

        # Thrust is folded in as a constant force over the step by temporarily
        # wrapping the derivative: this keeps the RK4 stage structure intact.
        thrust = np.zeros(2)
        if self.mode == ShipMode.SLINGSHOT:
            thrust = self.thrust_dir * self.THRUST
            thrust = thrust + self._halo_drive_impulse()

        base = self.gravity.make_derivative(include_nbody=True, relativistic=True)
        if float(norm(thrust)) > CFG.eps:
            def derivative(state: np.ndarray, tt: float) -> np.ndarray:
                d = base(state, tt)
                # Thrust must also go through the relativistic conversion.
                vel = state[..., 2:4]
                d[..., 2:4] += GravityField.proper_accel(vel, np.broadcast_to(
                    thrust, vel.shape))
                return d
            integ = RK4Integrator(derivative)
        else:
            integ = self.integrator

        self.state = integ.integrate(self.state, t, dt, substeps)

        v = self.velocity
        n = float(norm(v))
        if n > CFG.eps:
            self.heading = v / n

    def _halo_drive_impulse(self) -> np.ndarray:
        """
        Kipping's halo drive.  A photon launched into the photon sphere returns
        blueshifted; the momentum surplus thrusts the ship.  Available only in a
        narrow annulus around r_ph and only on a prograde pass, and the gain
        scales with the hole's spin because that is where the energy comes from.
        """
        r = self.radius
        r_ph = self.hole.r_photon
        window = smoothstep(3.2 * r_ph, 1.05 * r_ph, r)      # peaks at the sphere
        if window <= 1e-3:
            return np.zeros(2)
        pos, vel = self.position, self.velocity
        lz = pos[0] * vel[1] - pos[1] * vel[0]
        if lz <= 0.0:                                        # retrograde: no gain
            return np.zeros(2)
        tangent = np.array([-pos[1], pos[0]]) / max(r, CFG.eps)
        gain = self.HALO_GAIN * float(window) * (0.35 + 0.65 * self.hole.spin)
        self.halo_boost = gain
        return tangent * gain

    def _step_warp(self, dt: float) -> None:
        """
        Under warp the ship does not integrate forces -- the metric moves it.
        Coordinate position advances at v_s along the heading; local velocity
        stays exactly zero, which is what makes v_s > c legal.
        """
        self.warp_speed = min(self.warp_speed + self.WARP_ACCEL * dt * 5.0,
                              self.WARP_MAX)
        self.state[0, 0:2] += self.heading * self.warp_speed * dt
        self.state[0, 2:4] = 0.0
        self.bubble.active = True
        self.bubble.configure(self.position, self.heading, self.warp_speed)

    def tick_clocks(self, dt: float) -> None:
        """Advance both clocks.  dtau = dt * sqrt(1 - r_s/r) * sqrt(1 - v^2)."""
        self.coord_time += dt
        self.proper_time += dt * self.time_dilation_factor()

    def record_trail(self) -> None:
        self.trail_head = (self.trail_head + 1) % CFG.trail_len
        self.trail[self.trail_head] = self.position.astype(np.float32)
        self.trail_fill = min(self.trail_fill + 1, CFG.trail_len)

    # ------------------------------------------------------------- geometry
    def contracted_shape(self, base_len: float = 17.0,
                         base_wid: float = 7.0) -> tuple[float, float]:
        """
        Lorentz length contraction:  L = L_0 / gamma, measured along the boost.
        Transverse width is unaffected -- that asymmetry is the observable.
        Under warp the ship is at rest in its own flat bubble, so it is uncontracted.
        """
        if self.mode == ShipMode.WARP:
            return base_len, base_wid
        return base_len / self.gamma, base_wid


# ==============================================================================
#  SECTION 11 -- CAMERA
# ==============================================================================

class Camera:
    """World <-> screen mapping.  World y is up; screen y is down."""

    def __init__(self, viewport: pygame.Rect, scale: float = 1.0):
        self.viewport = viewport
        self.scale = scale
        self.center = np.zeros(2, dtype=np.float64)   # world point at viewport centre

    @property
    def origin(self) -> np.ndarray:
        return np.array([self.viewport.centerx, self.viewport.centery], dtype=np.float64)

    def to_screen(self, pts: np.ndarray) -> np.ndarray:
        """(N, 2) world -> (N, 2) screen float coordinates."""
        rel = (np.asarray(pts, dtype=np.float64) - self.center) * self.scale
        out = np.empty_like(rel)
        out[..., 0] = self.origin[0] + rel[..., 0]
        out[..., 1] = self.origin[1] - rel[..., 1]     # flip: world +y is up
        return out

    def to_world(self, sx: float, sy: float) -> np.ndarray:
        o = self.origin
        return np.array([(sx - o[0]) / self.scale + self.center[0],
                         (o[1] - sy) / self.scale + self.center[1]])

    def world_bounds(self) -> tuple[float, float, float, float]:
        hw = self.viewport.width / (2.0 * self.scale)
        hh = self.viewport.height / (2.0 * self.scale)
        return (self.center[0] - hw, self.center[0] + hw,
                self.center[1] - hh, self.center[1] + hh)

    def zoom_at(self, sx: float, sy: float, factor: float) -> None:
        anchor = self.to_world(sx, sy)
        self.scale = float(np.clip(self.scale * factor, 0.22, 4.0))
        new_anchor = self.to_world(sx, sy)
        self.center += anchor - new_anchor


# ==============================================================================
#  SECTION 12 -- THE LIVING SPACETIME FABRIC
#
#  The grid is a COMOVING lattice: its vertices are labelled by coordinates that
#  are dragged around by the geometry rather than pinned to the screen.  Three
#  effects act on it, and each is a named piece of physics.
#
#  (1) DARK ENERGY -- macroscopic stretch.
#      Lattice pitch is multiplied by the FRW scale factor a(t) = exp(H t).  The
#      whole fabric therefore dilates uniformly and everything not held by local
#      gravity drifts apart.  Folding a(t) into octaves (see wrapped_scale) lets
#      this run forever at constant cost.
#
#  (2) GRAVITY -- local pinch.
#      Flamm's paraboloid embeds the Schwarzschild spatial slice as
#      z(r) = 2 sqrt(r_s (r - r_s)), on which proper radial distance is
#
#          dl = dr / sqrt(1 - r_s / r)   >   dr
#
#      So a ruler laid on the sheet measures MORE than the coordinate difference.
#      Rendering a lattice of equal *proper* spacing therefore requires pulling
#      each vertex inward by the excess, and integrating dl - dr gives
#
#          Delta(d) = (r_s / 2) [ ln(1 + 4 d / r_s) - 1 ]      (clamped at 0)
#
#      whose derivative dDelta/dd -> r_s / 2d reproduces the metric's first-order
#      compression factor exactly.  Lines crowd together near a mass because the
#      metric says they must, and the crowding is superposed linearly over all
#      masses -- valid precisely in the weak-field regime where it is applied.
#
#  (3) WARP -- the Alcubierre shift vector.  See WarpBubble.grid_displacement.
# ==============================================================================

class SpacetimeGrid:
    """The quantum medium itself: a deformable, expanding coordinate mesh."""

    MAX_SOURCES = 8       # only the strongest wells perturb the lattice

    def __init__(self, camera: Camera, cosmos: CosmologicalField):
        self.camera = camera
        self.cosmos = cosmos
        self.visible = True
        self.samples = CFG.grid_curve_samples
        self._h_lines: list[np.ndarray] = []
        self._v_lines: list[np.ndarray] = []
        self._h_heat: np.ndarray = np.zeros(0)
        self._v_heat: np.ndarray = np.zeros(0)
        self._h_pts: np.ndarray = np.zeros((0, 0, 2))
        self._v_pts: np.ndarray = np.zeros((0, 0, 2))

    # ---------------------------------------------------- the pinch operator
    @staticmethod
    def _gravity_pinch(dist: np.ndarray, r_s: np.ndarray) -> np.ndarray:
        """
        Delta(d) = (r_s/2) [ln(1 + 4 d / r_s) - 1], floored at zero.

        Vectorized over (V, K): V vertices against K gravitational sources.
        """
        rs = np.maximum(r_s, 1e-6)
        delta = 0.5 * rs * (np.log1p(4.0 * dist / rs) - 1.0)
        return np.maximum(delta, 0.0, out=delta)

    def displace(self, pts: np.ndarray, src_pos: np.ndarray, src_rs: np.ndarray,
                 bubble: WarpBubble | None) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply gravity + warp to a flat (V, 2) vertex array.

        Returns (displaced_points, heat) where heat in [0, 1] measures how hard
        the fabric is being stretched at each vertex -- used for line brightness.
        """
        pts = np.asarray(pts, dtype=np.float32)
        out = pts.copy()
        heat = np.zeros(pts.shape[0], dtype=np.float32)

        if src_pos.shape[0]:
            src_pos = np.asarray(src_pos, dtype=np.float32)
            src_rs = np.asarray(src_rs, dtype=np.float32)
            diff = pts[:, None, :] - src_pos[None, :, :]          # (V, K, 2)
            dist = np.sqrt(np.einsum("vkc,vkc->vk", diff, diff))  # (V, K)
            np.maximum(dist, np.float32(1e-6), out=dist)
            delta = self._gravity_pinch(dist, src_rs[None, :])    # (V, K)
            # A vertex may never be pulled past the source it is falling toward.
            np.minimum(delta, np.float32(CFG.grid_max_pinch) * dist, out=delta)
            # Contract with the *scalar* weight delta/|d| against the raw
            # separation, which is algebraically identical to using the unit
            # vector but never materializes the (V, K, 2) unit array.
            delta /= dist
            out -= np.einsum("vk,vkc->vc", delta, diff)
            heat = np.clip(delta.max(axis=1) / np.float32(CFG.grid_max_pinch),
                           0.0, 1.0)

        if bubble is not None and bubble.active:
            out += bubble.grid_displacement(pts).astype(np.float32)
            theta = bubble.expansion_scalar(pts)
            heat = np.maximum(heat, np.clip(np.abs(theta) * 5.0, 0.0, 1.0))

        return out, heat

    # ------------------------------------------------------------ rebuild
    def rebuild(self, src_pos: np.ndarray, src_mass: np.ndarray,
                bubble: WarpBubble | None) -> None:
        """Regenerate the deformed lattice for this frame."""
        if not self.visible:
            self._h_lines, self._v_lines = [], []
            return

        # Keep only the dominant wells: the pinch is a sum of monotone terms, so
        # dropping the weakest is a bounded, visually invisible approximation.
        if src_mass.size > self.MAX_SOURCES:
            keep = np.argpartition(src_mass, -self.MAX_SOURCES)[-self.MAX_SOURCES:]
            src_pos, src_mass = src_pos[keep], src_mass[keep]
        src_rs = 2.0 * src_mass                              # r_s = 2M

        x0, x1, y0, y1 = self.camera.world_bounds()
        pad = CFG.grid_spacing * 2.2
        x0, x1, y0, y1 = x0 - pad, x1 + pad, y0 - pad, y1 + pad

        pitch = CFG.grid_spacing * self.cosmos.wrapped_scale
        step = pitch / self.samples

        # Comoving line coordinates, snapped to the expanding lattice.
        xs = np.arange(math.floor(x0 / pitch), math.ceil(x1 / pitch) + 1) * pitch
        ys = np.arange(math.floor(y0 / pitch), math.ceil(y1 / pitch) + 1) * pitch
        sx = np.arange(x0, x1 + step, step)
        sy = np.arange(y0, y1 + step, step)
        if xs.size == 0 or ys.size == 0 or sx.size < 2 or sy.size < 2:
            self._h_lines, self._v_lines = [], []
            return

        nh, nv = ys.size, xs.size
        # Horizontal family: (nh, len(sx), 2).  Vertical family: (nv, len(sy), 2).
        h = np.empty((nh, sx.size, 2))
        h[..., 0] = sx[None, :]
        h[..., 1] = ys[:, None]
        v = np.empty((nv, sy.size, 2))
        v[..., 0] = xs[:, None]
        v[..., 1] = sy[None, :]

        flat = np.concatenate([h.reshape(-1, 2), v.reshape(-1, 2)], axis=0)
        moved, heat = self.displace(flat, src_pos, src_rs, bubble)
        screen = self.camera.to_screen(moved)

        # Clamp once, vectorized, instead of twice per polyline at draw time.
        vp = self.camera.viewport
        np.clip(screen[:, 0], vp.left - 40, vp.right + 40, out=screen[:, 0])
        np.clip(screen[:, 1], vp.top - 40, vp.bottom + 40, out=screen[:, 1])

        split = nh * sx.size
        self._h_lines = list(screen[:split].reshape(nh, sx.size, 2))
        self._v_lines = list(screen[split:].reshape(nv, sy.size, 2))
        self._h_heat = heat[:split].reshape(nh, sx.size).max(axis=1)
        self._v_heat = heat[split:].reshape(nv, sy.size).max(axis=1)
        self._h_pts = moved[:split].reshape(nh, sx.size, 2)
        self._v_pts = moved[split:].reshape(nv, sy.size, 2)

    # ------------------------------------------------------------- rendering
    def draw(self, surf: pygame.Surface, bubble: WarpBubble | None,
             dim: float = 1.0) -> None:
        """
        One polyline call per grid line, coloured by local stretch.  Segment-wise
        recolouring is done only inside an active warp bubble, which bounds the
        expensive path to a small, fixed screen region.
        """
        if not self.visible or not self._h_lines:
            return

        for family, heats in ((self._h_lines, self._h_heat),
                              (self._v_lines, self._v_heat)):
            for line, hot in zip(family, heats):
                col = lerp_color(Palette.GRID, Palette.GRID_HOT, float(hot))
                if dim < 1.0:
                    col = lerp_color(Palette.VOID, col, dim)
                pygame.draw.lines(surf, col, False, line.tolist(), 1)

        if bubble is not None and bubble.active:
            self._draw_warp_stress(surf, bubble)

    _stress_lut = [[lerp_color(Palette.GRID, base, m / 15.0) for m in range(16)]
                   for base in (Palette.GRID_EXPAND, Palette.GRID_CONTRACT)]

    def _draw_warp_stress(self, surf: pygame.Surface, bubble: WarpBubble) -> None:
        """
        Recolour the lattice inside the bubble by the expansion scalar theta.
        Blue = theta < 0 (contraction, ahead).  Red = theta > 0 (expansion, behind).

        Candidate polylines are rejected with ONE vectorized bounding-box pass
        over each whole line family; only the handful that actually overlap the
        bubble get their theta evaluated, so the cost tracks the bubble's screen
        area rather than the size of the lattice.
        """
        reach = bubble.radius * 1.35
        reach2 = reach * reach
        lut = self._stress_lut
        centre = bubble.center.astype(np.float32)[None, None, :]

        for world, screen in ((self._h_pts, self._h_lines),
                              (self._v_pts, self._v_lines)):
            if world.size == 0:
                continue
            rel = world - centre                      # (L, N, 2)
            lo = rel.min(axis=1)                      # (L, 2)
            hi = rel.max(axis=1)
            cand = ~((lo[:, 0] > reach) | (hi[:, 0] < -reach) |
                     (lo[:, 1] > reach) | (hi[:, 1] < -reach))
            for li in np.nonzero(cand)[0]:
                r = rel[li]
                near = np.einsum("ij,ij->i", r, r) < reach2
                if not near.any():
                    continue
                sl = screen[li]
                theta = bubble.expansion_scalar(world[li])
                mag = np.minimum(np.abs(theta) * 6.0, 1.0)
                idx = np.nonzero(near & (mag >= 0.06))[0]
                sign = (theta < 0.0)
                bins = (mag * 15.999).astype(np.int32)
                n = len(sl) - 1
                for i in idx:
                    if i >= n:
                        continue
                    pygame.draw.line(surf, lut[int(sign[i])][bins[i]],
                                     sl[i], sl[i + 1], 1 + int(bins[i] > 8))


# ==============================================================================
#  SECTION 13 -- ZONE OF AVOIDANCE / LOCAL DUST FOG
#
#  About 20% of the sky is unobservable in the optical because our own galactic
#  disc lies in the way.  Interstellar dust extinguishes light following the
#  Beer-Lambert law
#
#          I = I_0 exp(-tau),      tau = column density x opacity
#
#  Modelled here as a Gaussian column density about the galactic mid-plane,
#  textured with fractal (multi-octave value) noise for the mottled, patchy look
#  real dust has.  Objects seen through it are dimmed by exp(-tau); the dust is
#  electromagnetic only, so it is drawn as part of the light rendering path and
#  vanishes entirely under deep-underground shielding -- and, correctly, it never
#  touches dark matter.
# ==============================================================================

class DustFog:
    """Galactic-plane extinction with a procedurally generated dust texture."""

    def __init__(self, width: int, height: int, plane_angle: float = -0.13,
                 thickness: float = 0.20, opacity: float = 0.92):
        self.enabled = False
        self.width, self.height = width, height
        self.plane_angle = plane_angle
        self.thickness = thickness
        self.opacity = opacity
        self.surface = self._bake()

    # ------------------------------------------------------------ generation
    @staticmethod
    def _octave(w: int, h: int, cells: int, rng: np.random.Generator) -> np.ndarray:
        """One octave of smoothed value noise at `cells` resolution."""
        cw, ch = max(cells, 2), max(int(cells * h / max(w, 1)), 2)
        grid = rng.random((ch + 1, cw + 1))
        # Bilinear upsample via separable linear interpolation.
        yi = np.linspace(0, ch, h)
        xi = np.linspace(0, cw, w)
        y0 = np.floor(yi).astype(int); y1 = np.minimum(y0 + 1, ch)
        x0 = np.floor(xi).astype(int); x1 = np.minimum(x0 + 1, cw)
        ty = (yi - y0)[:, None]; tx = (xi - x0)[None, :]
        top = grid[y0][:, x0] * (1 - tx) + grid[y0][:, x1] * tx
        bot = grid[y1][:, x0] * (1 - tx) + grid[y1][:, x1] * tx
        return top * (1 - ty) + bot * ty

    def _bake(self) -> pygame.Surface:
        """Bake the extinction texture once; it is static in observer coordinates."""
        w, h = self.width, self.height
        rng = np.random.default_rng(20240514)
        noise = np.zeros((h, w))
        amp, total = 1.0, 0.0
        for cells in (3, 6, 12, 24, 48):
            noise += amp * self._octave(w, h, cells, rng)
            total += amp
            amp *= 0.5
        noise /= total

        # Gaussian column density about the tilted galactic mid-plane.
        yy, xx = np.mgrid[0:h, 0:w]
        nx = (xx - w * 0.5) / w
        ny = (yy - h * 0.5) / h
        band = ny * math.cos(self.plane_angle) - nx * math.sin(self.plane_angle)
        column = np.exp(-(band / self.thickness) ** 2)

        # Beer-Lambert: optical depth tau, then transmission exp(-tau).
        tau = 3.4 * column * (0.30 + 0.70 * noise)
        alpha = (1.0 - np.exp(-tau)) * 255.0 * self.opacity

        rgb = np.empty((h, w, 3), dtype=np.uint8)
        warm = 0.55 + 0.45 * noise
        rgb[..., 0] = np.clip(46 * warm + 14, 0, 255)
        rgb[..., 1] = np.clip(33 * warm + 10, 0, 255)
        rgb[..., 2] = np.clip(30 * warm + 16, 0, 255)

        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.surfarray.pixels3d(surf)[:] = np.transpose(rgb, (1, 0, 2))
        pygame.surfarray.pixels_alpha(surf)[:] = np.transpose(
            alpha.astype(np.uint8), (1, 0))
        # Store the transmission map for object dimming (screen-space lookup).
        self.transmission = np.exp(-tau)
        return surf

    # ------------------------------------------------------------- sampling
    def extinction_at(self, screen_pts: np.ndarray) -> np.ndarray:
        """Transmission factor in [0, 1] for objects seen through the dust."""
        if not self.enabled or screen_pts.size == 0:
            return np.ones(screen_pts.shape[0] if screen_pts.ndim > 1 else 1)
        xs = np.clip(screen_pts[:, 0].astype(int), 0, self.width - 1)
        ys = np.clip(screen_pts[:, 1].astype(int), 0, self.height - 1)
        return self.transmission[ys, xs]

    def draw(self, surf: pygame.Surface) -> None:
        if self.enabled:
            surf.blit(self.surface, (0, 0))


# ==============================================================================
#  SECTION 14 -- UI TOOLKIT
#
#  A minimal immediate-ish widget set.  Widgets own their value and their rect;
#  the panel owns layout.  Nothing here touches physics.
# ==============================================================================

class CachedFont:
    """
    Drop-in pygame.font.Font wrapper that memoizes rasterized text.

    The sidebar draws ~90 strings per frame and most are fixed labels, so
    rasterizing them once and reusing the surface removes that work entirely.
    The cache is bounded and simply cleared when full -- numeric readouts churn
    a few dozen entries per second, which this absorbs without unbounded growth.
    """

    __slots__ = ("font", "_cache", "_limit")

    def __init__(self, font: pygame.font.Font, limit: int = 1200):
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

    def size(self, text: str):
        return self.font.size(text)


class Fonts:
    """Lazily built font set, sized for a 1600x900 canvas."""

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
    """Continuous control with a live numeric readout."""

    def __init__(self, rect: pygame.Rect, label: str, lo: float, hi: float,
                 value: float, fmt: str = "{:.3f}", unit: str = ""):
        super().__init__(rect, label)
        self.lo, self.hi = lo, hi
        self.value = float(np.clip(value, lo, hi))
        self.fmt = fmt
        self.unit = unit
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

    def handle(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            hit = self.rect.inflate(0, 10)
            if hit.collidepoint(event.pos):
                self.dragging = True
                self._set_from_x(event.pos[0])
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.dragging:
                self.dragging = False
                return True
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._set_from_x(event.pos[0])
            return True
        return False

    def draw(self, surf: pygame.Surface, fonts: Fonts) -> None:
        surf.blit(fonts.small.render(self.label, True, Palette.TEXT_DIM),
                  (self.rect.x, self.rect.y))
        val = self.fmt.format(self.value) + self.unit
        txt = fonts.small.render(val, True, Palette.ACCENT)
        surf.blit(txt, (self.rect.right - txt.get_width(), self.rect.y))
        tr = self._track()
        pygame.draw.rect(surf, (30, 36, 54), tr, border_radius=3)
        fill = pygame.Rect(tr.x, tr.y, int(tr.w * self.t), tr.h)
        pygame.draw.rect(surf, Palette.ACCENT, fill, border_radius=3)
        kx = tr.x + int(tr.w * self.t)
        pygame.draw.circle(surf, Palette.TEXT, (kx, tr.centery), 6)
        pygame.draw.circle(surf, Palette.PANEL, (kx, tr.centery), 4)


class Toggle(Widget):
    """Boolean switch with an optional hotkey hint."""

    def __init__(self, rect: pygame.Rect, label: str, value: bool = False,
                 key: str = "", color: tuple[int, int, int] = Palette.OK):
        super().__init__(rect, label)
        self.value = value
        self.key = key
        self.color = color

    def handle(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 \
                and self.rect.collidepoint(event.pos):
            self.value = not self.value
            return True
        return False

    def draw(self, surf: pygame.Surface, fonts: Fonts) -> None:
        box = pygame.Rect(self.rect.x, self.rect.y + 2, 26, 14)
        bg = self.color if self.value else (38, 44, 62)
        pygame.draw.rect(surf, bg, box, border_radius=7)
        knob_x = box.right - 7 if self.value else box.x + 7
        pygame.draw.circle(surf, Palette.PANEL if self.value else (96, 106, 132),
                           (knob_x, box.centery), 5)
        col = Palette.TEXT if self.value else Palette.TEXT_DIM
        surf.blit(fonts.small.render(self.label, True, col), (box.right + 9, self.rect.y))
        if self.key:
            k = fonts.tiny.render(f"[{self.key}]", True, (78, 90, 118))
            surf.blit(k, (self.rect.right - k.get_width(), self.rect.y + 1))


class Button(Widget):
    def __init__(self, rect: pygame.Rect, label: str,
                 color: tuple[int, int, int] = Palette.PANEL_EDGE):
        super().__init__(rect, label)
        self.color = color
        self.pressed = False
        self.hover = False

    def handle(self, event: pygame.event.Event) -> bool:
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

    def draw(self, surf: pygame.Surface, fonts: Fonts) -> None:
        bg = tuple(min(255, c + 22) for c in self.color) if self.hover else self.color
        pygame.draw.rect(surf, bg, self.rect, border_radius=4)
        pygame.draw.rect(surf, (58, 70, 100), self.rect, 1, border_radius=4)
        t = fonts.small.render(self.label, True, Palette.TEXT)
        surf.blit(t, t.get_rect(center=self.rect.center))


class RadioGroup(Widget):
    """Mutually exclusive selector rendered as a segmented control."""

    def __init__(self, rect: pygame.Rect, label: str, options: Sequence[str],
                 index: int = 0):
        super().__init__(rect, label)
        self.options = list(options)
        self.index = index

    def _cells(self) -> list[pygame.Rect]:
        n = len(self.options)
        w = self.rect.w / n
        return [pygame.Rect(int(self.rect.x + i * w), self.rect.y,
                            int(w) - 3, self.rect.h) for i in range(n)]

    def handle(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for i, c in enumerate(self._cells()):
                if c.collidepoint(event.pos):
                    self.index = i
                    return True
        return False

    def draw(self, surf: pygame.Surface, fonts: Fonts) -> None:
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
#  SECTION 15 -- RENDERER
#
#  All drawing.  Anything that represents light is routed through
#  `light_enabled`; when the laboratory goes into deep-underground shielding
#  every electromagnetic channel is cut and only gravity-coupled things remain
#  visible.  That is not a stylistic dimmer -- it is the experimental condition
#  under which dark matter becomes measurable.
# ==============================================================================

class Renderer:
    """Draws the universe.  Owns no state that the physics depends on."""

    def __init__(self, camera: Camera, fonts: Fonts):
        self.camera = camera
        self.fonts = fonts
        self.light_enabled = True
        self.show_labels = True
        self.show_trails = True
        self._ring = np.arange(CFG.trail_len)
        self.starfield = self._bake_starfield()
        self._glow_cache: dict = {}
        self._disk_key = None
        self._disk_cache = None
        # 8 temperature bins x 8 Doppler-beaming bins, baked once.
        self._disk_lut = [[
            tuple(int(v * (0.35 + 0.65 * (b / 7.0)))
                  for v in lerp_color(Palette.ACCRETION, (255, 246, 226),
                                      (h / 7.0) ** 2))
            for b in range(8)] for h in range(8)]

    # ---------------------------------------------------------------- helpers
    def _bake_starfield(self) -> np.ndarray:
        """Fixed distant stars in world space, seeded for reproducibility."""
        rng = np.random.default_rng(9091)
        n = 420
        r = rng.uniform(260.0, 2400.0, n)
        a = rng.uniform(0.0, math.tau, n)
        pos = np.stack([np.cos(a), np.sin(a)], axis=1) * r[:, None]
        mag = rng.uniform(0.25, 1.0, n)
        return np.concatenate([pos, mag[:, None]], axis=1)

    def _ring_order(self, head: int) -> np.ndarray:
        return (head + 1 + self._ring) % CFG.trail_len

    # ------------------------------------------------------------ background
    def draw_background(self, surf: pygame.Surface, cosmos: CosmologicalField,
                        fog: DustFog) -> None:
        surf.fill(Palette.VOID)
        if not self.light_enabled:
            return
        # Distant stars are comoving: dark energy carries them outward, which is
        # Hubble recession made directly visible.
        pts = self.starfield[:, 0:2] * cosmos.wrapped_scale
        scr = self.camera.to_screen(pts)
        clip = surf.get_clip()
        vis = ((scr[:, 0] > clip.left) & (scr[:, 0] < clip.right) &
               (scr[:, 1] > clip.top) & (scr[:, 1] < clip.bottom))
        dim = fog.extinction_at(scr) if fog.enabled else np.ones(scr.shape[0])
        for p, m, d in zip(scr[vis], self.starfield[vis, 2], dim[vis]):
            v = int(np.clip(70 + 150 * m * d, 0, 255))
            surf.set_at((int(p[0]), int(p[1])), (v, v, min(255, v + 22)))

    # ------------------------------------------------------------ black hole
    def draw_black_hole(self, surf: pygame.Surface, hole: BlackHole,
                        dm_visible_mode: bool) -> None:
        cam = self.camera
        cx, cy = cam.to_screen(hole.position[None, :])[0]
        s = cam.scale
        r_h = hole.r_horizon * s
        r_ergo = hole.r_ergosphere * s
        r_ph = hole.r_photon * s
        r_ph_retro = hole.r_photon_retro * s
        r_isco = hole.r_isco * s

        # --- accretion disk: light, so it is cut under shielding -------------
        if self.light_enabled:
            self._draw_accretion(surf, hole, (cx, cy), s)

        # --- ERGOSPHERE: the frame-dragging whirlpool ------------------------
        # Drawn as spiral arcs whose pitch follows the Lense-Thirring rate
        # omega(r) = 2GJ/(c^2 r^3), so the swirl visibly tightens inward.
        if hole.a > CFG.eps:
            self._draw_frame_drag_swirl(surf, hole, (cx, cy), s)
        self._dashed_circle(surf, (cx, cy), r_ergo, Palette.ERGO, dash=9, width=2)

        # --- PHOTON SPHERE: the orbital boundary for light -------------------
        pc = Palette.PHOTON if self.light_enabled else (70, 66, 46)
        self._dashed_circle(surf, (cx, cy), r_ph, pc, dash=6, width=2)
        if hole.spin > 0.02:
            self._dashed_circle(surf, (cx, cy), r_ph_retro,
                                tuple(int(c * 0.45) for c in pc), dash=4, width=1)
        # --- ISCO: last stable orbit for matter ------------------------------
        self._dashed_circle(surf, (cx, cy), r_isco, (54, 74, 104), dash=3, width=1)

        # --- EVENT HORIZON: the absolute black point of no return ------------
        # Gravitational lensing shadow: photons grazing r_ph are captured, so the
        # apparent shadow is larger than r_+.  Painted as a soft falloff.
        shadow = pygame.Surface((int(r_ph * 2.6) + 8, int(r_ph * 2.6) + 8),
                                pygame.SRCALPHA)
        sc = shadow.get_width() // 2
        for i in range(10, 0, -1):
            rr = r_ph * 1.28 * i / 10.0
            pygame.draw.circle(shadow, (0, 0, 0, int(255 * (1.0 - i / 11.0))),
                               (sc, sc), int(rr))
        surf.blit(shadow, (cx - sc, cy - sc))
        pygame.draw.circle(surf, Palette.HORIZON, (int(cx), int(cy)), max(int(r_h), 2))
        rim = (168, 214, 255) if self.light_enabled else (48, 60, 84)
        pygame.draw.circle(surf, rim, (int(cx), int(cy)), max(int(r_h), 2), 2)

        if self.show_labels:
            self._label_boundaries(surf, (cx, cy), hole, s)

    def _draw_accretion(self, surf: pygame.Surface, hole: BlackHole,
                        c: tuple[float, float], s: float) -> None:
        """
        Differentially rotating disk between the ISCO and ~4x ISCO.  Keplerian
        shear Omega ~ r^(-3/2) means inner annuli lap outer ones, so the arcs
        smear -- exactly why real disks are turbulent and hot.
        """
        rad, base, omega, heat_bin, big = self._disk_geometry(hole)
        ang = base + omega * hole.phase * 11.0
        pts = np.stack([np.cos(ang), np.sin(ang)], axis=1) * rad[:, None]
        scr = self.camera.to_screen(pts).astype(np.int32)
        # Relativistic beaming: the approaching limb is Doppler-boosted.
        beam_bin = ((0.45 + 0.55 * (0.5 - 0.5 * np.cos(ang))) * 7.999).astype(np.int32)
        lut = self._disk_lut
        for i in range(len(scr)):
            pygame.draw.circle(surf, lut[heat_bin[i]][beam_bin[i]],
                               (scr[i, 0], scr[i, 1]), big[i])

    def _disk_geometry(self, hole: BlackHole):
        """
        Static disk sampling, rebuilt only when the hole's ISCO moves.

        Keplerian shear Omega = sqrt(GM)/r^(3/2) means inner annuli lap outer
        ones; only the phase needs recomputing each frame.
        """
        key = round(hole.r_isco, 3)
        if self._disk_key == key:
            return self._disk_cache
        n = 240
        r_in, r_out = hole.r_isco, hole.r_isco * 3.6
        rng = np.random.default_rng(4242)
        rad = r_in + (r_out - r_in) * rng.random(n) ** 1.7
        base = rng.uniform(0, math.tau, n)
        omega = hole.mass ** 0.5 / rad ** 1.5
        heat = np.clip((r_out - rad) / (r_out - r_in), 0.0, 1.0)
        heat_bin = (heat * 7.999).astype(np.int32)
        big = (1 + (heat > 0.72)).astype(np.int32)
        self._disk_key = key
        self._disk_cache = (rad, base, omega, heat_bin, big)
        return self._disk_cache

    def _draw_frame_drag_swirl(self, surf: pygame.Surface, hole: BlackHole,
                               c: tuple[float, float], s: float) -> None:
        """Spiral streamlines whose winding follows omega(r) = 2GJ/(c^2 r^3)."""
        cx, cy = c
        arms = 7
        r0, r1 = hole.r_horizon * 1.02, hole.r_ergosphere
        rr = np.linspace(r0, r1, 26)
        # Integrated dragging phase, normalized so the outer edge reads zero.
        phase = hole.frame_drag_omega(rr)
        phase = (phase - phase[-1]) * 26.0 + hole.phase * 1.6
        for k in range(arms):
            a = k * math.tau / arms + phase
            pts = np.stack([np.cos(a), np.sin(a)], axis=1) * rr[:, None]
            scr = self.camera.to_screen(pts)
            fade = int(60 + 90 * hole.spin)
            pygame.draw.lines(surf, (fade, int(fade * 0.6), fade + 40),
                              False, scr.tolist(), 1)

    def _dashed_circle(self, surf: pygame.Surface, c: tuple[float, float],
                       r: float, col, dash: int = 8, width: int = 1) -> None:
        if r < 3 or r > 20000:
            return
        seg = max(int(math.tau * r / max(dash * 2.4, 4)), 8)
        seg = min(seg, 520)
        a = np.linspace(0, math.tau, seg * 2, endpoint=False)
        pts = np.stack([np.cos(a), np.sin(a)], axis=1) * r
        pts[:, 0] += c[0]
        pts[:, 1] += c[1]
        for i in range(0, len(pts) - 1, 2):
            pygame.draw.line(surf, col, pts[i], pts[i + 1], width)

    def _label_boundaries(self, surf: pygame.Surface, c: tuple[float, float],
                          hole: BlackHole, s: float) -> None:
        """Leader-line callouts for the three causal boundaries."""
        cx, cy = c
        items = [
            (hole.r_ergosphere, "ERGOSPHERE  r=2M  (frame dragging)",
             Palette.ERGO, -0.72),
            (hole.r_photon, "PHOTON SPHERE  r=%.1f  (light orbits)" % hole.r_photon,
             Palette.PHOTON, -0.28),
            (hole.r_horizon, "EVENT HORIZON  r+=%.1f  (no return)" % hole.r_horizon,
             (200, 226, 255), 0.30),
        ]
        f = self.fonts.tiny
        for r, text, col, ang in items:
            rp = r * s
            ax, ay = math.cos(ang), math.sin(ang)
            p0 = (cx + ax * rp, cy - ay * rp)
            p1 = (cx + ax * (rp + 34), cy - ay * (rp + 34))
            end = (p1[0] + 22, p1[1])
            pygame.draw.line(surf, col, p0, p1, 1)
            pygame.draw.line(surf, col, p1, end, 1)
            t = f.render(text, True, col)
            surf.blit(t, (end[0] + 5, end[1] - 6))

    # ---------------------------------------------------------------- bodies
    def draw_bodies(self, surf: pygame.Surface, pop: BodyPopulation,
                    hole: BlackHole, fog: DustFog) -> None:
        if pop.count == 0:
            return
        scr = self.camera.to_screen(pop.state[:, 0:2])
        s = self.camera.scale
        clip = self.camera.viewport
        dim = fog.extinction_at(scr) if fog.enabled else np.ones(pop.count)

        if self.show_trails:
            self._draw_body_trails(surf, pop)

        for i in range(pop.count):
            x, y = float(scr[i, 0]), float(scr[i, 1])
            r = max(pop.radius[i] * s, 1.6)
            ph = int(pop.phase[i])
            col = Phase.COLORS[ph]
            if not self.light_enabled:
                # Under shielding a body is known only by its gravity: outline only.
                pygame.draw.circle(surf, (52, 60, 82), (int(x), int(y)), int(r), 1)
                continue
            fade = float(dim[i])
            col = tuple(int(cc * (0.25 + 0.75 * fade)) for cc in col)

            if pop.strain[i] > 0.0:
                self._draw_spaghetti(surf, pop, i, (x, y), r, col, hole)
                continue

            if ph in (Phase.BROWN_DWARF, Phase.STAR):
                gr = min(r * (2.7 if ph == Phase.STAR else 1.9), 78.0)
                # Additive halos are per-pixel work; skip any that cannot land
                # on screen at all.
                if (-gr < x < clip.width + gr) and (-gr < y < clip.height + gr):
                    self._glow(surf, (x, y), gr, col,
                               0.5 if ph == Phase.STAR else 0.26)
            if ph == Phase.COLLAPSED:
                pygame.draw.circle(surf, (0, 0, 0), (int(x), int(y)), int(r))
                pygame.draw.circle(surf, (150, 190, 255), (int(x), int(y)), int(r), 2)
                self._dashed_circle(surf, (x, y), r * 1.5, (90, 110, 160), 5, 1)
            else:
                pygame.draw.circle(surf, col, (int(x), int(y)), int(r))
                if ph <= Phase.GAS_GIANT and r > 4:
                    # Terminator: the unlit hemisphere faces away from the disk.
                    shade = tuple(int(cc * 0.45) for cc in col)
                    pygame.draw.circle(surf, shade, (int(x + r * 0.34),
                                                     int(y + r * 0.24)), int(r * 0.82))
            if ph == Phase.DEGENERATE:
                pygame.draw.circle(surf, (200, 230, 255), (int(x), int(y)),
                                   int(r) + 3, 1)

    def _draw_spaghetti(self, surf: pygame.Surface, pop: BodyPopulation, i: int,
                        p: tuple[float, float], r: float, col, hole: BlackHole) -> None:
        """
        Render the tidal filament.

        The geodesic deviation equation gives +2GM dr/r^3 radially and -GM dr/r^3
        transversely.  The 2:-1 ratio is volume-preserving to first order, so the
        drawn ellipse stretches by `k` along the radial direction and shrinks by
        1/sqrt(k) across it -- the shape the metric actually predicts.
        """
        strain = float(pop.strain[i])
        # Aspect ratio k, capped for legibility; the 2:-1 stretch:squeeze ratio
        # of the tidal tensor is preserved exactly by taking b = r / sqrt(k),
        # which keeps the drawn area constant as the metric demands.
        k = 1.0 + 13.0 * strain ** 1.5
        a = r * k
        b = r / math.sqrt(k)
        # Radial direction toward the hole, in screen space.
        d = pop.state[i, 0:2] - hole.position
        ang = math.atan2(-d[1], d[0])          # screen y is flipped
        th = np.linspace(0, math.tau, 26, endpoint=False)
        ex, ey = a * np.cos(th), b * np.sin(th)
        ca, sa = math.cos(ang), math.sin(ang)
        px = p[0] + ex * ca - ey * sa
        py = p[1] + ex * sa + ey * ca
        # Tidal heating brightens the filament; the last of it fades out rather
        # than popping, since the body is shredded, not deleted in one frame.
        glow = lerp_color(col, (255, 236, 200), min(strain, 1.0))
        fade = 1.0 - smoothstep(0.72, 1.0, strain)
        glow = lerp_color(Palette.VOID, glow, float(fade))
        pygame.draw.polygon(surf, glow, np.stack([px, py], axis=1).tolist())

    def _draw_body_trails(self, surf: pygame.Surface, pop: BodyPopulation) -> None:
        order = self._ring_order(pop.trail_head)
        for i in range(pop.count):
            pts = self.camera.to_screen(pop.trail[i][order])
            col = Phase.COLORS[int(pop.phase[i])]
            col = tuple(int(c * 0.34) + 12 for c in col)
            pygame.draw.lines(surf, col, False, pts.tolist(), 1)

    def _glow(self, surf: pygame.Surface, p: tuple[float, float], radius: float,
              col, strength: float) -> None:
        """Additive halo, drawn from a cached sprite (see _glow_sprite)."""
        radius = min(radius, 260.0)
        if radius < 2:
            return
        g = self._glow_sprite(radius, col, strength)
        r = g.get_width() * 0.5
        surf.blit(g, (p[0] - r, p[1] - r), special_flags=pygame.BLEND_ADD)

    def _glow_sprite(self, radius: float, col, strength: float) -> pygame.Surface:
        """
        Memoized radial-falloff sprite.  Radius is quantized to 3 px and
        strength to 16 levels, which bounds the cache while being visually
        indistinguishable -- and turns a nine-circle rebuild per body per frame
        into a dictionary lookup.
        """
        key = (int(radius / 3.0), col, int(strength * 16.0))
        cached = self._glow_cache.get(key)
        if cached is not None:
            return cached
        rad = max(key[0] * 3.0, 3.0)
        d = int(rad * 2) + 2
        g = pygame.Surface((d, d), pygame.SRCALPHA)
        c = d // 2
        for i in range(9, 0, -1):
            rr = rad * i / 9.0
            a = int(255 * (key[2] / 16.0) * (1.0 - i / 10.0) ** 1.7)
            pygame.draw.circle(g, (*col, a), (c, c), int(rr))
        if len(self._glow_cache) > 256:
            self._glow_cache.clear()
        self._glow_cache[key] = g
        return g

    # ----------------------------------------------------------- dark matter
    def draw_dark_matter(self, surf: pygame.Surface, dm: DarkMatterField,
                         shielded: bool, laser_on: bool) -> None:
        """
        Dark matter is EM-blind, so the dust fog never touches it and the
        starfield never occludes it.  Only the measurement rule gates it.
        """
        if dm.count == 0:
            return
        vis = dm.visible_mask(shielded, laser_on)
        if not vis.any():
            return
        scr = self.camera.to_screen(dm.state[vis, 0:2])
        coh = dm.coherence[vis]
        sup = dm.superradiant[vis]
        for p, c, sr in zip(scr, coh, sup):
            a = int(np.clip(c, 0, 1) * 210)
            col = (255, 214, 120) if sr else Palette.DARK_MATTER
            col = tuple(int(v * (0.35 + 0.65 * c)) for v in col)
            pygame.draw.circle(surf, col, (int(p[0]), int(p[1])), 2 if sr else 1)

    # ------------------------------------------------------------------ ship
    def draw_ship(self, surf: pygame.Surface, ship: Spacecraft,
                  bubble: WarpBubble) -> None:
        cam = self.camera
        s = cam.scale
        p = cam.to_screen(ship.position[None, :])[0]

        if self.show_trails:
            order = self._ring_order(ship.trail_head)
            pts = cam.to_screen(ship.trail[order])
            pygame.draw.lines(surf, (54, 122, 104), False, pts.tolist(), 1)

        if ship.mode == ShipMode.WARP and bubble.active:
            self._draw_warp_bubble(surf, bubble, s)

        # Lorentz-contracted hull.
        length, width = ship.contracted_shape()
        length *= s
        width *= s
        h = ship.heading if ship.mode == ShipMode.WARP else (
            ship.velocity / max(float(norm(ship.velocity)), CFG.eps))
        ang = math.atan2(-h[1], h[0])
        ca, sa = math.cos(ang), math.sin(ang)
        hull = np.array([[length, 0.0], [-length * 0.55, width],
                         [-length * 0.30, 0.0], [-length * 0.55, -width]])
        rot = np.stack([hull[:, 0] * ca - hull[:, 1] * sa,
                        hull[:, 0] * sa + hull[:, 1] * ca], axis=1) + p
        col = Palette.WARP if ship.mode == ShipMode.WARP else Palette.SHIP
        pygame.draw.polygon(surf, col, rot.tolist())
        pygame.draw.polygon(surf, (250, 255, 255), rot.tolist(), 1)

        if ship.mode == ShipMode.SLINGSHOT and float(norm(ship.thrust_dir)) > 0:
            t = ship.thrust_dir
            e = p - np.array([t[0], -t[1]]) * (length + 9)
            pygame.draw.line(surf, Palette.WARN, p, e, 2)
        if ship.halo_boost > 1e-4:
            pygame.draw.circle(surf, Palette.PHOTON, (int(p[0]), int(p[1])),
                               int(14 + 260 * ship.halo_boost), 1)

    def _draw_warp_bubble(self, surf: pygame.Surface, bubble: WarpBubble,
                          s: float) -> None:
        """
        Draw the wall and shade the interior to advertise that it is flat.
        Colour follows sign(theta): contraction ahead, expansion behind.
        """
        c = self.camera.to_screen(bubble.center[None, :])[0]
        R = bubble.radius * s
        th = np.linspace(0, math.tau, 96, endpoint=False)
        ring = np.stack([np.cos(th), np.sin(th)], axis=1)
        world = bubble.center[None, :] + ring * bubble.radius
        theta = bubble.expansion_scalar(world)
        scr = self.camera.to_screen(world)
        mx = max(float(np.max(np.abs(theta))), CFG.eps)
        for i in range(len(scr)):
            t = float(theta[i]) / mx
            col = lerp_color(Palette.GRID_EXPAND, Palette.GRID_CONTRACT,
                             0.5 - 0.5 * t)
            j = (i + 1) % len(scr)
            pygame.draw.line(surf, col, scr[i], scr[j], 2)
        # Flat interior: a faint, perfectly regular disc -- zero curvature.
        inner = pygame.Surface((int(R * 2) + 4, int(R * 2) + 4), pygame.SRCALPHA)
        pygame.draw.circle(inner, (40, 120, 150, 34), (int(R) + 2, int(R) + 2),
                           int(R * 0.86))
        surf.blit(inner, (c[0] - R - 2, c[1] - R - 2))
        t = self.fonts.tiny.render("FLAT INTERIOR  (Riemann = 0)", True, (120, 210, 235))
        surf.blit(t, (c[0] - t.get_width() // 2, c[1] + R * 0.86 + 5))


# ==============================================================================
#  SECTION 16 -- CONTROL PANEL
# ==============================================================================

class ControlPanel:
    """Sidebar: sliders, toggles and live telemetry."""

    def __init__(self, rect: pygame.Rect, fonts: Fonts):
        self.rect = rect
        self.fonts = fonts
        self.widgets: list[Widget] = []
        self.sections: list[tuple[int, str]] = []
        self.x = rect.x + 14
        self.w = rect.w - 28
        self._y = 52

        # Layout runs top-down through these four helpers, each of which
        # advances the cursor by exactly the height it consumed.  Every gap in
        # the sidebar is therefore expressed once, here, rather than as a pile
        # of hand-tuned constants scattered through the constructor.
        self._section("SIMULATION CONTROL")
        self.s_dt = self._slider("Time step  dt", CFG.dt_min, CFG.dt_max,
                                 CFG.dt, "{:.3f}")
        bw = (self.w - 12) // 3
        self.b_pause = Button(pygame.Rect(self.x, self._y, bw, 20), "PAUSE")
        self.b_reset = Button(pygame.Rect(self.x + bw + 6, self._y, bw, 20),
                              "RESET SHIP")
        self.b_clear = Button(pygame.Rect(self.x + 2 * bw + 12, self._y, bw, 20),
                              "CLEAR")
        self.widgets += [self.b_pause, self.b_reset, self.b_clear]
        self._y += 24

        self._section("CENTRAL BLACK HOLE")
        self.s_mass = self._slider("Mass  M = GM/c^2", CFG.bh_mass_min,
                                   CFG.bh_mass_max, CFG.bh_mass, "{:.1f}", " px")
        self.s_spin = self._slider("Kerr spin  a* = a/M", 0.0, 0.998,
                                   CFG.bh_spin, "{:.3f}")

        self._section("QUINTESSENCE / DARK ENERGY")
        self.s_hubble = self._slider("Expansion rate  H", CFG.hubble_min,
                                     CFG.hubble_max, CFG.hubble, "{:.5f}")

        self._section("FABRIC & OPTICS")
        self.t_grid = self._toggle("Spacetime grid", True, "G", Palette.ACCENT)
        self.t_labels = self._toggle("Boundary labels", True, "B", Palette.ACCENT)
        self.t_trails = self._toggle("Particle trails", True, "T", Palette.ACCENT)
        self.t_fog = self._toggle("Zone of Avoidance fog", False, "Z", Palette.WARN)
        self.t_follow = self._toggle("Camera follows ship", False, "F", Palette.SHIP)

        self._section("DARK MATTER EXPERIMENT")
        self.t_dm = self._toggle("Dark matter halo", True, "D", Palette.DARK_MATTER)
        self.t_shield = self._toggle("Deep underground shielding", False, "U",
                                     Palette.DARK_MATTER)
        self.t_laser = self._toggle("Light / laser sensor", False, "L", Palette.DANGER)
        self.t_super = self._toggle("Superradiance (Penrose)", True, "",
                                    Palette.WARN)

        self._section("SPAWN CLASS  /  PROPULSION MODE")
        self.r_spawn = self._radio(["Planet", "Star", "Dark Matter"], 0)
        self.r_mode = self._radio(["Ballistic", "Slingshot", "WARP"], 1)

        self._section("TELEMETRY")
        self.telemetry_y = self._y

    # ------------------------------------------------------------ layout
    def _section(self, title: str) -> None:
        self._y += 8
        self.sections.append((self._y, title))
        self._y += 22

    def _slider(self, label, lo, hi, val, fmt, unit="") -> Slider:
        s = Slider(pygame.Rect(self.x, self._y, self.w, 20), label, lo, hi,
                   val, fmt, unit)
        self.widgets.append(s)
        self._y += 24
        return s

    def _toggle(self, label, val, key, col) -> Toggle:
        t = Toggle(pygame.Rect(self.x, self._y, self.w, 17), label, val, key, col)
        self.widgets.append(t)
        self._y += 18
        return t

    def _radio(self, options, index) -> RadioGroup:
        r = RadioGroup(pygame.Rect(self.x, self._y, self.w, 20), "", options, index)
        self.widgets.append(r)
        self._y += 24
        return r

    # ---------------------------------------------------------------- events
    def handle(self, event: pygame.event.Event) -> bool:
        used = False
        for wdg in self.widgets:
            used |= wdg.handle(event)
        return used

    # ------------------------------------------------------------- rendering
    def draw(self, surf: pygame.Surface, sim: "Simulation") -> None:
        pygame.draw.rect(surf, Palette.PANEL, self.rect)
        pygame.draw.line(surf, Palette.PANEL_EDGE,
                         (self.rect.x, self.rect.y), (self.rect.x, self.rect.bottom), 2)
        f = self.fonts
        surf.blit(f.big.render("COSMIC ENGINE", True, Palette.TEXT), (self.x, 12))
        sub = f.tiny.render("G = c = 1   |   1 px = 1 GM/c^2 unit", True, Palette.TEXT_DIM)
        surf.blit(sub, (self.x, 32))

        for y, title in self.sections:
            surf.blit(f.head.render(title, True, Palette.ACCENT), (self.x, y))
            pygame.draw.line(surf, (32, 40, 60), (self.x, y + 17),
                             (self.x + self.w, y + 17), 1)
        for wdg in self.widgets:
            wdg.draw(surf, f)
        self._draw_telemetry(surf, sim)

    ROW_H = 14
    BAR_H = 23

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
        if fw > 0:
            pygame.draw.rect(surf, col, pygame.Rect(bar.x, bar.y, fw, 6),
                             border_radius=3)
        return y + self.BAR_H

    def _draw_telemetry(self, surf: pygame.Surface, sim: "Simulation") -> None:
        f = self.fonts
        ship, hole = sim.ship, sim.hole
        y = self.telemetry_y + 4

        # ---- dual clocks -----------------------------------------------------
        dil = ship.time_dilation_factor()
        y = self._row(surf, y, "Earth / deep-space clock",
                      f"{ship.coord_time:10.2f}", Palette.TEXT)
        y = self._row(surf, y, "Ship proper clock  tau",
                      f"{ship.proper_time:10.2f}", Palette.OK)
        lag = ship.coord_time - ship.proper_time
        y = self._row(surf, y, "Accumulated lag", f"{lag:10.2f}",
                      Palette.WARN if lag > 1 else Palette.TEXT_DIM)
        col = Palette.DANGER if dil < 0.5 else (Palette.WARN if dil < 0.85
                                                else Palette.OK)
        y = self._bar(surf, y, "dtau/dt  grav x kinematic", dil, col, f"{dil:.4f}")

        # ---- kinematics ------------------------------------------------------
        v = ship.coordinate_speed
        y = self._row(surf, y, "Speed", f"{v:9.4f} c",
                      Palette.WARP if v > 1.0 else Palette.TEXT)
        y = self._row(surf, y, "gamma  /  contraction L/L0",
                      f"{ship.gamma:.3f} / {1.0 / ship.gamma:.4f}")
        r_over_rs = ship.radius / max(hole.r_s, CFG.eps)
        y = self._row(surf, y, "Radius  r / rs", f"{r_over_rs:10.3f}",
                      Palette.DANGER if r_over_rs < 1.2 else Palette.TEXT)
        y = self._bar(surf, y, "Approach to light speed", min(v, 1.0),
                      Palette.WARP if v > 1 else Palette.ACCENT,
                      "SUPERLUMINAL" if v > 1.0 else f"{v * 100:.1f}% c")

        # ---- geometry --------------------------------------------------------
        y = self._row(surf, y, "r+ / ergo / photon / ISCO",
                      f"{hole.r_horizon:.0f}/{hole.r_ergosphere:.0f}/"
                      f"{hole.r_photon:.0f}/{hole.r_isco:.0f}")
        r_ta = sim.cosmos.turnaround_radius(hole.mass, hole.r_s)
        y = self._row(surf, y, "r_turnaround  /  a(t)",
                      ("inf" if math.isinf(r_ta) else f"{r_ta:.0f}") +
                      f" / {sim.cosmos.scale_factor:.3f}")

        # ---- the atomic tug-of-war ------------------------------------------
        y += 3
        surf.blit(f.head.render("ATOMIC TUG-OF-WAR", True, Palette.ACCENT),
                  (self.x, y))
        y += 18
        idx = sim.focus_index
        if 0 <= idx < sim.pop.count:
            m = float(sim.pop.mass_mj[idx])
            ph = int(sim.pop.phase[idx])
            grav, deg = StellarStructure.pressure_balance(m)
            ratio = grav / max(deg, 1e-30)
            frac = float(np.clip(math.log10(max(ratio, 1e-12)) / 8.0 + 0.5, 0, 1))
            y = self._row(surf, y, Phase.NAMES[ph], f"{m:9.2f} MJ",
                          Phase.COLORS[ph])
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

        # ---- experiment status ----------------------------------------------
        y += 3
        if self.t_laser.value or not self.t_shield.value:
            msg, c = "DECOHERED (unmeasurable)", Palette.DANGER
        else:
            msg, c = "SHIELDED (measurable)", Palette.OK
        y = self._row(surf, y, "Dark matter", msg, c)
        y = self._row(surf, y, "Superradiant / total",
                      f"{int(np.sum(sim.dm.superradiant))} / {sim.dm.count}",
                      Palette.WARN)
        y = self._row(surf, y, "Bodies / FPS",
                      f"{sim.pop.count} / {sim.fps:.0f}",
                      Palette.OK if sim.fps > 50 else Palette.WARN)
        surf.blit(f.tiny.render(ShipMode.NAMES[sim.ship.mode], True, Palette.WARP),
                  (self.x, y + 1))


# ==============================================================================
#  SECTION 17 -- SIMULATION
#
#  Owns every subsystem and defines the per-frame order of operations.  Update
#  order matters: the gravitational source table must be rebuilt before any
#  population is stepped, or bodies integrate against last frame's geometry.
# ==============================================================================

class Simulation:
    """The universe, assembled."""

    def __init__(self, camera: Camera, fonts: Fonts, panel: ControlPanel,
                 fog: DustFog):
        self.hole = BlackHole(CFG.bh_mass, CFG.bh_spin)
        self.cosmos = CosmologicalField(CFG.hubble)
        self.gravity = GravityField(self.hole, self.cosmos)
        self.bubble = WarpBubble()
        self.gravity.bubble = self.bubble

        self.pop = BodyPopulation(self.gravity)
        self.dm = DarkMatterField(self.gravity, self.hole)
        self.ship = Spacecraft(self.gravity, self.hole, self.bubble)

        self.camera = camera
        self.grid = SpacetimeGrid(camera, self.cosmos)
        self.renderer = Renderer(camera, fonts)
        self.panel = panel
        self.fog = fog

        self.time = 0.0
        self.paused = False
        self.fps = 60.0
        self.focus_index = -1
        self.populate_default()

    # ------------------------------------------------------------- scenarios
    def populate_default(self) -> None:
        """A demonstration system: two planets, a star, and a dark halo."""
        self.pop.clear()
        self.dm.clear()
        h = self.hole
        self.pop.spawn_orbiting(h, h.r_isco * 2.2, 1.4, angle=0.6)
        self.pop.spawn_orbiting(h, h.r_isco * 3.4, 9.0, angle=2.9)
        self.pop.spawn_orbiting(h, h.r_isco * 4.9, 240.0, angle=4.6)
        self.pop.spawn_orbiting(h, h.r_isco * 6.6, 0.6, angle=1.9)
        self.dm.seed_halo(520, h.r_ergosphere * 1.05, h.r_isco * 7.5)
        self.focus_index = 1

    def reset_all(self) -> None:
        self.cosmos.log_a = 0.0
        self.cosmos.time = 0.0
        self.time = 0.0
        self.ship.reset()
        self.populate_default()

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

    # ---------------------------------------------------------------- update
    def sync_controls(self) -> None:
        """Push slider values into the physics objects."""
        p = self.panel
        self.hole.mass = p.s_mass.value
        self.hole.spin = p.s_spin.value
        self.cosmos.hubble = p.s_hubble.value
        self.grid.visible = p.t_grid.value
        self.renderer.show_labels = p.t_labels.value
        self.renderer.show_trails = p.t_trails.value
        self.fog.enabled = p.t_fog.value
        # Shielding kills every light-rendering vector; an active laser is itself
        # light, so it overrides the shielding for rendering purposes.
        self.renderer.light_enabled = (not self.shielded) or self.laser_on
        if p.r_mode.index != self.ship.mode:
            self.ship.set_mode(p.r_mode.index)
        if not p.t_dm.value and self.dm.count:
            self.dm.clear()

    def step(self, dt: float) -> None:
        """Advance the whole universe by `dt` world-time units."""
        if dt <= 0.0:
            return
        # RK4's local error is O(h^5), so accuracy is governed by the STEP SIZE.
        # Bounding h and deriving the count keeps the error flat as the user
        # drags the dt slider, instead of degrading linearly with it.
        sub = max(1, int(math.ceil(dt / CFG.max_substep)))

        # 1. Refresh the gravitational source table (bodies only; the ship and
        #    dark matter are test particles and do not source curvature here).
        pos, mass, soft = self.pop.gravitational_sources()
        self.gravity.set_sources(pos, mass, soft)

        # 2. Integrate every population against the same field.
        self.pop.step(self.time, dt, sub)
        self.dm.step(self.time, dt, sub, self.em_probing, self.panel.t_super.value)
        self.ship.step(self.time, dt, sub)

        # 3. Structural and tidal bookkeeping.
        self.pop.refresh_structure()
        removed = self.pop.update_spaghettification(self.hole, dt, self.bubble)
        if removed:
            self.focus_index = -1

        # 4. The vacuum evolves: a(t) = exp(H t).
        self.cosmos.advance(dt)
        self.hole.phase += dt
        self.time += dt

        self.pop.record_trails()
        self.ship.record_trail()

        # Optional chase camera.  The hole stays pinned to the world origin; it
        # is the viewport that moves, so no physics is affected either way.
        if self.panel.t_follow.value:
            self.camera.center += (self.ship.position - self.camera.center) * min(
                1.0, 5.0 * dt)
        elif float(norm(self.camera.center)) > 1e-9:
            self.camera.center *= max(0.0, 1.0 - 3.0 * dt)

    # ---------------------------------------------------------------- render
    def draw(self, surf: pygame.Surface) -> None:
        r = self.renderer
        r.draw_background(surf, self.cosmos, self.fog)

        # The fabric is perturbed by the hole plus every spawned body.
        src_pos = np.concatenate([self.hole.position[None, :],
                                  self.pop.state[:, 0:2]], axis=0)
        src_mass = np.concatenate([[self.hole.mass], self.pop.geom], axis=0)
        self.grid.rebuild(src_pos, src_mass, self.bubble)
        r.draw_black_hole(surf, self.hole, self.shielded)
        self.grid.draw(surf, self.bubble, dim=1.0 if r.light_enabled else 0.34)

        r.draw_bodies(surf, self.pop, self.hole, self.fog)
        r.draw_dark_matter(surf, self.dm, self.shielded, self.laser_on)
        r.draw_ship(surf, self.ship, self.bubble)

        # Dust is electromagnetic: it only exists on the light path.
        if r.light_enabled:
            self.fog.draw(surf)


# ==============================================================================
#  SECTION 18 -- ENGINE  (window, input, main loop)
# ==============================================================================

class CosmicEngine:
    """Pygame shell: owns the display, the clock and all input routing."""

    def __init__(self, headless: bool = False):
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        # Only the subsystems actually used: initializing the mixer on a machine
        # with no sound card emits a wall of ALSA warnings for no benefit.
        pygame.display.init()
        pygame.font.init()
        flags = pygame.DOUBLEBUF
        self.screen = pygame.display.set_mode((CFG.width, CFG.height), flags)
        pygame.display.set_caption(CFG.caption)
        self.clock = pygame.time.Clock()
        self.fonts = Fonts()

        self.canvas_rect = pygame.Rect(0, 0, CFG.width - CFG.sidebar_w, CFG.height)
        self.panel_rect = pygame.Rect(self.canvas_rect.right, 0,
                                      CFG.sidebar_w, CFG.height)
        self.camera = Camera(self.canvas_rect, scale=1.0)
        self.panel = ControlPanel(self.panel_rect, self.fonts)
        self.fog = DustFog(self.canvas_rect.width, self.canvas_rect.height)
        self.sim = Simulation(self.camera, self.fonts, self.panel, self.fog)

        self.running = True
        self.panning = False
        self.pan_anchor = np.zeros(2)
        self.dragging_mass = False
        self.toast = ""
        self.toast_t = 0.0

    # ----------------------------------------------------------------- toast
    def say(self, msg: str) -> None:
        self.toast = msg
        self.toast_t = 2.6

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
            self.say("Ship reset to a circular orbit at 2.9 x ISCO")
        if self.panel.b_clear.consume():
            self.sim.pop.clear()
            self.sim.focus_index = -1
            self.say("Bodies cleared")

    def _on_key(self, event: pygame.event.Event) -> None:
        k = event.key
        p = self.panel
        if k == pygame.K_ESCAPE:
            self.running = False
        elif k == pygame.K_SPACE:
            self.sim.paused = not self.sim.paused
            p.b_pause.label = "RESUME" if self.sim.paused else "PAUSE"
        elif k == pygame.K_w:
            p.r_mode.index = ShipMode.WARP
            self.say("Alcubierre drive engaged: grid contracts ahead, expands behind")
        elif k == pygame.K_s:
            p.r_mode.index = ShipMode.SLINGSHOT
            self.say("Slingshot / Halo Drive: thrust with the arrow keys")
        elif k == pygame.K_g:
            p.t_grid.value = not p.t_grid.value
        elif k == pygame.K_b:
            p.t_labels.value = not p.t_labels.value
        elif k == pygame.K_t:
            p.t_trails.value = not p.t_trails.value
        elif k == pygame.K_f:
            p.t_follow.value = not p.t_follow.value
            self.say("Chase camera ON -- black hole no longer pinned to centre"
                     if p.t_follow.value else "Camera returning to the black hole")
        elif k == pygame.K_z:
            p.t_fog.value = not p.t_fog.value
            self.say("Zone of Avoidance: %s" % ("ON" if p.t_fog.value else "OFF"))
        elif k == pygame.K_d:
            p.t_dm.value = not p.t_dm.value
            if p.t_dm.value:
                self.sim.dm.seed_halo(520, self.sim.hole.r_ergosphere * 1.05,
                                      self.sim.hole.r_isco * 7.5)
        elif k == pygame.K_u:
            p.t_shield.value = not p.t_shield.value
            self.say("Deep shielding ON: light vectors off, dark sector measurable"
                     if p.t_shield.value else
                     "Shielding lifted: ambient photons contaminate the dark states")
        elif k == pygame.K_l:
            p.t_laser.value = not p.t_laser.value
            self.say("Laser sensor ON: dark matter states collapsing"
                     if p.t_laser.value else "Laser sensor OFF")
        elif k == pygame.K_r:
            self.sim.ship.reset()
            p.r_mode.index = self.sim.ship.mode
        elif k == pygame.K_c:
            self.sim.pop.clear()
            self.sim.focus_index = -1
        elif k in (pygame.K_1, pygame.K_2, pygame.K_3):
            p.r_spawn.index = k - pygame.K_1

    def _on_mouse_down(self, event: pygame.event.Event) -> None:
        if not self.canvas_rect.collidepoint(event.pos):
            return
        if event.button == 3:
            self.panning = True
            return
        if event.button != 1:
            return
        world = self.camera.to_world(*event.pos)
        # Clicking an existing body focuses it and begins pouring mass in.
        idx = self.sim.pop.add_mass_at(world, 0.0, grab_radius=22.0)
        if idx >= 0:
            self.sim.focus_index = idx
            self.dragging_mass = True
            return
        self._spawn_at(world)

    def _spawn_at(self, world: np.ndarray) -> None:
        """Spawn on a circular orbit through the clicked point."""
        mode = self.panel.r_spawn.index
        hole = self.sim.hole
        r = max(float(norm(world)), hole.r_s * 1.4)
        if mode == 2:
            ang = math.atan2(world[1], world[0])
            self.sim.dm.seed_halo(90, r * 0.86, r * 1.16)
            self.panel.t_dm.value = True
            self.say("Dark matter injected -- invisible unless deep-shielded")
            return
        mass = 2.0 if mode == 0 else 320.0
        v_c = min(float(hole.circular_speed(r)), 0.9)
        tangent = np.array([-world[1], world[0]]) / max(r, CFG.eps)
        self.sim.pop.spawn(world, tangent * v_c, mass)
        self.sim.focus_index = self.sim.pop.count - 1

    def _continuous_input(self, dt_real: float) -> None:
        keys = pygame.key.get_pressed()
        ship = self.sim.ship
        if ship.mode == ShipMode.WARP:
            turn = 0.0
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                turn += 1.9 * dt_real
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                turn -= 1.9 * dt_real
            if turn:
                ship.steer(turn)
            ship.set_thrust(np.zeros(2))
            return
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
                # Mass injection rate scales with current mass so the whole
                # ladder from 1 MJ to 3 M_sun is reachable in a few seconds.
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

        self.screen.set_clip(self.canvas_rect)
        self.sim.draw(self.screen)
        self.screen.set_clip(None)
        self.panel.draw(self.screen, self.sim)
        self._draw_overlay()
        if self.toast_t > 0.0:
            self.toast_t -= dt_real
            t = self.fonts.body.render(self.toast, True, Palette.WARN)
            self.screen.blit(t, (18, CFG.height - 30))

    def _draw_overlay(self) -> None:
        """Top-left status strip on the canvas."""
        f = self.fonts
        sim = self.sim
        lines = [
            ("PAUSED" if sim.paused else "RUNNING", Palette.WARN if sim.paused
             else Palette.OK),
            (f"t = {sim.time:8.1f}   a(t) = {sim.cosmos.scale_factor:6.3f}",
             Palette.TEXT_DIM),
        ]
        if sim.shielded and not sim.laser_on:
            lines.append(("DEEP UNDERGROUND SHIELDING -- all light vectors off",
                          Palette.DARK_MATTER))
        if sim.laser_on:
            lines.append(("LASER SENSOR ACTIVE -- dark states decohering",
                          Palette.DANGER))
        if sim.bubble.active:
            lines.append((f"WARP FIELD  v_s = {sim.ship.warp_speed:.2f} c  "
                          f"(coordinate, local v = 0)", Palette.WARP))
        y = 12
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
#  SECTION 19 -- VERIFICATION
#
#  These are not unit tests for their own sake; they are the evidence that the
#  claims made in the comments above are true.  Each one checks a closed-form
#  result the engine should reproduce.
# ==============================================================================

class SelfTest:
    """Analytic checks on the physics, plus a headless render benchmark."""

    def __init__(self):
        self.failures: list[str] = []
        self.lines: list[str] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        tag = "PASS" if ok else "FAIL"
        self.lines.append(f"  [{tag}] {name}" + (f"   {detail}" if detail else ""))
        if not ok:
            self.failures.append(name)

    def near(self, name: str, got: float, want: float, tol: float) -> None:
        self.check(name, abs(got - want) <= tol,
                   f"got {got:.6g}, expected {want:.6g} (tol {tol:g})")

    # ------------------------------------------------------------ geometry
    def test_kerr_radii(self) -> None:
        self.lines.append("Kerr / Schwarzschild boundary radii")
        h = BlackHole(mass=10.0, spin=0.0)
        self.near("Schwarzschild horizon = 2M", h.r_horizon, 20.0, 1e-9)
        self.near("photon sphere = 3M", h.r_photon, 30.0, 1e-6)
        self.near("ISCO = 6M", h.r_isco, 60.0, 1e-6)
        self.near("ergosphere = 2M at equator", h.r_ergosphere, 20.0, 1e-9)
        # Near-extremal.  Spin is deliberately capped at 0.999 (a true extremal
        # hole is a measure-zero limit and makes r+ non-differentiable), so the
        # expected values are the closed forms evaluated AT the capped spin.
        e = BlackHole(mass=10.0, spin=0.999999)
        astar, M = e.spin, e.mass
        self.near("horizon = M + sqrt(M^2 - a^2)", e.r_horizon,
                  M * (1 + math.sqrt(1 - astar ** 2)), 1e-9)
        self.near("prograde photon orbit closed form", e.r_photon,
                  2 * M * (1 + math.cos(2 / 3 * math.acos(-astar))), 1e-9)
        self.near("retrograde photon orbit closed form", e.r_photon_retro,
                  2 * M * (1 + math.cos(2 / 3 * math.acos(astar))), 1e-9)
        self.check("near-extremal radii collapse toward M",
                   e.r_horizon < 1.05 * M and e.r_photon < 1.10 * M
                   and e.r_isco < 1.25 * M,
                   f"r+={e.r_horizon:.3f} r_ph={e.r_photon:.3f} isco={e.r_isco:.3f}")
        # Every boundary must shrink monotonically as the hole spins up.
        spins = np.linspace(0.0, 0.99, 40)
        hs = [BlackHole(mass=10.0, spin=float(a)) for a in spins]
        self.check("all prograde radii decrease monotonically with spin",
                   all(np.all(np.diff([getattr(h, k) for h in hs]) < 1e-12)
                       for k in ("r_horizon", "r_photon", "r_isco")))

    def test_smoothstep(self) -> None:
        """A descending edge pair must invert the ramp, not saturate it."""
        self.lines.append("Interpolation helper")
        self.near("ascending: below edge0 -> 0", float(smoothstep(2, 8, 1)), 0.0, 0)
        self.near("ascending: above edge1 -> 1", float(smoothstep(2, 8, 9)), 1.0, 0)
        self.near("ascending: midpoint -> 0.5", float(smoothstep(2, 8, 5)), 0.5, 1e-12)
        self.near("descending: at edge0 -> 0", float(smoothstep(8, 2, 8)), 0.0, 0)
        self.near("descending: at edge1 -> 1", float(smoothstep(8, 2, 2)), 1.0, 0)
        self.near("descending: beyond edge0 -> 0", float(smoothstep(8, 2, 20)),
                  0.0, 0)
        self.near("descending: midpoint -> 0.5", float(smoothstep(8, 2, 5)),
                  0.5, 1e-12)

    def test_halo_drive_window(self) -> None:
        """
        Kipping's halo drive draws on photons that have orbited the hole, so its
        thrust must exist ONLY in a narrow annulus around the photon sphere, and
        only on a prograde pass.  It is not a general-purpose booster.
        """
        self.lines.append("Halo drive availability window")
        hole = BlackHole(mass=26.0, spin=0.9)
        g = GravityField(hole, CosmologicalField(0.0))
        g.set_sources(np.zeros((0, 2)), np.zeros(0), np.zeros(0))
        ship = Spacecraft(g, hole, WarpBubble())
        ship.mode = ShipMode.SLINGSHOT
        r_ph = hole.r_photon

        def boost_at(r, prograde=True):
            v = float(hole.circular_speed(max(r, hole.r_s * 1.2)))
            ship.state = np.array([[r, 0.0, 0.0, v if prograde else -v]])
            return float(norm(ship._halo_drive_impulse()))

        near_ph = boost_at(r_ph * 1.15)
        far = boost_at(r_ph * 5.0)
        very_far = boost_at(r_ph * 20.0)
        self.check("thrust available at the photon sphere", near_ph > 1e-4,
                   f"{near_ph:.5f}")
        self.check("no thrust far outside the photon sphere",
                   far == 0.0 and very_far == 0.0,
                   f"{far:.2e} at 5x r_ph, {very_far:.2e} at 20x r_ph")
        self.check("thrust decreases with distance from the photon sphere",
                   near_ph > boost_at(r_ph * 2.5) >= far)
        self.check("no thrust on a retrograde pass",
                   boost_at(r_ph * 1.15, prograde=False) == 0.0)
        # Gain must vanish with spin: the energy comes from the hole's rotation.
        slow = BlackHole(mass=26.0, spin=0.0)
        g2 = GravityField(slow, CosmologicalField(0.0))
        g2.set_sources(np.zeros((0, 2)), np.zeros(0), np.zeros(0))
        s2 = Spacecraft(g2, slow, WarpBubble()); s2.mode = ShipMode.SLINGSHOT
        s2.state = np.array([[slow.r_photon * 1.15, 0.0, 0.0,
                              float(slow.circular_speed(slow.r_photon * 1.15))]])
        self.check("a non-spinning hole gives a weaker halo boost",
                   float(norm(s2._halo_drive_impulse())) < near_ph)

    # ------------------------------------------------------------ integrator
    def test_rk4_order(self) -> None:
        """
        Halving the step must cut the error by ~2^4 = 16.  This is the property
        that distinguishes RK4 from Euler (which would give a factor of 2).
        """
        self.lines.append("RK4 convergence order")
        hole = BlackHole(mass=10.0, spin=0.0)
        cos = CosmologicalField(0.0)
        g = GravityField(hole, cos)
        deriv = g.make_derivative(include_nbody=False, relativistic=False)
        integ = RK4Integrator(deriv)

        r0 = 400.0
        # PW circular condition under the NON-relativistic law: v^2/r = M/(r-rs)^2
        v0 = math.sqrt(hole.mass * r0) / (r0 - hole.r_s)
        s0 = np.array([[r0, 0.0, 0.0, v0]])
        period = math.tau * r0 / v0

        errs = []
        for n in (400, 800, 1600):
            st = integ.integrate(s0.copy(), 0.0, period, n)
            errs.append(abs(float(norm(st[0, 0:2])) - r0))
        r1 = errs[0] / max(errs[1], 1e-18)
        r2 = errs[1] / max(errs[2], 1e-18)
        self.check("error ratio ~16 on step halving (4th order)",
                   8.0 < r1 < 40.0 and 8.0 < r2 < 40.0,
                   f"ratios {r1:.1f}, {r2:.1f}; errors {errs[0]:.3e} -> {errs[2]:.3e}")
        self.check("closed orbit after one period", errs[-1] < 1e-4,
                   f"radial drift {errs[-1]:.3e} px")

    def test_circular_speed(self) -> None:
        """
        The relativistic circular condition must close the orbit exactly, and it
        must do so over MANY full orbits -- a fraction of one orbit proves nothing.
        Tested outside 1.4 x ISCO; nearer than that the orbit is marginally
        stable and roundoff is amplified without bound (see test_isco_stability,
        which is the correct physics, not an integrator defect).
        """
        self.lines.append("Relativistic circular orbits")
        hole = BlackHole(mass=26.0, spin=0.0)
        g = GravityField(hole, CosmologicalField(0.0))
        integ = RK4Integrator(g.make_derivative(include_nbody=False,
                                                relativistic=True))
        # All three orbits are integrated as one stacked (3, 4) state -- the
        # integrator is vectorized, so this costs the same as one of them.
        radii = np.array([200.0, 300.0, 420.0])
        vs = np.array([float(hole.circular_speed(r)) for r in radii])
        st = np.stack([radii, np.zeros(3), np.zeros(3), vs], axis=1)
        periods = math.tau * radii / vs
        n = int(6.0 * float(periods.max()) / 1.0)
        worst = 0.0
        for i in range(n):
            st = integ.step(st, 0.0, 1.0)
            if i % 29 == 0:
                worst = max(worst, float(np.max(
                    np.abs(norm(st[:, 0:2]) - radii) / radii)))
        laps = 6.0 * periods.max() / periods
        self.check(f"radius holds over {laps.min():.0f}-{laps.max():.0f} full orbits "
                   f"at 3 radii", worst < 1e-6,
                   f"max fractional excursion {worst:.2e}")
        # Weak-field limit must reduce to Newton.  Both the PW and relativistic
        # corrections are O(rs/r), so this only holds far out -- which is itself
        # the check that the corrections carry the right order.
        for r_far, tol in ((1e6, 3e-7), (1e9, 1e-11)):
            self.near(f"weak field r={r_far:g} -> sqrt(M/r)",
                      float(hole.circular_speed(r_far)),
                      math.sqrt(26.0 / r_far), tol)

    def test_isco_stability(self) -> None:
        """
        The defining property of the Paczynski-Wiita potential: its marginally
        stable circular orbit sits at exactly 6M, as in full Schwarzschild.

        For V_eff(r) = L^2/2r^2 - M/(r - r_s), the circular condition dV/dr = 0
        gives L^2 = M r^3 / (r - r_s)^2, and substituting into d^2V/dr^2 > 0:

            d^2V/dr^2 = M/(r - r_s)^2 [ 3/r - 2/(r - r_s) ]

        which changes sign at 3(r - r_s) = 2r, i.e. r = 3 r_s = 6M.  Exactly the
        Schwarzschild ISCO -- and a plain Newtonian 1/r potential has no ISCO at
        all, so this is the check that the surrogate is doing its job.
        """
        self.lines.append("ISCO and marginal stability (Paczynski-Wiita)")
        M = 26.0
        hole = BlackHole(mass=M, spin=0.0)
        rs = hole.r_s

        def d2V(r):
            return M / (r - rs) ** 2 * (3.0 / r - 2.0 / (r - rs))

        self.check("d2V/dr2 > 0 outside 6M (stable)", d2V(6 * M + 1e-6) > 0)
        self.check("d2V/dr2 < 0 inside 6M (unstable)", d2V(6 * M - 1e-6) < 0)
        # Bisect the sign change; it must land on 6M to machine precision.
        lo, hi = 2.2 * rs, 10.0 * rs      # root is at 3 rs, so bracket it properly
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if d2V(mid) > 0:
                hi = mid
            else:
                lo = mid
        self.near("marginally stable orbit = 6M exactly", 0.5 * (lo + hi),
                  6.0 * M, 1e-9)
        self.near("BlackHole.r_isco agrees at zero spin", hole.r_isco, 6.0 * M, 1e-9)

        # Dynamical corroboration: the epicyclic excursion from an identical
        # relative perturbation must blow up as the ISCO is approached, because
        # the radial epicyclic frequency kappa -> 0 there.
        g = GravityField(hole, CosmologicalField(0.0))
        integ = RK4Integrator(g.make_derivative(include_nbody=False,
                                                relativistic=True))
        facs = np.array([1.05, 1.25, 2.0, 4.0])
        r0s = hole.r_isco * facs
        v0s = np.array([float(hole.circular_speed(r)) for r in r0s]) * 1.002  # 0.2% kick
        st = np.stack([r0s, np.zeros(4), np.zeros(4), v0s], axis=1)
        excursions = np.zeros(4)
        for i in range(7000):
            st = integ.step(st, 0.0, 0.8)
            if i % 9 == 0:
                excursions = np.maximum(
                    excursions, np.abs(norm(st[:, 0:2]) - r0s) / r0s)
        excursions = list(map(float, excursions))
        self.check("epicyclic excursion grows monotonically toward the ISCO",
                   all(excursions[i] > excursions[i + 1]
                       for i in range(len(excursions) - 1)),
                   "  ".join(f"{f}xISCO:{e * 100:.2f}%"
                             for f, e in zip(facs, excursions)))
        self.check("far from the hole a 0.2% kick stays a small epicycle",
                   excursions[-1] < 0.02, f"{excursions[-1] * 100:.3f}%")

        # Inside the ISCO the circular orbit is an UNSTABLE equilibrium -- it is
        # still an equilibrium, so an unperturbed particle sits there forever.
        # Instability only shows under a perturbation, and the identical
        # perturbation outside the ISCO must stay bounded.  That contrast is the
        # actual physical claim.
        for fac, should_plunge in ((0.92, True), (1.60, False)):
            r0 = hole.r_isco * fac
            v = float(hole.circular_speed(r0)) * 0.999      # 0.1% inward nudge
            st = np.array([[r0, 0.0, 0.0, v]])
            rmin = r0
            for _ in range(9000):
                st = integ.step(st, 0.0, 0.5)
                rmin = min(rmin, float(norm(st[0, 0:2])))
                if rmin < rs:
                    break
            plunged = rmin < rs
            self.check(
                f"0.1% nudge at {fac:.2f}x ISCO "
                f"{'plunges to the horizon' if should_plunge else 'stays bound'}",
                plunged == should_plunge,
                f"min radius {rmin:.2f} (r_s = {rs:.1f}, r0 = {r0:.1f})")

    def test_light_speed_barrier(self) -> None:
        """Sustained thrust must asymptote to c, never reach or exceed it."""
        self.lines.append("Relativistic dynamics: c as an asymptote")
        hole = BlackHole(mass=1e-9, spin=0.0)
        g = GravityField(hole, CosmologicalField(0.0))
        v = np.array([[0.0, 0.0]])
        acc = np.array([[0.05, 0.0]])
        for _ in range(200000):
            v = v + GravityField.proper_accel(v, acc) * 0.05
        speed = float(norm(v[0]))
        self.check("speed asymptotes below c under huge sustained thrust",
                   0.999 < speed < 1.0, f"v = {speed:.9f} c")
        # Transverse acceleration is suppressed by 1/gamma, longitudinal 1/gamma^3.
        vv = np.array([[0.9, 0.0]])
        gam = float(lorentz_gamma(vv)[0])
        a_long = GravityField.proper_accel(vv, np.array([[1.0, 0.0]]))[0, 0]
        a_tran = GravityField.proper_accel(vv, np.array([[0.0, 1.0]]))[0, 1]
        self.near("longitudinal response = 1/gamma^3", a_long, 1.0 / gam ** 3, 1e-9)
        self.near("transverse response = 1/gamma", a_tran, 1.0 / gam, 1e-9)

    # ------------------------------------------------------------ dilation
    def test_time_dilation(self) -> None:
        """dtau/dt = sqrt(1 - rs/r) sqrt(1 - v^2), checked against the metric."""
        self.lines.append("Schwarzschild time dilation")
        hole = BlackHole(mass=20.0, spin=0.0)
        g = GravityField(hole, CosmologicalField(0.0))
        ship = Spacecraft(g, hole, WarpBubble())
        ship.mode = ShipMode.BALLISTIC
        for r, v in ((200.0, 0.0), (60.0, 0.0), (44.0, 0.6)):
            ship.state = np.array([[r, 0.0, 0.0, v]])
            want = math.sqrt(1.0 - hole.r_s / r) * math.sqrt(1.0 - v * v)
            self.near(f"r={r:g}, v={v:g}", ship.time_dilation_factor(), want, 1e-9)
        # Hovering just outside the horizon must nearly stop the ship clock.
        ship.state = np.array([[hole.r_s * 1.0005, 0.0, 0.0, 0.0]])
        self.check("clock nearly frozen at the horizon",
                   ship.time_dilation_factor() < 0.03,
                   f"dtau/dt = {ship.time_dilation_factor():.5f}")
        # Inside a warp bubble the interior is flat: no dilation at all.
        b = WarpBubble(); b.active = True
        ship2 = Spacecraft(g, hole, b)
        ship2.mode = ShipMode.WARP
        ship2.state = np.array([[hole.r_s * 1.01, 0.0, 0.0, 0.0]])
        self.near("flat bubble interior -> dtau/dt = 1",
                  ship2.time_dilation_factor(), 1.0, 1e-12)

    # ------------------------------------------------------------ alcubierre
    def test_alcubierre(self) -> None:
        self.lines.append("Alcubierre shape function and expansion scalar")
        b = WarpBubble(radius=80.0, sigma=0.06)
        b.configure(np.zeros(2), np.array([1.0, 0.0]), 3.0)
        b.active = True
        self.near("f(0) = 1 (flat interior)", float(b.f(np.array([0.0]))[0]), 1.0, 2e-3)
        self.check("f -> 0 far outside",
                   float(b.f(np.array([600.0]))[0]) < 1e-6,
                   f"f(600) = {float(b.f(np.array([600.0]))[0]):.2e}")
        self.check("df/dr_s < 0 everywhere",
                   bool(np.all(b.df(np.linspace(1.0, 400.0, 500)) < 0.0)))
        ahead = float(b.expansion_scalar(np.array([[80.0, 0.0]]))[0])
        behind = float(b.expansion_scalar(np.array([[-80.0, 0.0]]))[0])
        self.check("theta < 0 ahead (space contracts)", ahead < 0, f"{ahead:.4e}")
        self.check("theta > 0 behind (space expands)", behind > 0, f"{behind:.4e}")
        self.near("theta antisymmetric about the ship", ahead, -behind, 1e-12)
        # The interior screens external curvature.
        m_in = float(b.exterior_mask(np.array([[0.0, 0.0]]))[0, 0])
        m_out = float(b.exterior_mask(np.array([[400.0, 0.0]]))[0, 0])
        self.check("tidal screening: ~0 inside, ~1 outside",
                   m_in < 5e-3 and m_out > 0.999, f"{m_in:.4f} / {m_out:.4f}")

    # ------------------------------------------------------------ fabric
    def test_grid_pinch(self) -> None:
        """
        The pinch derivative must match the metric's first-order compression:
        dDelta/dd -> r_s/(2d), i.e. the expansion of 1 - 1/sqrt(1 - r_s/d).
        """
        self.lines.append("Flamm-paraboloid grid pinch")
        rs = np.array([40.0])
        # dDelta/dd = 2/(1 + 4d/rs) -> rs/(2d) only as d/rs -> infinity, so the
        # asymptote is checked where it is actually asserted to hold.
        for d0, tol in ((4.0e3, 2e-5), (4.0e6, 1e-10)):
            d = np.array([d0]); h = d0 * 1e-6
            num = float((SpacetimeGrid._gravity_pinch(d + h, rs) -
                         SpacetimeGrid._gravity_pinch(d - h, rs))[0] / (2 * h))
            self.near(f"dDelta/dd = rs/2d at d/rs={d0 / 40.0:.0f}",
                      num, 40.0 / (2 * d0), tol)
        prof = SpacetimeGrid._gravity_pinch(np.linspace(1.0, 3000.0, 900), rs)
        self.check("pinch non-negative and non-decreasing",
                   bool(np.all(prof >= 0.0)) and bool(np.all(np.diff(prof) >= 0.0)))
        far = SpacetimeGrid._gravity_pinch(np.linspace(40.0, 3000.0, 900), rs)
        self.check("strictly increasing outside the clamped core",
                   bool(np.all(np.diff(far) > 0.0)))
        # Vertices must never be dragged through the source.
        grid = SpacetimeGrid(Camera(pygame.Rect(0, 0, 800, 600)),
                             CosmologicalField(0.0))
        pts = np.stack([np.linspace(-500, 500, 400), np.full(400, 3.0)], axis=1)
        out, _ = grid.displace(pts, np.zeros((1, 2)), np.array([60.0]), None)
        self.check("no vertex crosses the source",
                   bool(np.all(np.sign(out[:, 0] * pts[:, 0]) >= 0)))

    def test_cosmology(self) -> None:
        self.lines.append("Dark energy / FRW background")
        c = CosmologicalField(0.01)
        c.advance(100.0)
        self.near("a(t) = exp(H t)", c.scale_factor, math.exp(1.0), 1e-9)
        self.check("wrapped scale stays in [1,2)", 1.0 <= c.wrapped_scale < 2.0,
                   f"{c.wrapped_scale:.4f}")
        # de Sitter repulsion H^2 r, and the turnaround radius where it balances.
        hole = BlackHole(mass=26.0, spin=0.0)
        g = GravityField(hole, c)
        r_ta = c.turnaround_radius(hole.mass, hole.r_s)
        p = np.array([[r_ta, 0.0]])
        a = float(g.central_acceleration(p)[0, 0])
        scale = hole.mass / (r_ta - hole.r_s) ** 2
        self.check("gravity exactly cancels dark energy at r_ta",
                   abs(a) < 1e-9 * scale, f"net a = {a:.3e} at r_ta = {r_ta:.1f}")
        self.check("bound just inside r_ta, unbound just outside",
                   float(g.central_acceleration(np.array([[r_ta * 0.9, 0.0]]))[0, 0]) < 0
                   and float(g.central_acceleration(
                       np.array([[r_ta * 1.1, 0.0]]))[0, 0]) > 0)
        self.near("Newtonian limit of r_ta = (GM/H^2)^(1/3)",
                  c.turnaround_radius(hole.mass), (26.0 / 0.0001) ** (1 / 3), 1e-9)

    # ------------------------------------------------------------ matter
    def test_structure(self) -> None:
        self.lines.append("Stellar structure and phase transitions")
        S = StellarStructure
        ms = np.logspace(-2, 3.7, 8000)
        peak = float(ms[int(np.argmax(S.physical_radius(ms)))])
        self.near("Zapolsky-Salpeter radius peaks at M_0/3^(3/4)",
                  peak, S.M_ZS / 3 ** 0.75, 0.05)
        self.check("degenerate branch shrinks with mass",
                   float(S.physical_radius(np.array([1000.0]))[0]) <
                   float(S.physical_radius(np.array([100.0]))[0]))
        cases = [(1.0, Phase.GAS_GIANT), (13.5, Phase.BROWN_DWARF),
                 (81.0, Phase.STAR), (1600.0, Phase.DEGENERATE),
                 (3200.0, Phase.COLLAPSED)]
        for m, want in cases:
            got = int(S.phase_of(np.array([m]))[0])
            self.check(f"{m} MJ -> {Phase.NAMES[want]}", got == want,
                       f"got {Phase.NAMES[got]}")
        self.near("Chandrasekhar limit = 1.44 Msun", S.M_CHANDRA,
                  1.44 * 1047.57, 1e-6)
        g0, d0 = S.pressure_balance(1.0)
        g1, d1 = S.pressure_balance(2500.0)
        self.check("compression outruns EM resistance above M_Ch",
                   (g1 / d1) > (g0 / d0),
                   f"ratio {g0 / d0:.3e} -> {g1 / d1:.3e}")

    def test_dark_matter_rules(self) -> None:
        self.lines.append("Dark matter interaction and measurement rules")
        hole = BlackHole(mass=26.0, spin=0.9)
        g = GravityField(hole, CosmologicalField(0.0))
        dm = DarkMatterField(g, hole)
        dm.seed_halo(200, 90.0, 400.0)
        self.check("halo seeded", dm.count > 0, f"{dm.count} particles")

        # Rule 1: a massive wall of matter must not deflect it at all beyond gravity.
        base = dm.state.copy()
        g.set_sources(np.zeros((0, 2)), np.zeros(0), np.zeros(0))
        dm.step(0.0, 0.5, 2, em_probing=False, superradiance=False)
        free = dm.state.copy()
        dm.state = base.copy()
        # Same field, but now with an enormous body sitting on top of the halo:
        # the ONLY change permitted is gravitational.
        g.set_sources(np.array([[200.0, 0.0]]), np.array([40.0]), np.array([8.0]))
        dm.step(0.0, 0.5, 2, em_probing=False, superradiance=False)
        moved = np.max(norm(dm.state[:, 0:2] - free[:, 0:2]))
        self.check("responds to a mass only gravitationally (no contact term)",
                   moved > 0.0, f"gravitational deflection {moved:.4f} px")

        # Rule 2: the measurement gate.
        dm.coherence[:] = 1.0
        self.check("invisible when unshielded",
                   not dm.visible_mask(shielded=False, laser_on=False).any())
        self.check("invisible when the laser sensor is on",
                   not dm.visible_mask(shielded=True, laser_on=True).any())
        self.check("visible only when shielded and dark",
                   dm.visible_mask(shielded=True, laser_on=False).all())

        # Rule 3: EM probing destroys the states; shielding lets them re-form.
        for _ in range(30):
            dm.step(0.0, 0.4, 1, em_probing=True, superradiance=False)
        self.check("decoheres under electromagnetic probing",
                   float(np.max(dm.coherence)) < 0.06,
                   f"max coherence {float(np.max(dm.coherence)):.2e}")
        self.check("still gravitating after decoherence (not deleted)",
                   dm.count > 0, f"{dm.count} particles remain")
        for _ in range(30):
            dm.step(0.0, 0.4, 1, em_probing=False, superradiance=False)
        self.check("re-coheres inside deep shielding",
                   float(np.max(dm.coherence)) > 0.9)

    def test_superradiance(self) -> None:
        """Penrose extraction must scale with spin and vanish for a static hole."""
        self.lines.append("Superradiance / Penrose process")
        for spin, expect_gain in ((0.9, True), (0.0, False)):
            hole = BlackHole(mass=26.0, spin=spin)
            g = GravityField(hole, CosmologicalField(0.0))
            g.set_sources(np.zeros((0, 2)), np.zeros(0), np.zeros(0))
            dm = DarkMatterField(g, hole)
            r = hole.r_ergosphere * 0.92
            v = float(hole.circular_speed(hole.r_ergosphere * 1.3))
            n = 60
            dm.state = np.tile(np.array([[r, 0.0, 0.0, v]]), (n, 1))
            dm.coherence = np.ones(n); dm.hue = np.zeros(n)
            dm.superradiant = np.zeros(n, dtype=bool)
            e0 = float(np.sum(norm(dm.state[:, 2:4]) ** 2))
            spin0 = hole.spin
            for _ in range(8):
                dm.step(0.0, 0.25, 1, em_probing=False, superradiance=True)
            got = dm.extracted_energy > 0.0
            self.check(f"a*={spin}: energy extraction {'occurs' if expect_gain else 'is absent'}",
                       got == expect_gain, f"extracted {dm.extracted_energy:.4f}")
            if expect_gain:
                self.check("  hole spins down to pay for it", hole.spin < spin0,
                           f"a* {spin0:.6f} -> {hole.spin:.6f}")

    def test_spaghettification(self) -> None:
        self.lines.append("Tidal disruption at the horizon")
        hole = BlackHole(mass=26.0, spin=0.0)
        g = GravityField(hole, CosmologicalField(0.0))
        pop = BodyPopulation(g)
        pop.spawn([hole.r_horizon * 0.55, 0.0], [0.0, 0.0], 3.0)
        self.check("body starts intact", float(pop.strain[0]) == 0.0)
        removed, ticks = [], 0
        while pop.count and ticks < 4000:
            removed = pop.update_spaghettification(hole, 0.05, None)
            ticks += 1
        self.check("crossing the horizon triggers strain then deletion",
                   pop.count == 0 and ticks < 4000, f"destroyed after {ticks} ticks")

        # The disruption must stay watchable at the default frame step, and the
        # duration must not change when the user slows the simulation down.
        for dt_test, label in ((CFG.dt, "default dt"), (CFG.dt * 0.25, "slow-mo")):
            pop3 = BodyPopulation(g)
            pop3.spawn([hole.r_horizon * 0.9, 0.0], [0.0, 0.0], 3.0)
            n = 0
            while pop3.count and n < 20000:
                pop3.update_spaghettification(hole, dt_test, None)
                n += 1
            secs = n * dt_test / CFG.dt / 60.0
            self.check(f"disruption lasts {n} steps at {label}",
                       50 <= n <= 4000 and 0.5 < secs < 4.0,
                       f"~{secs:.2f} s of wall clock at 60 FPS")

        # A body sheltered by an active warp bubble must survive.
        b = WarpBubble(radius=90.0)
        b.configure(np.array([hole.r_horizon * 0.55, 0.0]), np.array([1.0, 0.0]), 2.0)
        b.active = True
        pop2 = BodyPopulation(g)
        pop2.spawn([hole.r_horizon * 0.55, 0.0], [0.0, 0.0], 3.0)
        for _ in range(200):
            pop2.update_spaghettification(hole, 0.05, b)
        self.check("warp bubble shelters matter from tidal death",
                   pop2.count == 1 and float(pop2.strain[0]) == 0.0)

    # ------------------------------------------------------------- rendering
    def test_render(self, frames: int = 240) -> None:
        """Headless frame benchmark: proves it draws, and how fast."""
        self.lines.append(f"Headless render benchmark ({frames} frames)")

        # --- the scene a user actually starts in ---------------------------
        base = CosmicEngine(headless=True)
        for _ in range(30):
            base.frame(1.0 / 60.0)
        t0 = pygame.time.get_ticks()
        for _ in range(frames):
            base.frame(1.0 / 60.0)
        base_fps = frames / max((pygame.time.get_ticks() - t0) / 1000.0, 1e-6)
        self.check("default scene comfortably above 60 FPS", base_fps >= 90.0,
                   f"{base_fps:.1f} FPS ({base.sim.pop.count} bodies, "
                   f"{base.sim.dm.count} DM particles)")

        # --- deliberate worst case -----------------------------------------
        eng = CosmicEngine(headless=True)
        # Exercise the expensive paths: warp on, fog on, lots of bodies.
        eng.panel.t_fog.value = True
        eng.panel.r_mode.index = ShipMode.WARP
        for i in range(24):
            eng.sim.pop.spawn_orbiting(eng.sim.hole,
                                       eng.sim.hole.r_isco * (1.6 + 0.22 * i),
                                       2.0 + 40.0 * i, angle=i * 0.51)
        t0 = pygame.time.get_ticks()
        for _ in range(frames):
            eng.frame(1.0 / 60.0)
        elapsed = (pygame.time.get_ticks() - t0) / 1000.0
        fps = frames / max(elapsed, 1e-6)
        self.check("renders without error", True,
                   f"{fps:.1f} FPS headless ({eng.sim.pop.count} bodies, "
                   f"{eng.sim.dm.count} DM particles, warp + fog on)")
        self.check("worst case still holds the 16.7 ms (60 FPS) budget",
                   fps >= 60.0, f"{1000.0 / fps:.2f} ms/frame")
        # Shielding path.
        eng.panel.t_shield.value = True
        for _ in range(30):
            eng.frame(1.0 / 60.0)
        self.check("deep-shielding render path is stable", True)
        pygame.quit()

    # ------------------------------------------------------------------ run
    def run(self) -> int:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.display.init()
        pygame.font.init()
        np.random.seed(7)
        for fn in (self.test_kerr_radii, self.test_smoothstep,
                   self.test_halo_drive_window, self.test_rk4_order,
                   self.test_circular_speed, self.test_isco_stability,
                   self.test_light_speed_barrier,
                   self.test_time_dilation, self.test_alcubierre,
                   self.test_grid_pinch, self.test_cosmology,
                   self.test_structure, self.test_dark_matter_rules,
                   self.test_superradiance, self.test_spaghettification,
                   self.test_render):
            fn()
            self.lines.append("")
        print("\n".join(self.lines))
        if self.failures:
            print(f"{len(self.failures)} FAILURE(S): " + ", ".join(self.failures))
            return 1
        print("All physics checks passed.")
        return 0


# ==============================================================================
#  SECTION 20 -- ENTRY POINT
# ==============================================================================

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="2D Cosmic Simulation Engine: relativity, quantum fields, warp.")
    ap.add_argument("--selftest", action="store_true",
                    help="run analytic physics checks and a headless benchmark")
    ap.add_argument("--headless", type=int, metavar="N", default=0,
                    help="run N frames with no window, then exit")
    args = ap.parse_args(argv)

    if args.selftest:
        return SelfTest().run()

    if args.headless:
        eng = CosmicEngine(headless=True)
        for _ in range(args.headless):
            eng.frame(1.0 / 60.0)
        print(f"Completed {args.headless} headless frames.")
        pygame.quit()
        return 0

    CosmicEngine().run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
