# Cosmic Simulation Engine

A 2D relativistic spacetime sandbox in Python + Pygame + NumPy. Einstein's gravity,
a quintessence-driven vacuum, dark matter, tidal disruption and two kinds of
faster-than-you'd-like propulsion, all in one interactive 60 FPS canvas.

Everything on screen is driven by a stated equation. Where an exact result is
impractical in 2D real time, the code uses a **named** physical approximation and
says so in the comment next to it.

```bash
pip install pygame numpy
python cosmic_engine.py              # run the sandbox
python cosmic_engine.py --selftest   # 88 analytic physics checks + benchmark
python cosmic_engine.py --headless 600
```

## Unit system

Geometrized units, `G = c = 1`. One world unit is one pixel at zoom 1, mass is
quoted directly in length units (`M` is literally `GM/c²`), and velocity is in
units of `c` — so `v = 1.0` **is** the speed of light.

The payoff is that the textbook radii fall out for free and land at readable
sizes. With the default `M = 26 px`:

| Boundary | Formula | Default |
|---|---|---|
| Event horizon | `r₊ = M + √(M² − a²)` | 40 px |
| Ergosphere (equatorial) | `r_E = 2M` | 52 px |
| Photon sphere (prograde) | `2M{1 + cos(⅔ arccos(−a/M))}` | 44 px |
| ISCO | Bardeen–Press–Teukolsky | 68 px |

A circular orbit at the ISCO runs at `0.41 c`. The unit system was chosen so that
"looks right" and "is right" coincide.

## The physics

**Integration.** Classical RK4 only — Euler is never used. Global error `O(h⁴)`;
the self-test verifies the error drops by ~16× per step halving, and that a
circular orbit holds its radius to `1e-11` over 21 full orbits. The sub-step
*size* is bounded rather than the sub-step *count*, so accuracy stays flat as you
drag the `dt` slider.

**Force law.** One acceleration function serves every population, so momentum
bookkeeping stays consistent between subsystems:

- **Paczyński–Wiita** pseudo-Newtonian potential `Φ = −GM/(r − r_s)`. Not a fudge:
  it reproduces the Schwarzschild ISCO at exactly `6M` and the marginally bound
  orbit at `4M`, and diverges at the horizon so nothing can hover. The self-test
  bisects `d²V/dr²` and confirms the sign change sits on `6M` to 1e-9.
- **Lense–Thirring frame dragging** from weak-field gravitomagnetism,
  `a = (2J/r³)(v_y, −v_x)`. Infalling matter is swept prograde: the ergosphere whirlpool.
- **de Sitter vacuum repulsion** `a = H²r`, exact for a Λ-dominated universe.
- **Plummer-softened N-body** between spawned bodies, as a vectorized pair tensor.

**Relativistic dynamics.** The engine integrates `dp/dt = F` with `p = γmv`, whose
velocity form is `dv/dt = (1/γ)[a − (v·a)v]`. The subtracted longitudinal term is
what makes `c` an asymptote — the self-test confirms the longitudinal response is
`1/γ³` and the transverse `1/γ`. There is no speed clamp anywhere.

**The living fabric.** The grid is a *comoving* lattice under three effects:

1. *Dark energy* — pitch scales by the FRW factor `a(t) = exp(Ht)`, folded into
   octaves so it can stream outward forever at constant cost.
2. *Gravity* — on Flamm's paraboloid, proper radial distance `dl = dr/√(1 − r_s/r)`
   exceeds the coordinate difference, so a lattice of equal *proper* spacing needs
   each vertex pulled inward by `Δ(d) = (r_s/2)[ln(1 + 4d/r_s) − 1]`. Its derivative
   `→ r_s/2d` reproduces the metric's first-order compression exactly. Lines crowd
   near a mass because the metric says they must.
3. *Warp* — the Alcubierre shift vector (below).

**Atomic tug-of-war.** Gravitational compression `P ∝ M²/R⁴` against electron
degeneracy `P ∝ ρ^(5/3)`, interpolated by the Zapolsky–Salpeter relation
`R(M) = C·M^(1/3)/[1 + (M/M₀)^(4/3)]`. Radius peaks at `M₀/3^(3/4) ≈ 3.5 M_J`, then
**shrinks** as you keep feeding it — Jupiter, Saturn and a 50 M_J brown dwarf really
are about the same size, and this formula is why. Above the Chandrasekhar mass the
electron gas turns relativistic, the polytropic index softens to 4/3, and the
argument is lost:

| Threshold | Physics | Becomes |
|---|---|---|
| 13 M_J | deuterium ignition | Brown Dwarf |
| 80 M_J | sustained p–p hydrogen | Star |
| 1508 M_J (1.44 M☉) | Chandrasekhar limit | Degenerate Core |
| 3143 M_J (3.0 M☉) | neutron degeneracy fails | Black Hole |

**Spaghettification.** Geodesic deviation gives `+2GM·δr/r³` radially and
`−GM·δr/r³` transversely. The drawn ellipse stretches by `k` and shrinks by `1/√k`
— the 2:−1 ratio the metric predicts, preserving area. Matter inside an active warp
bubble is exempt, because the bubble interior is flat.

**Dark matter.** The interaction rule is enforced *structurally*: this population is
stepped through `GravityField` and nothing else. There is no collision code, no
pressure term and no EM coupling anywhere in the class — it passes through planets,
stars and walls because there is no code path by which it could not. Inside the
ergosphere, prograde particles are amplified by the Penrose process, bounded by the
horizon angular velocity `Ω_H = a/(2Mr₊)`, and the hole is spun down to pay for it.

*The observer effect:* these are fragile coherent states. An active laser sensor —
or simply operating outside a deep-shielded laboratory — collapses them. A decohered
particle is **not deleted**: it still gravitates, it is merely unmeasurable. To see
dark matter you must enable **Deep Underground Shielding**, which cuts every
light-rendering vector: the accretion disk, the starfield and the dust all go dark,
and bodies become bare gravitational outlines.

**Propulsion.**

- *Slingshot / Halo Drive.* A prograde ergosphere pass rides the dragged frame and
  leaves with more energy — the Penrose process, already in the force law, no
  special code. Kipping's halo drive adds thrust from photons that have orbited the
  hole and returned blueshifted; the self-test confirms it is available **only** in
  a narrow annulus around the photon sphere, **only** on a prograde pass, and that
  its gain vanishes with spin. The hull shows Lorentz contraction `L = L₀/γ`.
- *Alcubierre warp.* Metric `ds² = −dt² + (dx − v_s f(r_s)dt)² + dy²` with the top-hat
  shape function. Inside, `f = 1` identically, its gradient vanishes, and the region
  is Riemann-flat: zero proper acceleration, zero tidal force, clock running at the
  distant rate. The expansion scalar `θ = v_s(x_s/r_s)·df/dr_s` is negative ahead and
  positive behind — the grid is coloured by that exact quantity, not by an artistic
  impression. Coordinate speed is unbounded by `c` while local speed is identically
  zero.

**Clocks.** `dτ = dt·√(1 − r_s/r)·√(1 − v²)` — gravitational lapse times Lorentz
factor, both computed every frame. Hover near the horizon and the ship clock
visibly stalls against the deep-space clock.

**Zone of Avoidance.** Beer–Lambert extinction `I = I₀e^(−τ)` through a Gaussian
column density about the galactic mid-plane, textured with 5 octaves of value
noise. Being electromagnetic, it never touches dark matter.

## Controls

| | |
|---|---|
| Left click canvas | spawn the selected class |
| Left drag on a body | pour in mass — watch the phase transitions |
| Right drag / wheel | pan / zoom |
| `W` / `S` | Alcubierre warp / slingshot mode |
| Arrows | thrust (slingshot), steer (warp) |
| `D` `L` `U` | dark matter halo / laser sensor / deep shielding |
| `Z` `T` `G` `B` `F` | dust fog / trails / grid / labels / chase camera |
| `1` `2` `3` | spawn planet / star / dark matter |
| `Space` `R` `C` `Esc` | pause / reset ship / clear bodies / quit |

Sliders cover black hole mass and spin, dark energy expansion rate `H`, and the
time step (down to zero for frame-by-frame analysis).

## Performance

All tensor work is vectorized through NumPy; populations are stored as
structure-of-arrays so one RK4 step costs four vectorized derivative evaluations
regardless of particle count. Measured headless on the CI container:

- **112 FPS** default scene (4 bodies, 512 dark matter particles)
- **62 FPS** deliberate worst case (28 bodies, 514 DM particles, warp bubble and
  dust fog both active)

## Verification

`--selftest` runs 88 checks, each one a closed-form result the engine should
reproduce — Kerr radii against their analytic forms, RK4 convergence order, the
`1/γ³` vs `1/γ` acceleration split, time dilation against the metric, the ISCO
bisected out of the effective potential, epicyclic amplification approaching it,
the dark matter measurement rules, Penrose extraction scaling with spin — plus the
render benchmark above.
