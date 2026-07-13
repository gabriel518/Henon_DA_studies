# %%
# 02_phase_space_and_fma.py — tunes and frequency map analysis (FMA)
# ==================================================================
#
# The sextupole makes the oscillation frequency (the "tune") depend on the
# amplitude. Two questions follow:
#
#  1. WHERE in tune space does each particle sit? -> tune footprint.
#     Resonances m*Qx + n*Qy = p are dangerous places: when the
#     amplitude-dependent tune hits a low-order resonance, islands and
#     chaotic layers appear (scripts 01a/01b showed them in phase space).
#
#  2. HOW STABLE is the tune of each particle? -> tune diffusion map.
#     For a regular orbit the tune measured on the first half of the
#     tracking equals the tune on the second half to very high accuracy;
#     for a chaotic orbit it drifts. The quantity
#         d = log10 | Q(1st half) - Q(2nd half) |
#     is therefore a cheap chaos indicator.
#
# Tracking is done with Xsuite (on GPU if you want); the tunes are computed
# with the NAFF algorithm (nafflib), which converges much faster than a
# plain FFT (error ~ 1/N^4 with a Hann window instead of 1/N).

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import xobjects as xo
import xtrack as xt
import nafflib

# %% PARAMETERS -- edit this cell and re-run everything below
QX = 0.28          # working point (compare with the resonance lines!)
QY = 0.31
K2L = -2.0         # integrated sextupole strength
K3L = 1.0          # integrated octupole strength (try +/-2)
N_TURNS = 2000     # total turns; tunes are computed on the two halves
GRID_MAX = 0.3    # grid of initial amplitudes: 0 <= x,y <= GRID_MAX
N_GRID = 100        # points per side (N_GRID^2 particles)
R_LIM = 2.0        # aperture radius: particles beyond it are lost
MAX_RES_ORDER = 5  # draw resonance lines m*Qx + n*Qy = p with |m|+|n| <= this
GPU_DEVICE = None  # 0..3 on this machine; set to None to run on CPU


# %% Build the Hénon line and track the grid of initial amplitudes
# The aperture element removes escaping particles (their loss turn goes to
# particles.at_turn); the monitor records x, px, y, py of every particle at
# every turn, which is what NAFF needs.
if GPU_DEVICE is not None:
    context = xo.ContextCupy(device=GPU_DEVICE)
else:
    context = xo.ContextCpu()

line = xt.Line(elements=[xt.LimitEllipse(a=R_LIM, b=R_LIM),
                         xt.Multipole(knl=[0, 0, K2L, K3L]),
                         xt.LineSegmentMap(qx=QX, qy=QY, betx=1.0, alfx=0.0,
                                           bety=1.0, alfy=0.0)],
               element_names=['aperture', 'henon_kick', 'rotation'])
line.build_tracker(_context=context)

# the tiny offset avoids a particle with an exactly-zero signal in one
# plane (NAFF cannot extract a tune from an identically zero signal)
g = np.linspace(0, GRID_MAX, N_GRID) + 1e-4
# build the grid of initial conditions in the (x, y) plane, with px = py = 0
X, Y = np.meshgrid(g, g, indexing="ij")
x0 = X.ravel()
y0 = Y.ravel()

# %%
# plot the initial conditions in the (x, y)
fig, ax = plt.subplots(figsize=(6.5, 6))
ax.scatter(x0, y0, s=1)
ax.set_xlabel('x [map units]')
ax.set_ylabel('y [map units]')
ax.set_aspect('equal')
ax.set_xlim(0, GRID_MAX)
ax.set_ylim(0, GRID_MAX)
plt.show()
# %%

p = xt.Particles(x=x0.copy(), y=y0.copy(),
                 px=np.zeros_like(x0), py=np.zeros_like(x0),
                 _context=context)
line.track(p, num_turns=N_TURNS, turn_by_turn_monitor=True)
mon = line.record_last_track       # mon.x has shape (n_particles, N_TURNS)

p.move(_context=xo.ContextCpu())
p.sort(interleave_lost_particles=True)   # back to launch order
alive = p.state > 0
print(f'{alive.sum()} / {alive.size} particles survived {N_TURNS} turns')

# copy the monitor record to plain NumPy arrays ONCE: accessing mon.x pulls
# the whole buffer from the GPU every time, so don't do it inside a loop
X, PX = np.asarray(mon.x), np.asarray(mon.px)
Y, PY = np.asarray(mon.y), np.asarray(mon.py)

# %% NAFF tunes on the two halves of the tracking
# The complex signal x - i*px rotates at the tune frequency, so its main
# NAFF harmonic gives Q directly. Lost particles get NaN.
half = N_TURNS // 2
# automatically fill NaN for lost particles (they have a zero signal in one plane)
qx1 = np.full(x0.size, np.nan)     # horizontal tune, first half
qx2 = np.full(x0.size, np.nan)     # horizontal tune, second half
qy1 = np.full(x0.size, np.nan)
qy2 = np.full(x0.size, np.nan)
for i in range(x0.size):
    if alive[i]:
        qx1[i] = nafflib.get_tune(X[i, :half] - 1j * PX[i, :half])
        qx2[i] = nafflib.get_tune(X[i, half:] - 1j * PX[i, half:])
        qy1[i] = nafflib.get_tune(Y[i, :half] - 1j * PY[i, :half])
        qy2[i] = nafflib.get_tune(Y[i, half:] - 1j * PY[i, half:])

# tune diffusion indicator: distance in tune space between the two halves.
d = np.log10(np.sqrt((qx1 - qx2) ** 2 + (qy1 - qy2) ** 2) + 1e-16)

# %% Plot 1: tune footprint, colored by initial amplitude, with resonance
# lines m*qx + n*qy = p drawn up to order |m|+|n| = MAX_RES_ORDER
# (thicker line = lower order = stronger resonance).
r0 = np.sqrt(x0**2 + y0**2)
ok = alive & np.isfinite(qx1) & np.isfinite(qy1)

fig, ax = plt.subplots(figsize=(7, 6.5))
qx_lo, qx_hi = qx1[ok].min() - 0.01, qx1[ok].max() + 0.02
qy_lo, qy_hi = qy1[ok].min() - 0.02, qy1[ok].max() + 0.02

for m in range(-MAX_RES_ORDER, MAX_RES_ORDER + 1):
    for n in range(-MAX_RES_ORDER, MAX_RES_ORDER + 1):
        order = abs(m) + abs(n)
        if order == 0 or order > MAX_RES_ORDER:
            continue
        for pres in range(-2 * MAX_RES_ORDER, 2 * MAX_RES_ORDER + 1):
            if n != 0:
                # line qy = (pres - m*qx)/n across the plot window
                qx_line = np.array([qx_lo, qx_hi])
                qy_line = (pres - m * qx_line) / n
            elif m != 0:
                # vertical line qx = pres/m
                qx_line = np.array([pres / m, pres / m])
                qy_line = np.array([qy_lo, qy_hi])
            ax.plot(qx_line, qy_line, '-', color='gray',
                    lw=2.0 / order, alpha=0.6, zorder=0)

sc = ax.scatter(qx1[ok], qy1[ok], c=r0[ok], s=2, cmap='viridis')
ax.plot(QX, QY, 'r*', ms=12, label=f'linear working point ({QX}, {QY})')
ax.set_xlim(qx_lo, qx_hi)
ax.set_ylim(qy_lo, qy_hi)
ax.set_xlabel('$Q_x$')
ax.set_ylabel('$Q_y$')
ax.set_title(f'Tune footprint (resonances up to order {MAX_RES_ORDER})')
ax.legend(loc='lower left')
fig.colorbar(sc, ax=ax, label=r'initial amplitude $r_0=\sqrt{x_0^2+y_0^2}$  [map units]')
fig.tight_layout()
fig.savefig('02_tune_footprint.png', dpi=200)
plt.show()

# %% Plot 2: tune diffusion map in the initial-amplitude plane
# Gray = particle lost before N_TURNS (no tune could be computed).
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(x0[~ok], y0[~ok], c='lightgray', s=4, marker='s')
sc = ax.scatter(x0[ok], y0[ok], c=d[ok], s=4, marker='s',
                cmap='jet', vmin=-10, vmax=-2)
ax.set_xlabel('$x_0$  [map units]')
ax.set_ylabel('$y_0$  [map units]')
ax.set_title('Tune diffusion  $d=\\log_{10}|Q_{1st\\;half}-Q_{2nd\\;half}|$')
fig.colorbar(sc, ax=ax, label='d  (blue = regular, red = chaotic, gray = lost)')
fig.tight_layout()
fig.savefig('02_tune_diffusion_map.png', dpi=200)
plt.show()

# %% Plot 3: FMA in tune space, colored by tune diffusion, with resonance lines
# This is the most direct way to see which resonances are actually excited.

ok_fma = ok & np.isfinite(d)

fig, ax = plt.subplots(figsize=(7, 6.5))

qx_lo, qx_hi = qx1[ok_fma].min() - 0.01, qx1[ok_fma].max() + 0.02
qy_lo, qy_hi = qy1[ok_fma].min() - 0.02, qy1[ok_fma].max() + 0.02

for m in range(-MAX_RES_ORDER, MAX_RES_ORDER + 1):
    for n in range(-MAX_RES_ORDER, MAX_RES_ORDER + 1):
        order = abs(m) + abs(n)
        if order == 0 or order > MAX_RES_ORDER:
            continue

        for pres in range(-2 * MAX_RES_ORDER, 2 * MAX_RES_ORDER + 1):
            if n != 0:
                qx_line = np.array([qx_lo, qx_hi])
                qy_line = (pres - m * qx_line) / n
            elif m != 0:
                qx_line = np.array([pres / m, pres / m])
                qy_line = np.array([qy_lo, qy_hi])

            ax.plot(qx_line, qy_line, '-', color='gray',
                    lw=2.0 / order, alpha=0.6, zorder=0)

sc = ax.scatter(qx1[ok_fma], qy1[ok_fma],
                c=d[ok_fma], s=3, cmap='jet',
                vmin=-10, vmax=-2, zorder=2)

ax.plot(QX, QY, 'r*', ms=12, label=f'linear working point ({QX}, {QY})')

ax.set_xlim(0.27, 0.3)
ax.set_ylim(qy_lo, qy_hi)
ax.set_xlabel('$Q_x$')
ax.set_ylabel('$Q_y$')
ax.set_title(
    f'FMA in tune space: diffusion on resonance web '
    f'($|m|+|n| \\leq {MAX_RES_ORDER}$)'
)

ax.legend(loc='lower left')
fig.colorbar(sc, ax=ax,
             label=r'$d=\log_{10}|Q_{1st\;half}-Q_{2nd\;half}|$')

fig.tight_layout()
fig.savefig('02_fma_on_resonance_web.png', dpi=200)
plt.show()

# %%
