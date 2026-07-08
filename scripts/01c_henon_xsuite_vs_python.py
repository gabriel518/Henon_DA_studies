# %%
# 01c_henon_xsuite_vs_python.py — the same 4D map in Xsuite and in Python
# =======================================================================
#
# Xsuite is a real tracking code: lines, elements, particles, apertures,
# GPUs. The Hénon map is a two-element line:
#
#     henon_kick = xt.Multipole(knl=[0, 0, K2L, K3L])   # thin sextupole+octupole
#     rotation   = xt.LineSegmentMap(qx=QX, qy=QY, betx=1, bety=1)
#
# Here we track the same initial conditions with Xsuite and with our plain
# Python implementation of 01b and check that they agree to machine
# precision. If they do, we can trust the fast Python/numba loops of
# scripts 03-05 AND the Xsuite tracking (CPU or GPU) interchangeably.

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import xobjects as xo
import xtrack as xt
from pathlib import Path

# %% PARAMETERS -- edit this cell and re-run everything below
QX = 0.28
QY = 0.31
K2L = -2.0         # integrated sextupole strength
K3L = 1.0          # integrated octupole strength (ON by default here)
N_TURNS = 5000
N_PART = 20        # particles per family
X_MAX = 0.45       # largest launch amplitude (octupole shrinks the border a bit)
Y_RATIO = 0.5      # the "non-diagonal" family: y0 = Y_RATIO * x0
GPU_DEVICE = None  # None -> CPU. On some machines you can set 0, 1, 2 or 3
                   # to track on one of the GPUs

PLOT_DIR = Path('plots') if Path('plots').is_dir() else Path('../plots')

# %% Initial conditions: a non-diagonal family (y0 = Y_RATIO*x0) and the
# diagonal one (y0 = x0); all momenta zero.
x0 = np.concatenate([np.linspace(0.02, X_MAX, N_PART),
                     np.linspace(0.02, X_MAX, N_PART)])
y0 = np.concatenate([Y_RATIO * np.linspace(0.02, X_MAX, N_PART),
                     np.linspace(0.02, X_MAX, N_PART)])

# %% Track with Xsuite
context = xo.ContextCupy(device=GPU_DEVICE) if GPU_DEVICE is not None \
    else xo.ContextCpu()

henon_kick = xt.Multipole(knl=[0, 0, K2L, K3L])
rotation = xt.LineSegmentMap(qx=QX, qy=QY, betx=1.0, alfx=0.0,
                             bety=1.0, alfy=0.0)
line = xt.Line(elements=[henon_kick, rotation],
               element_names=['henon_kick', 'rotation'])
line.build_tracker(_context=context)

p = xt.Particles(x=x0.copy(), y=y0.copy(),
                 px=np.zeros_like(x0), py=np.zeros_like(x0),
                 _context=context)
line.track(p, num_turns=N_TURNS, turn_by_turn_monitor=True)
mon = line.record_last_track          # mon.x has shape (n_particles, N_TURNS)

# %% Track with plain Python (same map as 01b, vectorized over particles)
cx, sx = np.cos(2 * np.pi * QX), np.sin(2 * np.pi * QX)
cy, sy = np.cos(2 * np.pi * QY), np.sin(2 * np.pi * QY)

X = np.zeros((len(x0), N_TURNS))
PX = np.zeros_like(X)
Y = np.zeros_like(X)
PY = np.zeros_like(X)
x, px, y, py = x0.copy(), np.zeros_like(x0), y0.copy(), np.zeros_like(x0)
with np.errstate(over='ignore', invalid='ignore'):   # escapes overflow, harmless
    for n in range(N_TURNS):
        X[:, n], PX[:, n], Y[:, n], PY[:, n] = x, px, y, py
        dpx = -K2L / 2 * (x**2 - y**2) - K3L / 6 * (x**3 - 3 * x * y**2)
        dpy = +K2L / 2 * (2 * x * y) + K3L / 6 * (3 * x**2 * y - y**3)
        px, py = px + dpx, py + dpy
        x, px = cx * x + sx * px, -sx * x + cx * px
        y, py = cy * y + sy * py, -sy * y + cy * py

# %% Compare: maximum absolute difference over all bounded samples
# (once a particle escapes, both codes blow up to inf/nan in their own way,
# and chaotic orbits amplify round-off exponentially -- so we compare only
# while the orbit stays within the stable region)
bounded = np.abs(X) < 2.0
res = max(np.max(np.abs((X - mon.x)[bounded])),
          np.max(np.abs((PX - mon.px)[bounded])),
          np.max(np.abs((Y - mon.y)[bounded])),
          np.max(np.abs((PY - mon.py)[bounded])))
print(f'max |python - xsuite| over {N_TURNS} turns: {res:.3e}')
assert res < 1e-10, 'the two implementations disagree!'

# %% Overlay the phase-space portraits: black = Xsuite, red = Python.
# If the two implementations agree, the red dots sit exactly on the black
# ones and the plot looks like a single portrait.
fig, axes = plt.subplots(2, 2, figsize=(11, 10))
for row, sel, fam in [(0, slice(0, N_PART), f'y0 = {Y_RATIO}*x0'),
                      (1, slice(N_PART, 2 * N_PART), 'y0 = x0 (diagonal)')]:
    for i in range(*sel.indices(2 * N_PART)):
        ok = np.abs(mon.x[i]) < 2.0
        axes[row, 0].plot(mon.x[i][ok], mon.px[i][ok], '.k', ms=1.2)
        axes[row, 1].plot(mon.y[i][ok], mon.py[i][ok], '.k', ms=1.2)
        ok = np.abs(X[i]) < 2.0
        axes[row, 0].plot(X[i][ok], PX[i][ok], '.r', ms=0.4)
        axes[row, 1].plot(Y[i][ok], PY[i][ok], '.r', ms=0.4)
    axes[row, 0].set_title(f'{fam}: (x, px)')
    axes[row, 1].set_title(f'{fam}: (y, py)')
for ax in axes.ravel():
    ax.set_xlabel('x  resp.  y  [map units]')
    ax.set_ylabel('$p_x$  resp.  $p_y$')
    ax.set_aspect('equal')
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.7, 0.7)
fig.suptitle(f'Xsuite (black) vs Python (red), K2L = {K2L}, K3L = {K3L}')
fig.tight_layout()
fig.savefig(PLOT_DIR / '01c_xsuite_vs_python.png', dpi=200)
plt.show()

# %%
