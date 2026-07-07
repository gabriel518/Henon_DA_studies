# %%
# 01b_henon_map_4d.py — the 4D Hénon map, pure Python
# ===================================================
#
# Real machines have two transverse planes, and the sextupole couples them:
# its kick depends on BOTH x and y. Using the Xsuite multipole definition
# with knl = [0, 0, K2L, K3L], the kick is
#
#     dpx = -K2L/2 * (x^2 - y^2)          - K3L/6 * (x^3 - 3*x*y^2)
#     dpy = +K2L/2 * (2*x*y)              + K3L/6 * (3*x^2*y - y^3)
#
# followed by two independent rotations: (x, px) by 2*pi*QX and (y, py) by
# 2*pi*QY. With K2L = -2 (our default) the sextupole part is
#
#     dpx = + (x^2 - y^2) ,    dpy = - 2*x*y
#
# This script shows what the coupling does, in three steps of increasing
# vertical amplitude.

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# %% PARAMETERS -- edit this cell and re-run everything below
QX = 0.28          # horizontal tune
QY = 0.31          # vertical tune
K2L = -2.0         # integrated sextupole strength
K3L = 0.0          # integrated octupole strength (try +/-1)
N_TURNS = 5000
N_PART = 20        # initial conditions per family
X_MAX = 0.55       # largest horizontal launch amplitude
Y_RATIO_SMALL = 0.1    # step 2: y0 = Y_RATIO_SMALL * x0 (slightly off-plane)

PLOT_DIR = Path('plots') if Path('plots').is_dir() else Path('../plots')

# rotation coefficients, used in every cell below
cx, sx = np.cos(2 * np.pi * QX), np.sin(2 * np.pi * QX)
cy, sy = np.cos(2 * np.pi * QY), np.sin(2 * np.pi * QY)

# %% Step 1: launch exactly in the horizontal plane (y0 = 0)
# With y = py = 0 the coupling terms vanish identically (dpy = -2*x*y = 0),
# so the vertical plane never wakes up and the horizontal motion is EXACTLY
# the 2D map of script 01a. The 4D map contains the 2D one.
fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
for x0 in np.linspace(0.02, X_MAX, N_PART):
    x, px, y, py = x0, 0.0, 0.0, 0.0
    xs, pxs, ys, pys = [], [], [], []
    for n in range(N_TURNS):
        dpx = -K2L / 2 * (x**2 - y**2) - K3L / 6 * (x**3 - 3 * x * y**2)
        dpy = +K2L / 2 * (2 * x * y) + K3L / 6 * (3 * x**2 * y - y**3)
        px, py = px + dpx, py + dpy
        x, px = cx * x + sx * px, -sx * x + cx * px
        y, py = cy * y + sy * py, -sy * y + cy * py
        if x**2 + px**2 + y**2 + py**2 > 4:
            break
        xs.append(x); pxs.append(px); ys.append(y); pys.append(py)
    axes[0].plot(xs, pxs, '.', ms=0.5)
    axes[1].plot(ys, pys, '.', ms=0.5)
axes[0].set_title('horizontal plane: identical to the 2D map')
axes[1].set_title('vertical plane: never excited')
for ax in axes:
    ax.set_xlabel('x  resp.  y  [map units]')
    ax.set_ylabel('$p_x$  resp.  $p_y$')
    ax.set_aspect('equal')
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.8, 0.8)
fig.suptitle(f'Step 1: y0 = 0   (QX = {QX}, QY = {QY})')
fig.tight_layout()
fig.savefig(PLOT_DIR / '01b_step1_inplane.png', dpi=200)
plt.show()

# %% Step 2: a small vertical amplitude (y0 = 0.1 * x0)
# Now the coupling acts. The (x, px) curves get slightly "thick": they are
# no longer closed curves but projections of a 4D torus. The vertical plane
# oscillates too, driven by the -2*x*y kick.
fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
for x0 in np.linspace(0.02, X_MAX, N_PART):
    x, px, y, py = x0, 0.0, Y_RATIO_SMALL * x0, 0.0
    xs, pxs, ys, pys = [], [], [], []
    for n in range(N_TURNS):
        dpx = -K2L / 2 * (x**2 - y**2) - K3L / 6 * (x**3 - 3 * x * y**2)
        dpy = +K2L / 2 * (2 * x * y) + K3L / 6 * (3 * x**2 * y - y**3)
        px, py = px + dpx, py + dpy
        x, px = cx * x + sx * px, -sx * x + cx * px
        y, py = cy * y + sy * py, -sy * y + cy * py
        if x**2 + px**2 + y**2 + py**2 > 4:
            break
        xs.append(x); pxs.append(px); ys.append(y); pys.append(py)
    axes[0].plot(xs, pxs, '.', ms=0.5)
    axes[1].plot(ys, pys, '.', ms=0.5)
axes[0].set_title('horizontal: curves become "thick" projections')
axes[1].set_title('vertical: small driven oscillation')
for ax in axes:
    ax.set_xlabel('x  resp.  y  [map units]')
    ax.set_ylabel('$p_x$  resp.  $p_y$')
    ax.set_aspect('equal')
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.8, 0.8)
fig.suptitle(f'Step 2: y0 = {Y_RATIO_SMALL} * x0')
fig.tight_layout()
fig.savefig(PLOT_DIR / '01b_step2_small_y.png', dpi=200)
plt.show()

# %% Step 3: the diagonal (y0 = x0), fully coupled motion
# Equal amplitudes in both planes: the motion is genuinely 4-dimensional.
# Neither panel shows closed curves any more -- each is the shadow of a 4D
# object. Note also that the stable region along the diagonal is SMALLER
# than on the x axis: coupling opens more roads to instability.
fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
for x0 in np.linspace(0.02, 0.75 * X_MAX, N_PART):
    x, px, y, py = x0, 0.0, x0, 0.0
    xs, pxs, ys, pys = [], [], [], []
    for n in range(N_TURNS):
        dpx = -K2L / 2 * (x**2 - y**2) - K3L / 6 * (x**3 - 3 * x * y**2)
        dpy = +K2L / 2 * (2 * x * y) + K3L / 6 * (3 * x**2 * y - y**3)
        px, py = px + dpx, py + dpy
        x, px = cx * x + sx * px, -sx * x + cx * px
        y, py = cy * y + sy * py, -sy * y + cy * py
        if x**2 + px**2 + y**2 + py**2 > 4:
            break
        xs.append(x); pxs.append(px); ys.append(y); pys.append(py)
    axes[0].plot(xs, pxs, '.', ms=0.5)
    axes[1].plot(ys, pys, '.', ms=0.5)
axes[0].set_title('horizontal projection')
axes[1].set_title('vertical projection')
for ax in axes:
    ax.set_xlabel('x  resp.  y  [map units]')
    ax.set_ylabel('$p_x$  resp.  $p_y$')
    ax.set_aspect('equal')
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.8, 0.8)
fig.suptitle('Step 3: diagonal launch y0 = x0')
fig.tight_layout()
fig.savefig(PLOT_DIR / '01b_step3_diagonal.png', dpi=200)
plt.show()

# %%
# Things to try:
#  - Scan Y_RATIO_SMALL from 0 to 1: watch the transition 2D -> 4D.
#  - Move QY close to QX (e.g. 0.29): the difference resonance QX - QY = 0
#    couples the planes strongly and the projections change character.
#  - Octupole on (K3L = 1): its detuning acts with opposite signs in the
#    two planes (look at the kick formula: x^3 vs -y^3).