# %%
# 01a_henon_map_2d.py — the 2D Hénon map, pure Python
# ===================================================
#
# The simplest model of nonlinear betatron motion in one plane: in
# normalized coordinates (beta = 1, alpha = 0) one turn of the machine is
#
#   1. a thin multipole kick at the location of the nonlinear magnet,
#   2. a rotation of the (x, px) plane by the phase advance 2*pi*Q.
#
# We use exactly the Xsuite definition of the multipole kick,
#
#     dpx = - K2L/2 * x^2  -  K3L/6 * x^3
#
# (K2L, K3L = integrated sextupole and octupole strengths; the factors 1/2!
# and 1/3! come from the multipole expansion). With K2L = -2 the sextupole
# kick is dpx = +x^2, which is the classic Hénon map convention. The
# rotation is
#
#     x'  =  cos(2*pi*Q) * x + sin(2*pi*Q) * px
#     px' = -sin(2*pi*Q) * x + cos(2*pi*Q) * px
#
# Everything is dimensionless ("map units"): the sextupole strength sets
# the scale, so "weak nonlinearity" simply means "small amplitude".

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# %% PARAMETERS -- edit this cell and re-run everything below
Q = 0.28           # tune: number of betatron oscillations per turn 
K2L = -2.0         # integrated sextupole strength (-2 -> kick dpx = +x^2)
K3L = +1.0          # integrated octupole strength 
N_TURNS = 5000     # turns to track
X0_SINGLE = 0.001  # launch amplitude of the single particle in step 1
N_PART = 20        # number of initial conditions along the x axis
X_MAX = 1.0       # largest launch amplitude

PLOT_DIR = Path('plots') if Path('plots').is_dir() else Path('../plots')

# %% Step 1: follow ONE particle for a few turns
# Launch a single particle and store its position turn
# after turn. At small amplitude the kick is negligible and the map is
# almost a pure rotation: the points hop around a circle, Q tells how big
# the hop is (0.28 of a full circle per turn).
x, px = X0_SINGLE, 0.0
x_list, px_list = [x], [px]
for n in range(N_TURNS):
    px = px - K2L / 2 * x**2                     # sextupole kick (Xsuite sign)
    x, px = (np.cos(2 * np.pi * Q) * x + np.sin(2 * np.pi * Q) * px,
             -np.sin(2 * np.pi * Q) * x + np.cos(2 * np.pi * Q) * px)
    x_list.append(x)
    px_list.append(px)

fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
axes[0].plot(x_list[:16], px_list[:16], 'o-', ms=5)
for n in range(16):
    axes[0].annotate(str(n), (x_list[n], px_list[n]), fontsize=8,
                     xytext=(3, 3), textcoords='offset points')
axes[0].set_title('first 15 turns (numbers = turn index)')
axes[1].plot(x_list, px_list, '.k', ms=1)
axes[1].set_title(f'{N_TURNS} turns: the orbit fills a closed curve')
for ax in axes:
    ax.set_xlabel('x  [map units]')
    ax.set_ylabel('$p_x$  [map units]')
    ax.set_aspect('equal')
fig.suptitle(f'One particle, x0 = {X0_SINGLE}, Q = {Q}')
fig.tight_layout()
fig.savefig(PLOT_DIR / '01a_single_particle.png', dpi=200)
plt.show()

# %% Step 2: the full phase-space portrait (sextupole only here)
# Now launch many particles at increasing amplitude. Small amplitudes ->
# almost circles. Larger amplitudes -> the curves get distorted.
x0_list = np.linspace(0.02, X_MAX, N_PART)

fig, ax = plt.subplots(figsize=(6.5, 6.5))
for x0 in x0_list:
    x, px = x0, 0.0
    x_traj, px_traj = [], []
    for n in range(N_TURNS):
        px = px - K2L / 2 * x**2
        x, px = (np.cos(2 * np.pi * Q) * x + np.sin(2 * np.pi * Q) * px,
                 -np.sin(2 * np.pi * Q) * x + np.cos(2 * np.pi * Q) * px)
        if x**2 + px**2 > 4:          # escaped: no point tracking further
            break
        x_traj.append(x)
        px_traj.append(px)
    ax.plot(x_traj, px_traj, '.', ms=0.5)
ax.set_xlabel('x  [map units]')
ax.set_ylabel('$p_x$  [map units]')
ax.set_title(f'2D Hénon map, sextupole only (K2L = {K2L}), Q = {Q}')
ax.set_aspect('equal')
ax.set_xlim(-0.8, 0.8)
ax.set_ylim(-0.8, 0.8)
fig.tight_layout()
fig.savefig(PLOT_DIR / '01a_phase_space_sextupole.png', dpi=200)
plt.show()

# %% Step 3: add the octupole
# Same portrait with the octupole kick switched on. The octupole adds a
# strong amplitude-dependent tune shift ("detuning"): the tune of each
# curve changes with its size, so resonances are met at different
# amplitudes and the whole geography of islands and chaos rearranges.
fig, ax = plt.subplots(figsize=(6.5, 6.5))
for x0 in x0_list:
    x, px = x0, 0.0
    x_traj, px_traj = [], []
    for n in range(N_TURNS):
        px = px - K2L / 2 * x**2 - K3L / 6 * x**3     # sextupole + octupole
        x, px = (np.cos(2 * np.pi * Q) * x + np.sin(2 * np.pi * Q) * px,
                 -np.sin(2 * np.pi * Q) * x + np.cos(2 * np.pi * Q) * px)
        if x**2 + px**2 > 4:
            break
        x_traj.append(x)
        px_traj.append(px)
    ax.plot(x_traj, px_traj, '.', ms=0.5)
ax.set_xlabel('x  [map units]')
ax.set_ylabel('$p_x$  [map units]')
ax.set_title(f'with octupole: K2L = {K2L}, K3L = {K3L}, Q = {Q}')
ax.set_aspect('equal')
ax.set_xlim(-0.8, 0.8)
ax.set_ylim(-0.8, 0.8)
fig.tight_layout()
fig.savefig(PLOT_DIR / '01a_phase_space_octupole.png', dpi=200)
plt.show()
# %%
