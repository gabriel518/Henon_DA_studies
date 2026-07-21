# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# %% PARAMETERS
WX0 = 0.28         # LHC working point
WY0 = 0.31
MU = 0.5           # cubic nonlinearity intensity
MODULATION_ON = False   #tune modulation on/off
EPS = 32.0         # modulation amplitude (used only if MODULATION_ON)
N_MAX = int(1e5)   # turns tracked
R_C = 10.0

# polar grid of initial conditions, px = py = 0
N_ANGLES = 50
R_MIN, R_MAX, DR = 0.02, 0.50, 0.005

# %% Functions
OMEGA_K = 2 * np.pi / 868.12 * np.array([1, 2, 3, 6, 7, 10, 12])
E_K = 1e-4 * np.array([1.000, 0.218, 0.708, 0.254, 0.100, 0.078, 0.218])


def henon_step(x, px, y, py, wx, wy):
    """One turn of the 4D Hénon map: nonlinear kick, then rotation.
    Works on whole arrays of particles at once."""
    dpx = x**2 - y**2 + MU * (x**3 - 3 * x * y**2)
    dpy = -2 * x * y + MU * (y**3 - 3 * x**2 * y)
    px, py = px + dpx, py + dpy

    cx, sx = np.cos(2 * np.pi * wx), np.sin(2 * np.pi * wx)
    cy, sy = np.cos(2 * np.pi * wy), np.sin(2 * np.pi * wy)
    x, px = cx * x + sx * px, -sx * x + cx * px
    y, py = cy * y + sy * py, -sy * y + cy * py
    return x, px, y, py


def loss_turns(x0, y0, n_max, eps):
    """Track all particles together and return, for each of them, the turn
    at which it crossed R_C (n_max + 1 if it survived)."""
    x, px = x0.copy(), np.zeros_like(x0)
    y, py = y0.copy(), np.zeros_like(y0)
    loss = np.full(x0.size, n_max + 1, dtype=np.int64)

    for n in range(1, n_max + 1):
        # modulated frequencies of this turn (the same for every particle)
        mod = eps * np.sum(E_K * np.cos(OMEGA_K * n)) if eps else 0.0
        x, px, y, py = henon_step(x, px, y, py,
                                  WX0 * (1 + mod), WY0 * (1 + mod))

        # Record the turn of the particles that just crossed R_C and park
        # them at the origin. The origin is a fixed point of the map, so a
        # parked particle stays there and is never recorded twice.
        lost = x * x + px * px + y * y + py * py > R_C * R_C
        loss[lost] = n
        x[lost] = px[lost] = y[lost] = py[lost] = 0.0
    return loss


# %% Build the polar grid of initial conditions in the (x, y) plane
theta = np.linspace(0, np.pi / 2, N_ANGLES + 2)[1:-1]
r = np.arange(R_MIN, R_MAX + DR / 2, DR)
n_r = len(r)

TH, R = np.meshgrid(theta, r, indexing='ij')
x0 = (R * np.cos(TH)).ravel()
y0 = (R * np.sin(TH)).ravel()

# %% Track
eps = EPS if MODULATION_ON else 0.0
print(f'tracking {x0.size} particles for {N_MAX} turns, '
      f'modulation {"ON" if MODULATION_ON else "OFF"} (EPS = {eps})')
loss = loss_turns(x0, y0, N_MAX, eps).reshape(N_ANGLES, n_r)
print(f'{(loss > N_MAX).sum()} of {loss.size} particles survived')

# %% Dynamic aperture: for each angle walk outwards and stop at the first
# radius that does not survive N turns. Then average over the angles to get D(N).
N_list = np.unique(np.logspace(2, np.log10(N_MAX), 20).astype(np.int64))

r_stable = np.zeros((len(N_list), N_ANGLES))
for k, N in enumerate(N_list):
    for j in range(N_ANGLES):
        i = 0
        while i < n_r and loss[j, i] >= N:
            i += 1
        r_stable[k, j] = r[i - 1] if i > 0 else 0.0

D_N = r_stable.mean(axis=1)

for k in range(0, len(N_list), 5):
    print(f'D(N = {N_list[k]:>8d}) = {D_N[k]:.4f}')
print(f'D(N = {N_list[-1]:>8d}) = {D_N[-1]:.4f}')

# %% Plot the stability domain: color = turn at which the particle was lost,
# with the border r_stable(theta; N) drawn for a few N. Each longer time
# scale eats one more layer off the stable region.
fig, ax = plt.subplots(figsize=(7.5, 6))
pc = ax.pcolormesh(R * np.cos(TH), R * np.sin(TH), loss,
                   norm=LogNorm(vmin=1, vmax=N_MAX), cmap='viridis')
for k, color in zip(np.linspace(0, len(N_list) - 1, 4).astype(int),
                    ['red', 'darkorange', 'magenta', 'black']):
    ax.plot(r_stable[k] * np.cos(theta), r_stable[k] * np.sin(theta),
            '-', color=color, lw=1.3, label=f'N = {N_list[k]:.0e}')
ax.set_xlabel('$x_0$  [map units]')
ax.set_ylabel('$y_0$  [map units]')
ax.set_title(f'Stability domain, $\\omega$ = ({WX0}, {WY0}), MU = {MU}, '
             f'modulation {"ON" if MODULATION_ON else "OFF"}')
ax.set_aspect('equal')
ax.legend(loc='upper right', fontsize=8)
fig.colorbar(pc, ax=ax, label='loss turn (yellow = alive at N_MAX)')
fig.tight_layout()
fig.savefig('01d_stability_domain.png', dpi=200)
plt.show()

# %% Plot D(N): the dynamic aperture shrinks as we ask for longer stability
fig, ax = plt.subplots(figsize=(7, 5))
ax.semilogx(N_list, D_N, 'o-', ms=4)
ax.set_xlabel('N  [turns]')
ax.set_ylabel('D(N)  [map units]')
ax.set_title(f'Dynamic aperture vs turns, '
             f'modulation {"ON" if MODULATION_ON else "OFF"}')
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('01d_DA_vs_N.png', dpi=200)
plt.show()

# %%