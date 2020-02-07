"""
 SIMULATING INTERACTING GALAXIES AND THE FORMATION OF TIDAL TAILS

 simulate_halos

 This script provides functions for initialising, simulating and testing
 the interaction between galaxies with uniform spherical mass centres (halos)
 in planar parabolic orbit, simulates their interaction and displays their
 resulting paths.

 Further simulations then experiment with varying halo radii.
 
"""
# --------------
# Libraries and constants
# --------------
import numpy as np    # arrays, vectorization and mathematical objects
from scipy.integrate import odeint    # stepwise integration

# plots, snapshots and animation
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
# plotting parameters
matplotlib.rc('xtick', labelsize=12) 
matplotlib.rc('ytick', labelsize=12) 
matplotlib.rcParams.update({'font.size': 15})

# useful constants
pi = np.pi
cos = np.cos
sin = np.sin
# useful functions
mag = np.linalg.norm
array = np.array
# scaling of universe
G = 1    # gravitational constance
M = 1    # reference mass of bodies

# project functions
from base_funcs import stack_sol, initialise_parabola, initialise_stars, s_circular
from display_funcs import snapshot_parabola, animate_2D

# --------------
# Interacting galaxies with dark matter halos
# --------------

def g_inside_halo(m, A, x, x1s):
    # Return grav. acceleration at positions (x1s) inside halo mass (m) radius (A) centred at (x)
    
    rs = x - x1s.transpose()    # separations
    inv_As = G * m / A**3
    return inv_As * rs.transpose()    # return in matrix form


def s_circular_halo(m, A, R):
    # Speed of stable circular orbit at radius (R) about halo radius (A) mass (m)
    
    if R > A:
        return s_circular(m, R)
    else:
        return R * np.sqrt(G * m / A**3)


def circle_halo(m, A, R, N, anticloc, C, v_C, n_D):
    """
    # Give stable circle(m, R, N, anticloc, C, v_C, n_D) but for halo radius (A)
    # of mass rather than point
    """
    s = s_circular_halo(m, A, R)
    sign = 1 if anticloc else -1
    return ring(R, N, (sign * s), C, v_C, n_D)


def initialise_stars_halo(m, A, Rs, density, anticloc, C, v_C, n_D):
    """
     initialise_stars(m, Rs, density, anticloc, C, v_C, n_D) but for halo radius
     (A) of mass rather than point
    """
    star_circles = [circle_halo(m, A, R, density*R, anticloc, C, v_C, n_D) for R in Rs]
    return np.concatenate(star_circles, axis=1)


def analytic_omega_halo(m):
    # Analytic ang. velocity for circular orbit radius (R) about (m)
    return np.sqrt(G * m / A**3)


def analytic_halo(m, R, t):
    """
     Analytic position in xy-plane of particle in orbit (R) about (m) at time (t)
     - assumes orbit starts in x-axis at t=0
    """
    omega = analytic_omega_halo(m)
    x_at_t = lambda T : R * np.array( [cos(omega*T), sin(omega*T)] )
    return x_at_t(t)


def halo_errors(sol, m, R, t):
    """
     Differences between computed positions of (sol) and analytic circular orbit
     (R) about (m) at (t)
    """
    D = 2    # sol assumed 2D
    xs = stack_sol(sol, D)[:, :D, 1].transpose()    # extract position of second particle at for each t
    difs = xs - analytic_halo(m, R, t)
    return mag(difs, axis=0)


# Analyse errors
# --------------

# Initialise
D = 2

A = 6.5
T_halo = 2 * pi / analytic_omega_halo(M)
end_time = 100 * T_halo
step_size = 0.04
t = timescale(end_time, step_size)


N = 1
anticloc = True
Rs = np.arange(2, 7)

halo = np.zeros((2*D, 1))
C = halo[D:, 0]
v_C = halo[:D, 0]

star_circles = [circle_halo(M, A, R, N, anticloc, C, v_C, D) for R in Rs]
stars = np.concatenate(star_circles, axis=1)

galaxy = np.append(halo, stars, axis=1)
q0 = galaxy.flatten()

g_halo_A = lambda m, x, x1s : g_halo(m, A, x, x1s)

halo_circle_sol = odeint(g_diffs, q0, t, args=([M], g_halo_A, D))

snap_start = int( (end_time - ((3 / 4) * T_halo)) / step_size )    # step no. for 1/4 way through last orbit
snap_end = len(halo_circle_sol[snap_start:]) - 1
snapshot_circle(halo_circle_sol[snap_start:], snap_end, t[snap_start:], A, [], D, 'snapshot - halo test')

# compare simulation and theory
xs = stack_sol(halo_circle_sol, D)[:, 0, 1].transpose()    # extract xs of test particle
analytic_xs = analytic_halo(M, R, t)[0]

start = int( (99/100) * len(t) )    # begin plot at start of 100th orbit
plt.plot(t[start:], xs[start:], label='computed')
plt.plot(t[start:], analytic_xs[start:], 'k:', label='analytical')
plt.legend(loc='best')
plt.xlabel('t/s')
plt.ylabel('x component of position/m')
plt.savefig('comparison of computed and analytic circular motion.pdf')
plt.show()

# plot errors for R=2
h_errs = halo_errors(halo_circle_sol, M, Rs[0], t)

plt.plot(t, h_errs)
plt.xlabel('t/s')
plt.ylabel('Absolute error/m')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.savefig('Absolute error for halo circle R=2.pdf')
plt.show()
plt.show()

# Investigate
# --------------

# standard setup
D = 2
step_size = 0.04
end_time = 150.0
t = timescale(end_time, step_size)

Ms = [M, M]
Rs = np.arange(2,7)
density = 200
anticloc = True
R_0 = 30
R_min = 12

As = np.linspace(1.5, 6.5, 6)    # halo radii specified to contain each ring in turn

for A in As:
    # initialise
    black_holes = initialise_parabola(Ms, R_0, R_min, D)
    
    C = black_holes[:D, 0]
    v_C = black_holes[D:, 0]
    stars = initialise_stars_halo(Ms[0], A, Rs, density, anticloc, C, v_C, D)
    
    galaxies = np.append(black_holes, stars, axis=1)
    q0 = galaxies.flatten()
    
    # create compatible gfunc for given halo radius A
    g_halo_A = lambda m, x, x1 : g_halo(m, A, x, x1)
    
    # solve
    h_sol = odeint(g_diffs, q0, t, args=(Ms, g_halo_A, D))
    
    # save
    title = 'collision solutions - A = ' + str(A) + '.txt'
    np.savetxt(title, h_sol[0::interval])

# animate
halo_sols = [np.loadtxt('collision solutions - A = ' + str(A) + '.txt') for A in As]

ax_lims = [-35, 25, -35, 25]
for i, A in enumerate(As):
    animate_2D(halo_sols[i], num_frames, ax_lims, D, 'Animation - A = ' + str(A))

# snapshots
step = 90
snapshot_parabola(halo_sols[1], 90, t, As[1], n_paths, outer_lims, D, 'snapshot - A = 2.5')
snapshot_parabola(halo_sols[3], 90, t, As[3], n_paths, outer_lims, D, 'snapshot - A = 4.5')
snapshot_parabola(halo_sols[5], 90, t, As[5], n_paths, outer_lims, D, 'snapshot - A = 6.5')

# Solve A=6.5 case for longer (350s) time
# --------------
# standard setup but with 350s of steps and subsequently larger starting separation
D = 2
step_size = 0.04
end_time = 350.0
t = timescale(end_time, step_size)

Ms = [M, M]
Rs = np.arange(2,7)
density = 200
anticloc = True
R_0 = 60
R_min = 12

A = 6.5

# initialise
black_holes = initialise_parabola(Ms, R_0, R_min, D)

C = black_holes[:D, 0]
v_C = black_holes[D:, 0]
stars = initialise_stars_halo(Ms[0], A, Rs, density, anticloc, C, v_C, D)

galaxies = np.append(black_holes, stars, axis=1)
q0 = galaxies.flatten()

g_halo_A = lambda m, x, x1 : g_halo(m, A, x, x1)

# solve
large_halo_sol = odeint(g_diffs, q0, t, args=(Ms, g_halo_A, D))

# save
title = 'collision solutions - A = ' + str(A) + 'over 350s.txt'
np.savetxt(title, large_halo_sol[0::interval])

# animate
animate_2D(large_halo_sol, num_frames, [], D, 'Animation - A = ' + str(A) + ' over 350s')

snapshot_parabola(large_halo_sol, 4300, t, A, n_paths, inner_lims, D, 'snapshot - halo bulge')
snapshot_parabola(large_halo_sol, 5600, t, A, n_paths, inner_lims, D, 'snapshot - halo miss')
snapshot_parabola(large_halo_sol, 6500, t, A, n_paths, outer_lims, D, 'snapshot - halo tail')
snapshot_parabola(large_halo_sol, 8000, t, A, n_paths, outer_lims, D, 'snapshot - halo spiral')
