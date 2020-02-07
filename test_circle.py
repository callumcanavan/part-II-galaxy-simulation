"""
 SIMULATING INTERACTING GALAXIES AND THE FORMATION OF TIDAL TAILS

 test_circle

 This script provides functions for simulating the circular orbit of a single 
 star around a single point mass and tests that it can maintain its circular
 orbit over many periods.

"""

# --------------
# Libraries and constants
# --------------
import numpy as np    # arrays, vectorization and mathematical objects
import time    # finding processing times
from scipy.integrate import odeint    # stepwise integration

# plots, snapshots and animation
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
# plotting parameters
matplotlib.rc('xtick', labelsize=12) 
matplotlib.rc('ytick', labelsize=12) 
matplotlib.rcParams.update({'font.size': 15})

# useful functions
array = np.array
zeros = np.zeros
mag = np.linalg.norm
# useful constants
pi = np.pi
cos = np.cos
sin = np.sin
# scaling of universe
G = 1    # gravitational constance
M = 1    # reference mass of bodies

# project functions
from base_funcs import stack, stack_sol, circle
from display_funcs import snapshot_circle

# --------------
# Circular orbit test
# --------------

def analytic_omega(m, R):
    # Analytic ang. velocity for circular orbit radius (R) about (m)
    return np.sqrt(G * m / R**3)


def analytic_circle(m, R, t):
    """
     Analytic position in xy-plane of particle in orbit (R) about (m) at time (t)
     - assumes orbit starts in x-axis at t=0
    """
    omega = analytic_omega(m, R)
    x_at_t = lambda T : R * np.array( [cos(omega*T), sin(omega*T)] )
    return x_at_t(t)


def circle_errors(sol, m, R, t):
    # Differences between computed positions of (sol) and analytic circular orbit (R) about (m) at (t)
    
    D = 2    # sol assumed 2D
    xs = stack_sol(sol, D)[:, :D, 1].transpose()    # extract position of second particle at for each t
    difs = xs - analytic_circle(m, R, t)
    return mag(difs, axis=0)


def timescale(end_time, step_size):
    # Generate time array between 0 and (end_time) with (step_size) between each frame
    return np.linspace(0, end_time, int(end_time/step_size))


def single_star_sol(m, R, step_size):
    # Initialise and integrate circular (R) orbit about (m) with time intervals (step_size)
    
    D = 2

    # simulate over 100 periods of R=2 orbit
    R2_period = 2 * pi / analytic_omega(m, 2)
    end_time = 100 * R2_period
    t = timescale(end_time, step_size)

    C = v_C = zeros(2)
    black_hole = stack(array([C, v_C]), D)
    
    N = 1
    anticloc = True

    star = circle(m, R, N, anticloc, C, v_C, D)
    galaxy = np.append(black_hole, star, axis=1)
    q0 = galaxy.flatten()
    
    sol = odeint(g_diffs, q0, t, args=([M], g_point, D))
    return sol

# Find errors for R between 2 and 6
# ------------
Rs = np.arange(2,7)
step_size = 0.04
circle_sols = [single_star_sol(M, R, step_size) for R in Rs]

R2_period = 2 * pi / analytic_omega(M, 2)
end_time = 100 * R2_period
t = timescale(end_time, step_size)

abs_errs = [circle_errors(sol, M, R, t) for sol, R in zip(circle_sols, Rs)]
rel_errs = [err / R for err, R in zip(abs_errs, Rs)]

max_errs = [np.max(errs) for errs in abs_errs]

# Plot results
# ------------
plt.plot(t, abs_errs[0])
plt.xlabel('t/s')
plt.ylabel('Absolute error/m')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.savefig('Absolute error for circle R=2.pdf')
plt.show()

plt.plot(t, rel_errs[0])
plt.xlabel('t/s')
plt.ylabel('Relative error')
plt.savefig('Relative error for circle R=2.pdf')
plt.show()

plt.plot(Rs, max_errs)
plt.xlabel('Radius of orbit/m')
plt.ylabel('Maximum error/m')
plt.savefig('Max error of circular orbit vs R.pdf')
plt.show()

# compare simulation and theory for R=2 orbit
D = 2
i = 0    # index of R=2 orbit
R = Rs[i]
xs = stack_sol(circle_sols[i], D)[:, 0, 1].transpose()    # extract xs of test particle
analytic_xs = analytic_circle(M, R, t)[0]

start = int( (99/100) * len(t) )    # begin plot at start of 100th orbit
plt.plot(t[start:], xs[start:], label='computed')
plt.plot(t[start:], analytic_xs[start:], 'k:', label='analytical')
plt.legend(loc='best')
plt.xlabel('t/s')
plt.ylabel('x component of position/m')
plt.savefig('comparison of computed and analytic circular motion.pdf')
plt.show()

step = len(t) - 1
A = 0
snapshot_circle(circle_sols[i], step, t, A, [], D, 'snapshot - circle R=2')

# Find errors for different step sizes
# ------------
R = 2    # compare for same R=2 since this had max error
step_exps = np.linspace(-3, 0, num=14)
step_sizes = 10**step_exps

circle_step_sols = [single_star_sol(M, R, step_size) for step_size in step_sizes]

R2_period = 2 * pi / analytic_omega(M, R)
end_time = 100 * R2_period

max_errs_circle = [np.max(circle_errors(sol, M, R, timescale(end_time, step_size))) for sol, step_size in zip(circle_step_sols, step_sizes)]

fig, ax = plt.subplots()
plt.plot(step_exps, max_errs_circle, 'b+', markersize=10,)
plt.xlabel('log(Step size/s)')
plt.ylabel('Maximum error/m', size=13)
plt.tick_params(labelsize=10)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.savefig('Max error of circular orbit vs step size.pdf')
plt.show()
