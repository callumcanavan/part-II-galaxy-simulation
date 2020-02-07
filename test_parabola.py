"""
 SIMULATING INTERACTING GALAXIES AND THE FORMATION OF TIDAL TAILS

 test_parabola

 This script provides functions for simulating the parabolic orbit of two 
 point masses about each other and tests that these are accurate to the 
 analytical solutions.
 
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
from base_funcs import stack, stack_sol, initialise_parabola, quantity_over_time, E
from display_funcs import snapshot_parabola

# --------------
# Parabolic orbit test
# --------------

def math_parabola(ms, R_min, y):
    """
     Calculate from (y) the analytic x-value for a body in parabolic
     orbit of two bodies (ms), min separation = (R_min)
     - assumes body being tested has -ve vertex
    """
    m1, m2 = ms
    R_1 = R_min * m2 / (m1 + m2)
    return (y**2 / (4 * R_1)) - R_1


def parabola_errors(sol, ms, R_min):
    """
     Difference between computed and analytic x-values for given y-values of
     (sol), with min separation (R_min)
    """
    D = 2
    xs, ys = stack_sol(sol, D)[:, :D, 0].transpose()
    difs = xs - math_parabola(ms, R_min, ys)
    return difs


def min_separation(sol, n_D):
    """
     Return minimum separation throughout (sol) between first two particles in
     (n_D) space
    """
    xs = stack_sol(sol, n_D)[:, :n_D]
    seps = mag(xs[:,:,0] - xs[:,:,1], axis=1)
    return np.min(seps)


def parabola_sol(R_min, step_size):
    # Initialise and integrate parabolic orbit of (R_min) with time steps (step_size) 
    D = 2
    end_time = 1000.0
    t = timescale(end_time, step_size)
    
    Ms = [M, M]    # masses of bodeis
    R_0 = 125    # initiatl separation
    black_holes = initialise_parabola(Ms, R_0, R_min, D)
    q0 = black_holes.flatten()
    
    sol = odeint(g_diffs, q0, t, args=(Ms, g_point, D))
    return sol


# Compare errors for different R_mins
R_mins = np.arange(12,17)
Ms = [M, M]
step_size = 0.04
parabola_sols = [parabola_sol(R_min, step_size) for R_min in R_mins]

end_time = 1000.0
t = timescale(end_time, step_size)

abs_errs = [parabola_errors(sol, Ms, R_min) for sol, R_min in zip(parabola_sols, R_mins)]
max_errs = np.max(np.abs(abs_errs), axis=1)

# Plot results
# --------------
min_seps = array([min_separation(sol, D) for sol in parabola_sols])

plt.plot(R_mins, R_mins - min_seps, 'ko')
plt.xlabel('R_min/m')
plt.ylabel('Error in R_min/m')
plt.savefig('Errors in min separations.pdf')
plt.show()

plt.plot(R_mins, max_errs)
plt.xlabel('Minimum separation/m')
plt.ylabel('Max error in x/m')
plt.savefig('Max error of parabolic orbit vs R_min.pdf')
plt.show()

i = 0    # looking at R_min = 11 since this had max error
end_time = 1000.0
t = timescale(end_time, step_size)

plt.plot(t, abs_errs[i])
plt.xlabel('t/s')
plt.ylabel('Error in x/m')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.savefig('Absolute error for parabola R_min=11')
plt.show()

# check variation of energy
Es = quantity_over_time(E, parabola_sols[i], Ms, D)

plt.plot(t, Es)
plt.xlabel('t/s')
plt.ylabel('Total energy/J')
plt.savefig('Energy over time for parabola R_min=11')
plt.show()

# compare orbit of first particle to analytical orbit for R_min=11
sol = parabola_sols[i]
R_min = R_mins[i]
xs = stack_sol(sol, D)[:, :D, 0].transpose()
analytic_xs = math_parabola(Ms, R_min, xs[1])

plt.plot(xs[0], xs[1], 'r', label='computed')
plt.plot(analytic_xs, xs[1], 'k:', label='analytical')
plt.legend(loc='best')
plt.xlabel('x/m')
plt.ylabel('y/m')
plt.savefig('Comparison of computed and analytic parabolic motion R=11.pdf')
plt.show()

# create snapshot for both particles
snapshot_parabola(sol, 0, t, 0, 2, [], D, 'snapshot - parabola R_min = 11')

# Find errors for different step sizes
# --------------
step_exps = np.linspace(-3, 0, num=14)
step_sizes = 10**step_exps

parabola_step_sols = [parabola_sol(R_min, step_size) for step_size in step_sizes]

max_errs_parab = [np.max(np.abs(parabola_errors(sol, Ms, R_min))) for sol in parabola_step_sols]

# Plot results
# --------------
plt.plot(step_exps, max_errs_parab, 'r+', markersize=10)
plt.xlabel('log(Step size/s)')
plt.ylabel('Maximum error/m')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.savefig('Max error of parabolic orbit vs step size.pdf')
plt.show()
