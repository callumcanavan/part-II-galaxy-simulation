"""
 SIMULATING INTERACTING GALAXIES AND THE FORMATION OF TIDAL TAILS

 simulate_long

 This script provides functions for initialising the interaction between two
 galaxies with point mass centres in planar parabolic orbit (referred to
 here as the "standard setup"), simulates their interaction over a long
 period of time and displays their resulting paths.

 Further simulations then experiment with varying separation of galaxies,
 varying galaxy mass ratios, and reversing the orbit of stars wrt the orbit
 of parabolic orbit between galaxies.
 
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
# scaling of universe
G = 1    # gravitational constance
M = 1    # reference mass of bodies

# project functions
from base_funcs import initialise_parabola, initialise_stars
from display_funcs import snapshot_parabola, animate_2D

# --------------
# Interacting galaxies over 300s in standard setup
# --------------

def initialise_galaxies(ms, Rs, density, anticloc, R_0, R_min, n_D):
    """
     Return q matrix for parabolic orbit (ms, R_0, R_min) with circular
     rings (ms[0], Rs, density, anticloc) of particles about the first
     mass in (n_D) space
    """
    
    # initialise parabolic orbit
    black_holes = initialise_parabola(ms, R_0, R_min, n_D)
    
    # initialise circular rings about first particle
    C = black_holes[:n_D, 0]
    v_C = black_holes[n_D:, 0]
    stars = initialise_stars(ms[0], Rs, density, anticloc, C, v_C, n_D)
    
    galaxies = np.concatenate((black_holes, stars), axis=1)
    return galaxies


# Simulate for standard setup
# --------------
D = 2
step_size = 0.04    # step_size used for rest of experiment
end_time = 350.0
t = timescale(end_time, step_size)

# standard setup:
Ms = [M, M]
Rs = np.arange(2,7)
density = 200
anticloc = True
R_0 = 60
R_min = 12

galaxies = initialise_galaxies(Ms, Rs, density, anticloc, R_0, R_min, D)
q0 = galaxies.flatten()

long_sol = odeint(g_diffs, q0, t, args=(Ms, g_point, D))

# save num_frames of solution for possible snapshots/animation later
num_frames = 150
interval = int(len(t) / num_frames)
np.savetxt('standard collision over 350s.txt', long_sol[0::interval])

# save animation
animate_2D(long_sol, num_frames, [], D, 'Animation - standard collision over 350s')

A = 0
n_paths = 2
outer_lims = [-32, 32, -32, 32]
snapshot_parabola(long_sol, 0, t, A, n_paths, outer_lims, D, 'snapshot - long sol initial')

inner_lims = [-20, 20, -20, 20]
snapshot_parabola(long_sol, 4400, t, A, 2, inner_lims, D, 'snapshot - long sol bulge')
snapshot_parabola(long_sol, 4750, t, A, n_paths, inner_lims, D, 'snapshot - long sol join')
snapshot_parabola(long_sol, 5100, t, A, n_paths, inner_lims, D, 'snapshot - long sol form')
snapshot_parabola(long_sol, 5300, t, A, n_paths, inner_lims, D, 'snapshot - long sol tail')
snapshot_parabola(long_sol, 5600, t, A, n_paths, inner_lims, D, 'snapshot - long sol overtake')
snapshot_parabola(long_sol, len(t)-1, t, A, n_paths, [-45, 45, -55, 35 ], D, 'snapshot - long tail spread')

# --------------
# Interacting galaxies of varying R_mins
# --------------

# standard setup (exc R_min)
D = 2
step_size = 0.04
end_time = 150.0
t = timescale(end_time, step_size)
R_0 = 30
Ms = [M, M]
density = 200
anitcloc = True
Rs = np.arange(2,7)

# no of frames for saving
num_frames = 100
interval = int(end_time/(step_size*num_frames))

# vary R_min between 12 and 16 in 0.5 steps
R_mins = np.linspace(12, 16, 11, True)

# also find processing time of each simulation
processing_times = []

for R_min in R_mins:
    # initialise
    galaxies = initialise_galaxies(Ms, Rs, density, anticloc, R_0, R_min, D)
    q0 = galaxies.flatten()
    
    p_t0 = time.time()    # time before simulation
    
    col_sol = odeint(g_diffs, q0, t, args=(Ms, g_point, D))
  
    p_t1 = time.time()    # time after simulation
    processing_times += [p_t1 - p_t0]    # processing time
    
    # save solution
    title = 'collision solutions - R_min = ' + str(R_min) + '.txt'
    np.savetxt(title, col_sol[0::interval])

    np.savetxt('processing times, separation variation', processing_times)    # save processing times

# Plot and animate
# --------------    
plt.plot(R_mins, processing_times, '+', markersize=10)
plt.xlabel('Minimum separation/m')
plt.ylabel('Processing time/s')
plt.savefig('Processing time vs min separation.pdf')
plt.show()

# animate each sol
min_sep_sols = [np.loadtxt('collision solutions - R_min = ' + str(R_min) + '.txt') for R_min in R_mins]

ax_lims = [-35, 25, -35, 25]    # same ax lims used for all animations to be measure for consistency
for i, R_min in enumerate(R_mins):
    animate_2D(min_sep_sols[i], num_frames, ax_lims, D, 'Animation - R_min = ' + str(R_mins[i]))

A = 0
t_frames = t[0::interval]
inner_lims = [-20, 20, -20, 20]
snapshot_parabola(min_sep_sols[0], 70, t_frames, A, n_paths, inner_lims, D, 'snapshot - R_min=12')
snapshot_parabola(min_sep_sols[10], 70, t_frames, A, n_paths, inner_lims, D, 'snapshot - R_min=16')

# import measured tail lengths
measured_lengths_rmin = np.loadtxt('Tail length measurements - variation in R_min')
tail_lengths, errors = measured_lengths_rmin

# calculate gradient and estimate error by finding min/max gradient approximately within error bars
l = len(tail_lengths) - 1

m_max = (tail_lengths[l] - errors[l] - tail_lengths[0] - errors[0]) / (R_mins[l] - R_mins[0])
intercept_max = tail_lengths[0] + errors[0] - m_max * R_mins[0]

m_min = (tail_lengths[l] + errors[l] - tail_lengths[0] + errors[0]) / (R_mins[l] - R_mins[0])
intercept_min = tail_lengths[0] - errors[0] - m_min * R_mins[0]

m_avg = (m_max + m_min) / 2
intercept_avg = (intercept_max + intercept_min) / 2
m_error = (m_min - m_max) / 2

plt.errorbar(R_mins, m_avg*R_mins + intercept_avg, yerr = errors, capsize=5)
plt.figtext(0.15, 0.2, "gradient = " + str(np.round(m_avg)) + r'$\pm$' + str(np.round(m_error)))
plt.xlabel('Minimum separation/m')
plt.ylabel('Tail length/m')
plt.savefig('Variation of tail length vs R_min.pdf')
plt.show()

# check that min/max gradients are approximately within errors
plt.errorbar(R_mins, m_avg*R_mins + intercept_avg, yerr = errors, capsize=5)
plt.plot(R_mins, m_max*R_mins + intercept_max, 'r')
plt.plot(R_mins, m_min*R_mins + intercept_min, 'r')
plt.figtext(0.15, 0.2, "gradient = " + str(np.round(m_avg)) + r'$\pm$' + str(np.round(m_error)))
plt.xlabel('Minimum separation/m')
plt.ylabel('Tail length/m')
plt.savefig('Minmax errors for tail length vs R_min.pdf')
plt.show()

# --------------
# Interacting galaxies with varying mass ratio
# --------------

# Standard setup (exc masses)
D = 2
step_size = 0.04
end_time = 150.0
t = timescale(end_time, step_size)

Rs = np.arange(2,7)
density = 200
anticloc = True
R_0 = 30
R_min = 12

M2s = M * np.linspace(0.5, 1.1, 7)    # mass range of perturbing body

num_frames = 100
interval = int(len(t) / num_frames)

for M2 in M2s:
    # initialise
    Ms = [M, M2]    # centre of galaxy mass fixed, perturbing mass varied
    galaxies = initialise_galaxies(Ms, Rs, density, anticloc, R_0, R_min, D)
    q0 = galaxies.flatten()
    
    # solve
    m_sol = odeint(g_diffs, q0, t, args = (Ms, g_point, D))
    
    # save
    title = 'collision solutions - M2 = ' + str(M2) + '.txt'
    np.savetxt(title, m_sol[0::interval])

    # animate
mass_sols = [np.loadtxt('collision solutions - M2 = ' + str(M2) + '.txt') for M2 in M2s]

ax_lims = [-35, 25, -35, 25]
num_frames = 100
for i, M2 in enumerate(M2s):
    animate_2D(mass_sols[i], num_frames, ax_lims, D, 'Animation - M2 = ' + str(M2))

# snapshots
A = 0
step = 80
snapshot_parabola(mass_sols[0], step, t, A, n_paths, inner_lims, D, 'snapshot - M2 = 0.5')
snapshot_parabola(mass_sols[6], step, t, A, n_paths, outer_lims, D, 'snapshot - M2 = 1.1')

# Errors
# --------------

# measured tail lengths
tail_lengths = np.array([0.944, 0.849, 0.777, 0.68, 0.646, 0.534, 0.529, 0.487])
errors = np.array([0.007,0.009,0.016,0.017,0.027,0.030,0.040,0.049])
M2s = np.array([0.015, 0.042, 0.074, 0.101, 0.137, 0.173, 0.203, 0.223])

# calculate gradient and estimate error by finding min/max gradient approximately within error bars
l = len(tail_lengths) - 1

m_max = (tail_lengths[l] + errors[l] - tail_lengths[0] + errors[0]) / (M2s[l] - M2s[0])
intercept_max = tail_lengths[0] + errors[0] - m_min * M2s[0]

m_min = (tail_lengths[l] - errors[l] - tail_lengths[0] - errors[0]) / (M2s[l] - M2s[0])
intercept_min = tail_lengths[0] - errors[0] - m_max * M2s[0]

m_avg = (m_max + m_min) / 2
intercept_avg = (intercept_max + intercept_min) / 2
m_error = (m_max - m_min) / 2

plt.errorbar(M2s, m_avg*M2s + intercept_avg, yerr = errors, capsize=5)
plt.figtext(0.4, 0.2, "gradient = " + str(np.round(m_avg)) + r'$\pm$' + str(np.round(m_error)) + r'mkg$^{-1}$')
plt.xlabel('Mass of perturber/kg')
plt.ylabel('Tail length/m')
plt.savefig('Variation of tail length vs M2.pdf')
plt.show()

# check that min/max gradients are approximately within errors
plt.errorbar(M2s, m_avg*M2s + intercept_avg, yerr = errors, capsize=5)
plt.plot(M2s, m_max*M2s + intercept_min, 'r')
plt.plot(M2s, m_min*M2s + intercept_max, 'r')
plt.xlabel('Mass of perturber/m')
plt.ylabel('Tail length/m')
plt.savefig('Minmax errors for tail length vs M2.pdf')
plt.show()

# --------------
# Interacting galaxies with reversed star orbit
# --------------

# standard setup (exc anticloc)
D = 2
step_size = 0.04
end_time = 150.0
t = timescale(end_time, step_size)

Ms = [M, M]
Rs = np.arange(2,7)
density = 200
R_0 = 30
R_min = 12

anticloc = False    # stars orbit about galaxy clockwise rather than anticlockwise

galaxies = initialise_galaxies(Ms, Rs, density, anticloc, R_0, R_min, D)
q0 = galaxies.flatten()

rev_sol = odeint(g_diffs, q0, t, args=(Ms, g_point, D))

num_frames = 100
interval = int(end_time/(step_size*num_frames))
np.savetxt('collision solutions - reversed star orbit', rev_sol[0::interval])

animate_2D(rev_sol, num_frames, ax_lims, D, 'Animation - reversed star orbit')

A = 0
snapshot_parabola(rev_sol, 1500, t, A, n_paths, inner_lims, D, 'snapshot - reversed orbit pt 1')
snapshot_parabola(rev_sol, 2300, t, A, n_paths, inner_lims, D, 'snapshot - reversed orbit pt 2')
snapshot_parabola(rev_sol, 3400, t, A, n_paths, inner_lims, D, 'snapshot - reversed orbit pt 3')