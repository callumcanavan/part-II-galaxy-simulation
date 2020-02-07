"""
 SIMULATING INTERACTING GALAXIES AND THE FORMATION OF TIDAL TAILS

 time_test

 This script provides a function for calculating g for a group of point
 masses without vectorization (i.e. using loops) and compares the time it takes
 to simulate with odeint using this funciton vs the vectorized implementation.
 
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
mag = np.linalg.norm
# scaling of universe
G = 1    # gravitational constance
M = 1    # reference mass of bodies

# project functions
from simulate_long import initialise_galaxies

# Comparing run times to unvectorised code
# --------------
def g_point_loop(m, x, x1s):
    # Calculate g as in g_point but with loops
    
    gs = np.zeros(x1s.shape)
    for i in range(x1s.shape[1]):
        r = x - x1s[:,i]
        gs[:,i] = (G * m / (mag(r)**3)) * r
    return gs


# standard setup (exc R_min, and with density of 12)
D = 2
step_size = 0.04
end_time = 150.0
t = timescale(end_time, step_size)
R_0 = 30
Ms = [M, M]
density = 12
anitcloc = True
Rs = np.arange(2,7)

d12_loop_processing_times = []
d12_vec_processing_times = []

for R_min in R_mins:
    # initialise
    galaxies = initialise_galaxies(Ms, Rs, density, anticloc, R_0, R_min, D)
    q0 = galaxies.flatten()
    
    # simulate using looping code for g
    p_t0 = time.time()    # time before simulation
    loop_sol = odeint(g_diffs, q0, t, args=(Ms, g_point_loop, D))    
    p_t1 = time.time()    # time after simulation
    d12_loop_processing_times += [p_t1 - p_t0]    # processing time
    
    # save solution to check correctness
    title = 'loop d=12 collision solutions - R_min = ' + str(R_min) + '.txt'
    np.savetxt(title, col_sol[0::interval])
    
    # simulate using looping code for g
    p_t0 = time.time()    # time before simulation
    loop_sol = odeint(g_diffs, q0, t, args=(Ms, g_point, D))    
    p_t1 = time.time()    # time after simulation
    d12_vec_processing_times += [p_t1 - p_t0]    # processing time
    
    # save solution
    title = 'vector d=12 collision solutions - R_min = ' + str(R_min) + '.txt'
    np.savetxt(title, col_sol[0::interval])

# load vectorised processing times for density=200
d200_vec_processing_times = np.loadtxt('processing times, separation variation')

plt.plot(R_mins, d12_loop_processing_times, 'r+', markersize=10, label= r'loop, d=12m$^{-1}$')
plt.plot(R_mins, d200_vec_processing_times, 'k+', markersize=10, label= r'vectorised, d=200m$^{-1}$')
plt.plot(R_mins, d12_vec_processing_times, 'b+', markersize=10, label= r'vectorised, d=12m$^{-1}$')
plt.legend(loc='best', prop = {'size' : 10})
plt.xlabel(r'$R_{min}$/m')
plt.ylabel('Processing time/s')
plt.savefig('Processing times for vectorised and unvectrorised code.pdf')
plt.show()