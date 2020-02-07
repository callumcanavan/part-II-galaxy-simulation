"""
 SIMULATING INTERACTING GALAXIES AND THE FORMATION OF TIDAL TAILS

 simulate_3D

 This script provides functions for initialising, simulating and testing
 the interaction between galaxies in which the orbital plane of stars in
 one galaxy is rotated wrt the plane in which the two galaxies orbit
 about each other.
 
 Further simulations then experiment with varying the angle of rotation,
 and displays their resulting paths.
 
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
zeros = np.zeros
# scaling of universe
G = 1    # gravitational constance
M = 1    # reference mass of bodies

# project functions
from base_funcs import stack_sol, parabola_vals, initialise_stars, initialise_parabola
from display_funcs import snapshot_3D, snapshot_projection, animate_3D

# --------------
# Interacting galaxies in 3D
# --------------

def rot_matrix(theta, ax):
    """
     Generates 3x3 matrix which, when dotted with vector in 3D, rotates it
     anticlockwise by (theta) radians about axis (ax)
     - ax must be 0 (for x), 1 (for y) or 2 (for z)
    """
    D = 3    # for rotations in 3D
    rot_M = np.zeros((D, D))
    
    c = cos(theta)
    s = sin(theta)
    
    # set values for matrix, using mods to ensure positions remain within
    # 3D using right-handed configuration
    rot_M[ax, ax] = 1
    rot_M[(ax+1) % D, (ax+1) % D] = c
    rot_M[(ax+2) % D, (ax+2) % D] = c
    rot_M[(ax+2) % D, (ax+1) % D] = s
    rot_M[(ax+1) % D, (ax+2) % D] = -s
    
    return rot_M


def rotate(p, theta, ax):
    # Rotates 3D vector (p) (theta) radians anticlockwise about axis (ax)
    
    rot_M = rot_matrix(theta, ax)
    return np.dot(rot_M, p)


def initialise_rotated_parabola(ms, R_0, R_min, theta, ax):
    """
     Return q matrix for (R_0, R_min) parabolic orbit of (ms), rotated
     angle (theta) about axis (ax) from the xy plane
    """
    D = 3    # for 3D simulations
    plane_vals = parabola_vals(ms, R_0, R_min, D)    # vals in xy plane
    
    # rotate values
    X1, X2, V1, V2 = [rotate(val, theta, ax) for val in plane_vals]
    
    # put into matrix form
    Xs = np.concatenate((X1.reshape(D,1), X2.reshape(D,1)), axis=1)
    Vs = np.concatenate((V1.reshape(D,1), V2.reshape(D,1)), axis=1)
     
    return np.append(Xs, Vs, axis=0)


def initialise_rotated_galaxies(ms, Rs, density, anticloc, R_0, R_min, theta, ax):
    """
     Initialise galaxies (ms, Rs, density, anticloc, R_0, R_min) but with parabolc
     orbit rotated (theta) radians about axis (ax)
    """
    D = 3    # for 3D simulations
    
    # initialise rotated parabolic orbit of first two bodies
    black_holes = initialise_rotated_parabola(ms, R_0, R_min, theta, ax)
    
    # initialise star orbit about first particle parallel to xy plane
    C = black_holes[:D, 0]
    v_C = black_holes[D:, 0]
    stars = initialise_stars(ms[0], Rs, density, anticloc, C, v_C, D)
    
    # return in matrix form
    galaxies = np.concatenate((black_holes, stars), axis=1)
    return galaxies


def plane_angle(sol, axis):
    # Calculate angle of separation of first two particles in (sol) relative to (axis)
    
    D = 3
    unit = zeros(D)
    unit[axis] = 1
    
    x1s = stack_sol(sol, D)[:, :D, 0]
    x2s = stack_sol(sol, D)[:, :D, 1]
    rs = x2s - x1s
    Rs = mag(rs, axis=1)
    
    dotted = np.dot(rs, unit)
    
    return np.arccos(dotted / Rs)


def rotated_errors(sol, axis, angle):
    # Find error over time of (sol)'s angle with (axis) with respect to expected (angle)
    return plane_angle(sol, axis) - angle


# Test 3D g_diffs in xy plane
# --------------

# standard setup parabola (in 3D)
D = 3

step_size = 0.04
end_time = 1000.0
t = timescale(end_time, step_size)

Ms = [M, M]
R_0 = 125
R_min = 12

black_holes_xy = initialise_parabola(Ms, R_0, R_min, D)
q0 = black_holes_xy.flatten()
xy_sol = odeint(g_diffs, q0, t, args=(Ms, g_point, D))

snapshot_3D(xy_sol, 0, t, [], '3D test, no rotation')

# Test rotation
# --------------
axes = [0, 1]
theta = pi / 2
angle_sols = []
for axis in axes:
    # rotate about x axis
    black_holes_ang = initialise_rotated_parabola(Ms, R_0, R_min, theta, axis)
    q0 = black_holes_ang.flatten()
    
    # solve
    sol = odeint(g_diffs, q0, t, args=(Ms, g_point, D))
    angle_sols += [sol]
    
    snapshot_3D(sol, 0, t, [], 'rotation test, axis = ' + str(axis))
    
    ref_ax = 1 if (axis==0) else 0    # plane should be perp. to axis not rotated about
    ang_errs = rotated_errors(sol, ref_ax, pi/2)    
    plt.plot(t, ang_errs)
    plt.xlabel('t/s')
    plt.ylabel('Angle error/radians')
    #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.savefig('Angle error for rotation axis = ' + str(axis) + '.pdf')
    plt.show()


# Investigation
# --------------

# standard setup (in 3D)
D = 3

step_size = 0.04
end_time = 150.0
t = timescale(end_time, step_size)

Ms = [M, M]
Rs = np.arange(2,7)
density = 200
anticloc = True
R_0 = 30
R_min = 12

# initialise
galaxies = initialise_rotated_galaxies(Ms, Rs, density, anticloc, R_0, R_min, 0, 0)
q0 = galaxies.flatten()

# solve
rot0_sol = odeint(g_diffs, q0, t, args=(Ms, g_point, D))

# save and animate
num_frames = 150
interval = int(len(t) / num_frames)

np.savetxt('3D collision solutions - standard.txt', rot0_sol[0::interval])

ax_lims_3D = [-30, 30, -30, 30, -30, 30]    # standard ax lims for 3D animations
animate_3D(rot0_sol, num_frames, ax_lims_3D, '3D animation - no rotation')

# Varying rotation of parabola about x and y axes
# --------------
ns = np.arange(1,5)    # ceofficients of (pi/2) for rotation angles

# simulate for each rotation
for n in ns:
    theta = n * pi / 4
    
    # rotate theta about x axis then solve and save
    ax = 0
    galaxies = initialise_rotated_galaxies(Ms, Rs, density, anticloc, R_0, R_min, theta, ax)
    q0 = galaxies.flatten()
    rot_x_sol = odeint(g_diffs, q0, t, args=(Ms, g_point, D))
    title = '3D collision solutions - x rotation, n = ' + str(n) + '.txt'
    np.savetxt(title, rot_x_sol[0::interval])
    
    # rotate theta about y axis then solve and save
    ax = 1
    galaxies = initialise_rotated_galaxies(Ms, Rs, density, anticloc, R_0, R_min, theta, ax)
    q0 = galaxies.flatten()
    rot_y_sol = odeint(g_diffs, q0, t, args=(Ms, g_point, D))
    title = '3D collision solutions - y rotation, n = ' + str(n) + '.txt'
    np.savetxt(title, rot_y_sol[0::interval])

# animate in 3D
rotated_x_sols = [np.loadtxt('3D collision solutions - x rotation, n = ' + str(n) + '.txt') for n in ns]
rotated_y_sols = [np.loadtxt('3D collision solutions - y rotation, n = ' + str(n) + '.txt') for n in ns]
for i, n in enumerate(ns):
    animate_3D(rotated_x_sols[i], num_frames, ax_lims_3D, '3D animation - x rotation, n = ' + str(n))
    animate_3D(rotated_y_sols[i], num_frames, ax_lims_3D, '3D animation - y rotation, n = ' + str(n))

# animate 2D projections in xy plane
D = 3
ax_lims = [-35, 25, -35, 25]
animate_2D(rot0_sol, num_frames, ax_lims, D, 'Animation - 2D projection, standard')
for i, n in enumerate(ns):
    animate_2D(rotated_x_sols[i], num_frames, ax_lims, D, 'Animation - 2D projection, x rotation, n = ' + str(n))
    animate_2D(rotated_y_sols[i], num_frames, ax_lims, D, 'Animation - 2D projection, y rotation, n = ' + str(n))

# snapshots
no_rotation_sol = np.loadtxt('3D collision solutions - standard.txt')
step = len(no_rotation_sol) - 1
inner_lims_3D = [-20, 20, -20, 20, -20, 20]
t_snap = t[0::interval]
snapshot_3D(no_rotation_sol, step, t_snap, inner_lims_3D, '3D snapshot - no rotation')

step = 149
for i, n in enumerate(ns):
    snapshot_3D(rotated_x_sols[i], step, t_snap, inner_lims_3D, '3D snapshot - x rotation, n = ' + str(n))
    snapshot_3D(rotated_y_sols[i], step, t_snap, inner_lims_3D, '3D snapshot - y rotation, n = ' + str(n))

# 2D projections of n=1 x rotation
step = 149
A = 0
helix_sol = rotated_x_sols[0]

# to find lims which place galaxy centre at origin for each of 3 projections
centre = stack_sol(helix_sol, D)[step, :D, 0]
individual_lims = [[x-20, x+20] for x in centre]
project_lims = [np.concatenate(np.delete(individual_lims, i, 0))for i in range(D)]

axes = np.arange(3)
for axis in axes:
    snapshot_projection(helix_sol, axis, step, t_snap, project_lims[axis], 'snapshot projection - helix ' + str(axis))