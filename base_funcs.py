"""
 SIMULATING INTERACTING GALAXIES AND THE FORMATION OF TIDAL TAILS

 base_funcs

 This script provides a basis for modelling (via stepwise integration using
 scipy's odeint) the time evolution of bodies interacting by gravity.

 Interacting galaxies are then represented by two massive bodies in parabolic
 orbit, with one surrounded by "stars" (test masses with no gravitational field),
 and the formation of tidal tails in this process can be tested and investigated for 
 different approach parameters (varying separation, masses, "halo" radius, and
 angle of approach between plane of stars and plane of parabola) using the various
 other python files in this project.

 NOTES:
 - The state of a system of particles is represented by 2D matrix where each 
   column (i) contains the position components then velocity components of
   the ith particle. e.g. for a 2D system of N particles:
   q_matrix = [[x0, ..., x(N-1)],
                [y0, ..., y(N-1)],
                [v_x0, ..., v_x(N-1)],
                [v_y0, ..., v_y(N-1)]]
   For compatibility, this must be flattened before being passed into g_diffs/odeint,
   and solutions of odeint will also be in flattened form.
 - Switching to flattened and matrix forms can be done with numpy.ndarray.flatten and
   stack, respectively.
 - for loops are used instead of vectorization when number of evaluations (e.g.
   number of massive bodies) is assumed small.
   
"""

# --------------
# Libraries and constants
# --------------

import numpy as np    # arrays, vectorization and mathematical objects
import time    # finding processing times
from scipy.integrate import odeint    # stepwise integration

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

# --------------
# Differentiation due to gravity
# --------------

def stack(q, n_D):
    # Puts flattened state vector (q) for (n_D)-dimensional system into matrix form
    return q.reshape(2 * n_D, -1)


def stack_sol(sol, n_D):
    # For solution (sol) of (n_D) dimensional simulation, put state vector at each time step into matrix form
    return sol.reshape(sol.shape[0], 2*n_D, -1)


def g_point(m, x, x1s):
    """
    Return grav. acceleration at positions (x1s) due to point mass (m) at (x)
     - x1s must be in matrix form
    """ 
    rs = x - x1s.transpose()    # separations
    inv_squares = G * m / mag(rs, axis=1)**3
    return inv_squares * rs.transpose()    # return in matrix form


def g_field(ms, xs, x1s, gfunc):
    """
     Return total grav. acceleration at positions (x1s) due to point masses (ms) at (xs)
     - xs and x1s must be in matrix form
    """
    
    gs = [gfunc(m, xs[:,i], x1s) for i, m in enumerate(ms)]    # can use loop since not many masses used
    return np.sum(gs, axis=0)


def ext_g_field(ms, xs, gfunc):
    """
    Return grav. acceleration of each mass in (ms) due to all other 
    ms at (xs) with (gfunc)
    """
    
    D = xs.shape[0]
    field = zeros(xs.shape)
    
    # use simple loop since len(ms) assumed small
    for i, m in enumerate(ms):
        ext_f = g_point(m, xs[:,i], np.delete(xs, i, 1))
        tot_f = np.insert(ext_f, i, zeros(D), axis=1)
        field += tot_f

    return field


def g_diffs(q, t, ms, gfunc, n_D):
    """
     Returns differentials w.r.t. time (t) of state vector (q) in flattened form.
     Accelerations are due to gravitational field (gfunc) of masses (ms) in (n_D)-space.
     - first len(ms) particles in q are assumed to be massive
     - gfunc must take args (m, x, x1)
    """
    
    n_massive = len(ms)    # no. of massive bodies
    
    # put into matrix form and extract positions, velocities
    q_matrix = stack(q, n_D)
    all_xs, all_vs = np.split(q_matrix, 2)
    
    # check trivial case of no massive bodies
    if n_massive == 0:
        all_as = zeros(all_xs.size)    # accelerations must be zero
        return np.append(all_vs.flatten(), all_as)
    
    # split into positions Xs of massive bodies and test_xs of test particles
    Xs, test_xs = np.hsplit(all_xs, [n_massive])
    
    # find acceleration due to grav. field of massive bodies
    if test_xs.shape[1] > 0:
        # no. of test particles potentially large, use vectorized process
        test_as = g_field(ms, Xs, test_xs, gfunc)
    else:
        test_as = np.empty([n_D, 0])
    
    # find acc. of each massive body due to grav. field of other massive bodies only
    if Xs.shape[1] > 1:
        As = ext_g_field(ms, Xs, gfunc)
    else:
        As = zeros((n_D, 1))
    
    # combine massive and test particle acceleration
    all_as = np.append(As, test_as, axis=1)
    
    # return diff. of positions (velocities) and of velocities (accs) in flattened form
    return np.append(all_vs, all_as).flatten()


# --------------
# Functions for analysis of solutions
# --------------

def get_xs(q, n_D):
    # Get positions of flattened state vector (q) for (n_D) space
    return stack(q, n_D)[:n_D]


def get_vs(q, n_D):
    # Get velocities of (q) for (n_D) space
    return stack(q, n_D)[n_D:]


def quantity_over_time(qfunc, sol, ms, n_D):
    """
     Evaluate quantity (qfunc) for each q in (sol) with massive bodies (ms) in (n_D) space
     - qfunc must take args (ms, q, n_D)
    """
    axis_func = lambda q : qfunc(ms, q, n_D)
    return np.apply_along_axis(axis_func, 1, sol)


def T_particle(m, v):
    # Kinetic energy of particle mass (m) with velocity (v)
    return 0.5 * m * mag(v)**2   


def T(ms, q, n_D):
    # Total kinetic energy of particles mass (ms) with velocities (vs)
    vs = get_vs(q, n_D)
    return np.sum( [T_particle(m, vs[:,i]) for i, m in enumerate(ms)] )


def V_field(ms, xs, x):
    # Grav. potential at (x) due to masses (ms) at (xs)
    return - np.sum( [ G * m / mag(xs[:,i] - x) for i, m in enumerate(ms) ] )


def U(ms, q, n_D):
    # Potential energy due to gravity (relative to infinity) due to masses (ms) in state (q) in (n_D) space
    
    xs = get_xs(q, n_D)
    return 0.5 * np.sum( [m * V_field(np.delete(ms,i), np.delete(xs,i,1), xs[:,i]) for i, m in enumerate(ms)] )


def E(ms, q, n_D):
    # Total energy of masses (ms) in state (q) in (n_D) space
    return T(ms, q, n_D) + U(ms, q, n_D)


def p(ms, q, n_D):
    # Total momentum of masses (ms) in state (q) in (n_D) space
    vs = get_vs(q, n_D)
    return np.sum( [m * vs[:,i] for i, m in enumerate(ms)] )


def L(ms, q, n_D):
    # Total angular momentum of masses (ms) in state (q) in (n_D) space
    xs, vs = np.split(stack(q, n_D), 2)
    return np.sum( [m * np.cross(xs[:,i], vs[:,i]) for i, m in enumerate(ms)] )


# --------------
# Initialising circular orbits
# --------------

def ring(R, N, s, C, v_C, n_D):
    """
     Return xs and vs in matrix form of ring radius (R) of (N) particles
     moving tangentially at speed (s) in xy-plane about centre (C), 
     which moves with velocity (v_C) in (n_D) space
    """
    
    # equally spaced angles between 0 and 2*pi
    thetas = np.linspace(0, 2*pi, N, False)
    
    # convert to sets of points and tangent velocities in xy-plane about stationary origin
    cos_thetas, sin_thetas = cos(thetas), sin(thetas)
    x0s = R * array([cos_thetas, sin_thetas])
    v0s = s * array([- sin_thetas, cos_thetas])
    
    # embed in (n_D) space
    padding = zeros((n_D - 2, N))
    padded_x0s = np.append(x0s, padding, axis=0)
    padded_v0s = np.append(v0s, padding, axis=0)
    
    # shift xs and vs to be about point C moving at v_C
    xs = padded_x0s + C.reshape(n_D, 1)
    vs = padded_v0s + v_C.reshape(n_D, 1)
    
    # combine into matrix form
    return np.append(xs, vs, axis=0)


def s_circular(m, R):
    # Speed of stable circular orbit at radius (R) about point mass (m)
    return np.sqrt(G * m / R)


def circle(m, R, N, anticloc, C, v_C, n_D):
    """
     Return q matrix of ring of particles in circular orbit about
     mass (m) using ring().
     Boolean anticloc determines if orbit it anticlockwise (True) or clockwise (False)
    """
    s = s_circular(m, R)
    sign = 1 if anticloc else -1
    return ring(R, N, (sign * s), C, v_C, n_D)


def initialise_stars(m, Rs, density, anticloc, C, v_C, n_D):
    """
     Return q matrix of circlular orbits of radii (Rs) about the same point
     with density*R particles per orbit, using circle().
    """
    star_circles = [circle(m, R, density*R, anticloc, C, v_C, n_D) for R in Rs]
    return np.concatenate(star_circles, axis=1)

# --------------
# Initialising parabolic orbits
# --------------

def parabola_vals(ms, R_0, R_min, n_D):
    """
     Return initial positions [X1, X2] and speeds [V1, V2] for masses 1 and 2
     in (ms), such that their orbit is parabolic with starting separation R_0
     and min separation (R_min), and is in the xy-plane of (n_D) space
     - given in [X1, X2, V1, V2] form rather than q matrix form for easier
       manipulation by e.g. rotations
    """
    
    D = 2    # constrained to xy plane
    
    # find formula coefficients
    m1, m2 = ms    # extract masses
    a1 = m2 / (m1 + m2)
    a2 = m1 / (m1 + m2)
    
    # X and V values for mass 1
    x1 = a1 * (R_0 - 2 * R_min)
    y1 = 2 * a1 * np.sqrt(R_min * (R_0 - R_min))
    vx1 = - np.sqrt( (2 * G * a1 * m2) / (R_0 * (1 + (R_min/(R_0 - R_min)))) )
    vy1 = - np.sqrt( (2 * G * a1 * m2) / (R_0 * (1 + (R_0 - R_min)/R_min)) )
    
    X1 = array([x1, y1])
    V1 = array([vx1, vy1])
    
    # X and V values for mass 2
    x2 = - a2 * (R_0 - 2 * R_min)
    y2 = - 2 * a2 * np.sqrt(R_min * (R_0 - R_min))
    vx2 = np.sqrt( (2 * G * a2 * m1) / (R_0 * (1 + (R_min/(R_0 - R_min)))) )
    vy2 = np.sqrt( (2 * G * a2 * m1) / (R_0 * (1 + (R_0 - R_min)/R_min)) )
    
    X2 = array([x2, y2])
    V2 = array([vx2, vy2])
    
    padding = np.zeros((2*D, n_D - D))    # embed in n_D space
    return np.append([X1, X2, V1, V2], padding, axis=1)


def initialise_parabola(ms, R_0, R_min, n_D):
    # Return q matrix for parabola values given by parabola_vals()
    
    X1, X2, V1, V2 = parabola_vals(ms, R_0, R_min, n_D)

    Xs = np.append(X1.reshape(n_D,1), X2.reshape(n_D,1), axis=1)
    Vs = np.append(V1.reshape(n_D,1), V2.reshape(n_D,1), axis=1)
    
    return np.append(Xs, Vs, axis=0)