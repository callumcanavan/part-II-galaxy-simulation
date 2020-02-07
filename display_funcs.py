"""
 SIMULATING INTERACTING GALAXIES AND THE FORMATION OF TIDAL TAILS

 display_funcs

 This script provides functions for the animated and still-image display
 of paths traced out by solutions of odeint over time.
   
"""

# --------------
# Libraries and constants
# --------------

import numpy as np    # arrays, vectorization and mathematical objects

# plots, snapshots and animation
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D    # 3D plots

# plotting parameters
matplotlib.rc('xtick', labelsize=12) 
matplotlib.rc('ytick', labelsize=12) 
matplotlib.rcParams.update({'font.size': 15})

from base_funcs import stack_sol

# --------------
# Animate
# --------------

def animate_2D(sol, n_frames, ax_lims, n_D, filename):
    """
     Save animation of (n_D)-dimensional (sols) as (filename).html
     within (ax_lims) over (n_frames)
     - if n_D > 2, the system is projected into the xy-plane
    """
    # extract x and y coordinates of system over whole simulation
    p = stack_sol(sol, n_D)
    xs, ys = p[:, 0], p[:, 1]
    
    plt.close("all")
    fig, ax = plt.subplots()
    
    # get axes limits (or create them if not specified)
    if np.size(ax_lims) == 0:
        # capture all existing points with at least 100(rb-1)% whitespace on all sides
        rb = 1.25
        min1 = min2 =  rb * np.min([xs, ys])
        max1 = max2 = rb * np.max([xs, ys])
    else:
        min1, max1, min2, max2 = ax_lims
    
    # set axes
    plt.xlim(min1, max1)
    plt.ylim(min2, max2)
    plt.xlabel('x/m')
    plt.ylabel('y/m')
    plt.gca().set_aspect('equal', adjustable='box')    # ensure equal scaling of axes
    fr, = ax.plot([], [], 'bo', markersize=0.1)
    
    inter = int( sol.shape[0] / n_frames )    # number of steps between each frame
    
    # function specifying how a given frame is generated
    def next_frame(i):
        # update coords of each particle
        j = inter*i
        fr.set_data(xs[j], ys[j])
        return fr
    
    # animate and save
    ani = animation.FuncAnimation(fig, next_frame, frames=n_frames, interval=20)

    ani.save(filename + '.html')
    plt.close()


def animate_3D(sol, n_frames, ax_lims, filename):
    """
     Save 3D animation of (sols) [assumed 3D] as (filename).html
     within (ax_lims) over (n_frames)
    """
    # extract x, y, z coordinates of system over whole simulation
    D = 3
    p = stack_sol(sol, D)
    xs, ys, zs = p[:, 0], p[:, 1], p[:,2]
    
    # get axes limits (or create them if not specified)
    if np.size(ax_lims) == 0:
        # capture all points with least whitespace possible
        xmax = ymax = zmax = np.max(np.abs([xs, ys, zs]))
        xmin = ymin = zmin = - xmax
    else:
        xmin, xmax, ymin, ymax, zmin, zmax = ax_lims
        
    
    plt.close("all")
    fig = plt.figure()
    
    # set axes
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel('x/m', fontsize=12)
    ax.set_ylabel('y/m', fontsize=12)
    ax.set_zlabel('z/m', fontsize=12)
    plt.gca().set_aspect('equal', adjustable='box')    # ensure equal scaling of axes
    fr, = ax.plot([], [], 'bo', markersize=0.1)
    
    # plot path over time of first two particles
    ax.plot(xs[:,0], ys[:,0], zs[:,0], 'r-', linewidth = 0.3)
    ax.plot(xs[:,1], ys[:,1], zs[:,1], 'r-', linewidth = 0.3)
    
    inter = int( sol.shape[0] / n_frames )    # number of steps between each frame
    
    # function specifying how a given frame is generated
    def next_frame(i):
        # update coords of each particle
        j = inter*i
        fr.set_data(xs[j], ys[j])
        fr.set_3d_properties(zs[j])
        return fr
    
    # animate and save
    ani = animation.FuncAnimation(fig, next_frame, frames=n_frames, interval=20)

    ani.save(filename + '.html')
    plt.close()


# --------------
# Snapshot
# --------------

def snapshot_circle(sol, step, t, A, ax_lims, n_D, filename):
    """
     Plot snapshot of particles in (sol) at frame (step) in time (t)
     (A) is radius of first particle halo (set to 0 if point mass)
     Plot within (ax_lims) and save as (filename).pdf
     - Also plots path of all particles apart from the first over t
     - projects particles into xy-plane if n_D > 2
    """
    D = 2
    
    p = stack_sol(sol, n_D)
    xs, ys = p[:, 0], p[:, 1]    # xs and ys over all time
    
    # xs and ys at step
    X, Y = xs[step, 0], ys[step, 0]    # centre
    test_x, test_y = xs[step, 1:], ys[step, 1:]    # test particles
    
    # get axes limits (or create them if not specified)
    if np.size(ax_lims) == 0:
        rb = 1.25 # plot captures all existing points with 100(rb-1)% whitespace on all sides
        min1 = min2 =  rb * np.min([xs, ys])
        max1 = max2 = rb * np.max([xs, ys])
    else:
        min1, max1, min2, max2 = ax_lims
    
    fig, ax = plt.subplots()
    
    plt.gca().set_aspect('equal', adjustable='box')    # ensure equal scaling of axes
    plt.xlim(min1, max1)
    plt.ylim(min2, max2)
    ax.scatter(X, Y, c='r', s=40)
    ax.scatter(test_x, test_y, c='b', s=20)
    ax.plot(xs[:,1:], ys[:,1:], 'b', linewidth=0.1)    # plot path of all but first particle
    
    # plot halo of radius A around first particle
    circle1 = plt.Circle((X, Y), 0.2, facecolor='none', edgecolor='r', linewidth=0.5)
    circle1.set_radius(A)
    ax.add_artist(circle1)
    
    plt.title('t = ' + str(round(t[step], 1)) + 's')
    plt.xlabel('x/m')
    plt.ylabel('y/m')
    plt.savefig(filename + '.pdf')
    plt.show()


def snapshot_parabola(sol, step, t, A, num_paths, ax_lims, n_D, filename):
    """
     Plot snapshot of particles in (sol) at frame (step) in time (t)
     (A) is radius of first two particles' halos (set to 0 if point mass)
     Plot within (ax_lims) and save as (filename).pdf
     - Also plots path of first (num_paths) particles over t
     - projects particles into xy-plane if n_D > 2
    """
    D = 2
    
    p = stack_sol(sol, n_D)    # xs and ys over all time
    xs, ys = p[:, 0], p[:, 1]
    
    X, Y = xs[step, :2], ys[step, :2]    # centres
    test_x, test_y = xs[step, 2:], ys[step, 2:]    # test particles
    
    if np.size(ax_lims) == 0:
        rb = 1.25 # plot captures all existing points with 100(rb-1)% whitespace on all sides
        min1 = min2 =  rb * np.min([xs[step], ys[step]])
        max1 = max2 = rb * np.max([xs[step], ys[step]])
    else:
        min1, max1, min2, max2 = ax_lims
    
    fig, ax = plt.subplots()
    
    plt.gca().set_aspect('equal', adjustable='box')    # ensure equal scaling of axes
    plt.xlim(min1, max1)
    plt.ylim(min2, max2)
    ax.scatter(X, Y, c='r', s=20)
    ax.scatter(test_x, test_y, c='b', s=0.1)
    
    # plot path of first num_paths particles
    for i in range(num_paths):
        ax.plot(xs[:,i], ys[:,i], 'r', linewidth=0.5)
    
    # plot halo of radius A around first two particles
    circle1 = plt.Circle((X[0], Y[0]), 0.2, facecolor='none', edgecolor='r', linewidth=0.9, linestyle='--')
    circle1.set_radius(A)
    ax.add_artist(circle1)
    
    circle2 = plt.Circle((X[1], Y[1]), 0.2, facecolor='none', edgecolor='r', linewidth=0.9, linestyle='--')
    circle2.set_radius(A)
    ax.add_artist(circle2)
    
    plt.title('t = ' + str(round(t[step], 1)) + 's')
    plt.xlabel('x/m')
    plt.ylabel('y/m')
    plt.savefig(filename + '.pdf')
    plt.show()


def snapshot_3D(sol, step, t, ax_lims, filename):
    """
     Plot 3D snapshot of particles in (sol) at frame (step) in time (t)
     Plot within (ax_lims) and save as (filename).pdf
     - sol is assumed to be 3D
    """
    D = 3
    
    p = stack_sol(sol, D)    # xs, ys, zs over all time
    xs, ys, zs = p[:, 0], p[:, 1], p[:, 2]
    
    X, Y, Z = xs[step, :2], ys[step, :2], zs[step, :2]    # centres
    test_x, test_y, test_z = xs[step, 2:], ys[step, 2:], zs[step, 2:]    # test particles
    
    # get ax_lims or generate if none given
    if np.size(ax_lims) == 0:
        # capture all points with least whitespace possible
        xmax = ymax = zmax = np.max(np.abs([xs, ys, zs]))
        xmin = ymin = zmin = - xmax
    else:
        xmin, xmax, ymin, ymax, zmin, zmax = ax_lims
    
    plt.close("all")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')    # 3D plot
    
    # set axes
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel('x/m', fontsize=12)
    ax.set_ylabel('y/m', fontsize=12)
    ax.set_zlabel('z   ', fontsize=12)
    plt.tick_params(labelsize=10)
    
    ax.scatter(X, Y, Z, c='r', s=10)
    ax.scatter(test_x, test_y, test_z, c='b', s=0.8)
    
    plt.gca().set_aspect('equal', adjustable='box')    # ensure equal scaling of axes
    
    # plot path of first two particles
    ax.plot(xs[:,0], ys[:,0], zs[:,0], 'r', linewidth=0.4)
    ax.plot(xs[:,1], ys[:,1], zs[:,1], 'r', linewidth=0.4)
    
    plt.title('t = ' + str(round(t[step], 1)) + 's')
    plt.savefig(filename + '.pdf')
    plt.show()


def snapshot_projection(sol, axis, step, t, ax_lims, filename):
    """
     Plot 2D snapshot of 3D (sol) after removing components parallel to (axis)
     - e.g. axis=1 gives projection in yz plane
    """
    D = 3
    
    p = stack_sol(sol, D)    
    pro_p = np.delete(p, axis,  1)    # remove axis component of position
    x1s, x2s = pro_p[:, 0], pro_p[:, 1]    # x1s and x2s over all time
    
    X1, X2 = x1s[step, :2], x2s[step, :2]    # centres
    test_x1, test_x2 = x1s[step, 2:], x2s[step, 2:]    # test particles
    
    if np.size(ax_lims) == 0:
        rb = 1.25 # plot captures all existing points with 100(rb-1)% whitespace on all sides
        min1 = min2 =  rb * np.min([x1s[step], x2s[step]])
        max1 = max2 = rb * np.max([x1s[step], x2s[step]])
    else:
        min1, max1, min2, max2 = ax_lims
    
    fig, ax = plt.subplots()
    
    plt.gca().set_aspect('equal', adjustable='box')    # ensure equal scaling of axes
    plt.xlim(min1, max1)
    plt.ylim(min2, max2)
    ax.scatter(X1, X2, c='r', s=20)
    ax.scatter(test_x1, test_x2, c='b', s=0.1)
    
    plt.title('t = ' + str(round(t[step], 1)) + 's')
    
    # plot axes titles depending on projection plane
    if axis==0:
        plt.xlabel('y/m')
        plt.ylabel('z/m')
    elif axis==1:
        plt.xlabel('x/m')
        plt.ylabel('z/m')
    else:
        plt.xlabel('x/m')
        plt.ylabel('y/m')
    
    plt.savefig(filename + '.pdf')
    plt.show()