# can invert colormap by addint '_r' to the end of cmap name


import numpy as np
import matplotlib.pyplot as plt
def draw_basis(ax, exyz = np.eye(3), clrs = ['r','g','b'], lbls = ['$e_x$', '$e_y$','$e_z$'], alpha = 1, ls = 'solid',ax_lbls = ['X','Y', 'Z'], **kwargs):
    """ Draws 3 arrows from origin on 3d plot. each row of exyz is a vector."""
    for e, clr, lbl in zip(exyz, clrs, lbls):
        ax.quiver(*np.zeros(3), *e, color=clr, label=lbl, alpha = alpha, ls = ls, **kwargs)

    # Set axis limits: get old, calc new from exyz, get overall bounds by combining limits
    min_max_ax = np.array([ax.get_xlim(),ax.get_ylim(),ax.get_ylim()])
    min_max_exyz = np.array([   [np.min(exyz[:,0]), np.max(exyz[:,0])],
                                [np.min(exyz[:,1]), np.max(exyz[:,1])],
                                [np.min(exyz[:,2]), np.max(exyz[:,2])]])
    
    min_max_join = np.hstack((min_max_ax, min_max_exyz))  # join old and new min_max, xses with xses..

    min_max_new = np.array([np.min(min_max_join, axis = 1), 
                            np.max(min_max_join, axis = 1)]).T

    ax.set_xlim(min_max_new[0])
    ax.set_ylim(min_max_new[1])
    ax.set_zlim(min_max_new[2])

    ax.set_xlabel(ax_lbls[0])
    ax.set_ylabel(ax_lbls[1])
    ax.set_zlabel(ax_lbls[2])




