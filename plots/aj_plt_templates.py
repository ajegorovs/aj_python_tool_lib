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


class VectorPlotter:
    def __init__(self,  axis = False):
        if axis:
            self.ax = axis
        else:
            self.fig, self.ax = plt.subplots( subplot_kw=dict(projection='3d'))
        self.vectors_all = np.zeros(shape = (0,3))
        self.autoscaled = False

    def draw_basis(self, origin = np.zeros(3), exyz = np.eye(3), 
                   clrs = ['r','g','b'], lbls = ['$e_x$', '$e_y$','$e_z$'], alpha = 1, 
                   ls = 'solid', ax_lbls = ['X','Y', 'Z'], **kwargs):
        
        """ Draws 3 arrows from origin on 3d plot. each row of exyz is a vector."""
        x0, y0, z0 = origin
        for e, clr, lbl in zip(exyz, clrs, lbls):
            x,y,z = origin + e
            self.ax.quiver(x0,y0,z0,x,y,z,color=clr, label=lbl, alpha = alpha, ls = ls, **kwargs)
            self.vectors_all = np.vstack((self.vectors_all,np.array([x,y,z])))

        self.ax.set_xlabel(ax_lbls[0])
        self.ax.set_ylabel(ax_lbls[1])
        self.ax.set_zlabel(ax_lbls[2])


    def plot_vector(self, xyz, origin = np.zeros(3), **kwargs):
        #x,y,z = origin + xyz
        self.ax.quiver(*origin, *xyz, **kwargs)
        self.vectors_all = np.vstack((self.vectors_all,np.array(xyz)))
    
    def autoscale(self, scale = 1.0):
        """ get min max x-y-z of all active vectors and set axis limits """
        min_max_xyz = np.zeros((3,2))
        for i in range(3):
            min_max_xyz[i] = [np.min(self.vectors_all[:,i]),np.max(self.vectors_all[:,0])]
        min_max_xyz *= scale
        self.set_custom_scale(min_max_xyz)
        self.autoscaled = True

    def set_custom_scale(self, min_max_xyz):
        functions = [self.ax.set_xlim, self.ax.set_ylim, self.ax.set_zlim]
        for i, (fn,min_max) in enumerate(zip(functions,min_max_xyz)):
            fn(*min_max)

    def show(self):
        self.ax.set_aspect('equal')
        if not self.autoscaled: self.autoscale()
        plt.show()

# Example usage:
# vp = VectorPlotter()
# vp.draw_basis()
# vp.autoscale(1.5)
# vp.show()