"""
Assumes python 3, numpy and mayavi is installed.
"""
import numpy as np

vertices = np.load("./data/ch2o_vertices.npy")
triangles = np.load("./data/ch2o_triangles.npy")
esp = np.load("./data/ch2o_esp.npy")

from mayavi import mlab


# Mayavi settings:   https://docs.enthought.com/mayavi/mayavi/auto/mlab_figure.html
#   Setting background color to white
#   Setting colour of all text to black
#   Setting size to (500, 500) in pixels
white = (1, 1, 1)  # White
black = (0, 0, 0)  # Black
size = (800, 700)

# Plotting isosurface with property (electrostatic)
mlab.figure(1, size=size, bgcolor=white, fgcolor=black)
mesh = mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles, scalars=esp)
mlab.colorbar(orientation="horizontal", title="Electrostatic Potential")
mesh.actor.mapper.scalar_visibility = True
mesh.scene.anti_aliasing_frames = 8  # Setting anti-aiasing highest
mlab.title("Isosurface of CH2O")
mlab.savefig("./data/ch2o_isosurface_with_electrostatic.png")
mlab.show()



# Plotting isosurface with molecular coordinates
# https://docs.enthought.com/mayavi/mayavi/auto/example_chemistry.html#example-chemistry
coordinates = np.array([[ 2.27823914e+00,  4.13899085e-07,  3.12033662e-07],  # Oxygen
                        [ 1.01154892e-02,  1.09802629e-07, -6.99333116e-07],  # Carbon
                        [-1.09577141e+00,  1.77311416e+00,  1.42544321e-07],  # Hydrogen
                        [-1.09577166e+00, -1.77311468e+00,  2.44755133e-07]   # Hydrogen
                        ])
mlab.figure(1, size=size, bgcolor=white, fgcolor=black)
mlab.clf()
mlab.title("Isosurface of CH2O")
# Plot atomics
mlab.points3d(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2],
              scale_factor=1,
              resolution=20,
              color=(1, 0, 0),
              scale_mode='none')
mesh = mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles, scalars=esp,
                            opacity=0.3)
# Plot the bonds between them:
mlab.plot3d(coordinates[:2, 0], coordinates[:2, 1], coordinates[:2, 2], [1, 1],
            tube_radius=0.1, color=black)  # Oxygen to Carbon
mlab.plot3d(coordinates[1:3, 0], coordinates[1:3, 1], coordinates[:2, 2], [1, 1],
            tube_radius=0.1, color=black)  # Carbon to H1
mlab.plot3d(coordinates[[1, 3], 0], coordinates[[1, 3], 1], coordinates[[1, 3], 2], [1, 1],
            tube_radius=0.1, color=black)  # Carbon to H2
mlab.savefig("./data/ch2o_isosurface_with_electrostatic_with_coordinates.png")
mlab.show()

