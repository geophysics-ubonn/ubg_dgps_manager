# Uni-Bonn-Geophysics dGPS Manager

A Jupyter-based gui that allows importing gps data points, cleaning them up,
and finally, generating FE meshes for electrical tomography applications.

![screenshot_1.jpg](screenshot_1.jpg)
![screenshot_1.jpg](screenshot_2.jpg)
![screenshot_1.jpg](screenshot_3.jpg)

## Installation

You can directly install from this git repository:

    pip install git+https://github.com/geophysics-ubonn/ubg_dgps_manager

We also provide a pypi package:

    pip install ubg_dgps_manager

## Usage

Open a Juypter Notebook or Juypter Lab session and execute the following code
in one cell:

    import ubg_dgps_manager
    mgr = ubg_dgps_manager.gui()
    mgr.show()

## Help

A help text is displayed in the GUI after initialization.
