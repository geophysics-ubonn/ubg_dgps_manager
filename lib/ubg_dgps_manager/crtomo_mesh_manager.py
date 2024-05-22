import os
import io
import tempfile
import subprocess
import zipfile
import shutil
import IPython

import crtomo

import numpy as np
import ipywidgets as widgets
from IPython.display import display
import shapely
import shapely.plotting
import matplotlib.pyplot as plt


class crtomo_mesh_mgr(object):
    def __init__(self, electrode_positions, output=None):
        self.electrode_positions = electrode_positions

        # compute a few parameters
        self.distance_x = np.abs(
            self.electrode_positions[:, 0].max(
            ) - self.electrode_positions[:, 0].min()
        )
        self.distance_z = np.abs(
            self.electrode_positions[:, 1].max(
            ) - self.electrode_positions[:, 1].min()
        )
        self.distances = np.sqrt(
            (self.electrode_positions[1:, 0] -
             self.electrode_positions[:-1, 0]) ** 2 +
            (self.electrode_positions[1:, 1] -
             self.electrode_positions[:-1, 1]) ** 2
        )

        self.eldist_min = self.distances.min()
        self.eldist_max = self.distances.max()

        self.mesh = None
        self.gui = []

        if output is None:
            self.output = widgets.Output()
        else:
            self.output = output

        # used to plot the geometry
        self.output_geometry = widgets.Output()
        # used to plot the mesh
        self.output_mesh = widgets.Output()
        self.output_links = widgets.Output()

        self._build_widgets()

    def on_widget_changed(self, change):
        self.plot_geometry()

    def _build_widgets(self):

        offset_x = self.distance_x / 3.0
        offset_z = self.distance_x / 2.0

        loc_precision = 3
        self.widgets = {
            'but_gen_mesh': widgets.Button(
                description='Generate Mesh',
            ),
            'upper_left_x': widgets.FloatText(
                value=np.round(
                    self.electrode_positions[0, 0] - offset_x,
                    loc_precision,
                ),
                step=0.5,
                description='Upper left corner (X):',
            ),
            'upper_left_z': widgets.FloatText(
                value=np.round(
                    self.electrode_positions[0, 1],
                    loc_precision,
                ),
                step=0.5,
                description='Upper left corner (Z):',
            ),
            'lower_left_x': widgets.FloatText(
                value=np.round(
                    self.electrode_positions[0, 0] - offset_x,
                    loc_precision,
                ),
                step=0.5,
                description='Lower left corner (X):',
            ),
            'lower_left_z': widgets.FloatText(
                value=np.round(
                    self.electrode_positions[:, 0].min() - offset_z,
                    loc_precision,
                ),
                step=0.5,
                description='Lower left corner (Z):',
            ),
            'upper_right_x': widgets.FloatText(
                value=np.round(
                    self.electrode_positions[-1, 0] + offset_x,
                    loc_precision,
                ),
                step=0.5,
                description='Upper right corner (X):',
            ),
            'upper_right_z': widgets.FloatText(
                value=np.round(
                    self.electrode_positions[-1, 1],
                    loc_precision,
                ),
                step=0.5,
                description='Upper right corner (Z):',
            ),
            'lower_right_x': widgets.FloatText(
                value=np.round(
                    self.electrode_positions[-1, 0] + offset_x,
                    loc_precision,
                ),
                step=0.5,
                description='Lower right corner (X):',
            ),
            'lower_right_z': widgets.FloatText(
                value=np.round(
                    self.electrode_positions[:, 0].min() - offset_z,
                    loc_precision,
                ),
                step=0.5,
                description='Lower right corner (Z):',
            ),
            'char_length_1': widgets.FloatText(
                value=np.round(self.eldist_min / 3.0, 4),
                step=0.1,
                description='Size at electrodes:',
            ),
            'char_length_2': widgets.FloatText(
                value=np.round(self.eldist_min, 4),
                step=0.1,
                description='Size at boundaries:',
            ),
        }
        self.widgets['but_gen_mesh'].on_click(self.generate_mesh)

        self.widgets['upper_left_x'].observe(
            self.on_widget_changed, names='value')
        self.widgets['upper_left_z'].observe(
            self.on_widget_changed, names='value')
        self.widgets['lower_left_x'].observe(
            self.on_widget_changed, names='value')
        self.widgets['lower_left_z'].observe(
            self.on_widget_changed, names='value')

        self.widgets['upper_right_x'].observe(
            self.on_widget_changed, names='value')
        self.widgets['upper_right_z'].observe(
            self.on_widget_changed, names='value')
        self.widgets['lower_right_x'].observe(
            self.on_widget_changed, names='value')
        self.widgets['lower_right_z'].observe(
            self.on_widget_changed, names='value')

        gui = widgets.VBox([
            widgets.HBox([
                widgets.VBox(
                    [
                        widgets.HBox([
                            self.widgets['upper_left_x'],
                            self.widgets['upper_left_z'],
                        ]),
                        widgets.HBox([
                            self.widgets['lower_left_x'],
                            self.widgets['lower_left_z'],
                        ]),
                        widgets.HBox([
                            self.widgets['upper_right_x'],
                            self.widgets['upper_right_z'],
                        ]),
                        widgets.HBox([
                            self.widgets['lower_right_x'],
                            self.widgets['lower_right_z'],
                        ]),
                        widgets.HBox([
                            self.widgets['char_length_1'],
                            self.widgets['char_length_2'],
                        ]),
                        self.widgets['but_gen_mesh'],
                    ]
                ),
                self.output_geometry,
            ]),
            self.output_mesh,
            self.output_links,
        ])
        self.gui = gui

    def show(self):
        self.output.clear_output()
        with self.output:
            display(self.gui)
        display(self.output)
        self.plot_geometry()

    def plot_geometry(self):
        with plt.ioff():
            fig, ax = plt.subplots()

            points_all = [
                [self.widgets[
                    'upper_left_x'].value, self.widgets['upper_left_z'].value],
            ]
            for electrode in self.electrode_positions:
                points_all += [electrode]

            points_all += [
                [self.widgets['upper_right_x'].value,
                 self.widgets['upper_right_z'].value],
                [self.widgets['lower_right_x'].value,
                 self.widgets['lower_right_z'].value],
                [self.widgets['lower_left_x'].value,
                 self.widgets['lower_left_z'].value],
            ]
            polygon = shapely.geometry.Polygon(points_all)
            self.polygon = polygon

            shapely.plotting.plot_polygon(polygon, ax=ax, linestyle='-')
            if polygon.is_simple:
                ax.set_title('Geometry looks valid!', loc='left')

            ax.scatter(
                self.electrode_positions[:, 0],
                self.electrode_positions[:, 1],
                label='electrodes',
                s=200,
                color='k',
            )
            # outer mesh corners
            ax.scatter(
                self.widgets['upper_left_x'].value,
                self.widgets['upper_left_z'].value,
                label='upper left',
                s=200,
            )
            ax.scatter(
                self.widgets['lower_left_x'].value,
                self.widgets['lower_left_z'].value,
                label='lower left',
                s=200,
            )
            ax.scatter(
                self.widgets['upper_right_x'].value,
                self.widgets['upper_right_z'].value,
                label='upper right',
                s=200,
            )
            ax.scatter(
                self.widgets['lower_right_x'].value,
                self.widgets['lower_right_z'].value,
                label='lower right',
            )

            ax.legend()
            ax.grid()
            self.output_geometry.clear_output()
            with self.output_geometry:
                display(fig)
            del (fig)

    def generate_mesh(self, button):
        print('Generating mesh')
        tempdir = tempfile.mkdtemp(
            prefix='dgps_manager_tmp_meshdir_'
        )
        # tempdir = 'tmp'
        print('PWD', os.getcwd())
        np.savetxt(
            tempdir + os.sep + 'electrodes.dat',
            self.electrode_positions, fmt='%.4f %.4f'
        )

        with open(tempdir + os.sep + 'boundaries.dat', 'w') as fid:
            fid.write('{:.4f} {:.4f} {}\n'.format(
                self.widgets['upper_left_x'].value,
                self.widgets['upper_left_z'].value,
                12
            ))
            for electrode in self.electrode_positions:
                fid.write('{:.4f} {:.4f} {}\n'.format(
                    *electrode, 12
                ))
            fid.write('{:.4f} {:.4f} {}\n'.format(
                self.widgets['upper_right_x'].value,
                self.widgets['upper_right_z'].value,
                11
            ))
            fid.write('{:.4f} {:.4f} {}\n'.format(
                self.widgets['lower_right_x'].value,
                self.widgets['lower_right_z'].value,
                11
            ))
            fid.write('{:.4f} {:.4f} {}\n'.format(
                self.widgets['lower_left_x'].value,
                self.widgets['lower_left_z'].value,
                11
            ))

        with open(tempdir + os.sep + 'char_length.dat', 'w') as fid:
            fid.write('{:.4f}\n'.format(
                self.widgets['char_length_1'].value
            ))
            fid.write('{:.4f}\n'.format(
                self.widgets['char_length_2'].value
            ))
            # repeat the previous values two times for extra lines and nodes,
            # which we do not use here
            fid.write('{:.4f}\n'.format(
                self.widgets['char_length_2'].value
            ))
            fid.write('{:.4f}\n'.format(
                self.widgets['char_length_2'].value
            ))

        pwd = os.getcwd()
        print(pwd)
        os.chdir(tempdir)
        try:
            subprocess.call('cr_trig_create grid', shell=True)
        except Exception:
            print('Caught an exception while calling cr_trig_create')
            os.chdir(pwd)
            return
        os.chdir('grid')
        self.mesh = crtomo.crt_grid()
        os.chdir(pwd)
        self.plot_mesh()

        # now we generate a zip file of the directory
        filenames = []
        for root, dirs, files in os.walk(tempdir):
            # level = root.replace('.', '').count(os.sep)
            filenames += [root + os.sep + file for file in files]
        buffer = io.BytesIO()
        nzip = zipfile.ZipFile(buffer, mode='w')
        for filename in filenames:
            nzip.write(filename)
        nzip.close()
        buffer.seek(0)
        self.mesh_dir_zip = buffer

        self.output_links.clear_output()

        tmp_dir_2 = tempfile.mkdtemp(prefix='dpgs_manager_dl_', dir=pwd)
        with self.output_links:
            display(
                widgets.HTML(
                    ''.join((
                        'Created temporary directory: {} '.format(tmp_dir_2),
                        '<br />',
                        'This directory is used to provide downloadable files',
                        '<br />',
                        'After downloading, you can delete this directory',
                        '<br />',
                        '<h3>For download, use the middle mouse button to ',
                        'open the link in a new tab! </h3>',
                    ))
                )
            )

        # save the zip blob
        zip_blob = tmp_dir_2 + os.sep + '/cr_trig_create_directory.zip'
        with open(zip_blob, 'wb') as fid:
            fid.write(buffer.read())

        file_list = [
            tempdir + os.sep + 'grid' + os.sep + 'elem.dat',
            tempdir + os.sep + 'grid' + os.sep + 'elec.dat',
            zip_blob,
        ]

        for file in file_list:
            filename = file
            if os.path.isfile(filename):
                target_file = tmp_dir_2 + os.sep + os.path.basename(file)
                if not os.path.isfile(target_file):
                    # copy only if the target file is not already present
                    shutil.copy(filename, target_file)
                with self.output_links:
                    display(
                         IPython.display.FileLink(
                             os.path.relpath(
                                target_file,
                                os.getcwd()
                             )
                         )
                    )
            else:
                print('TARGET does not exist:', filename)

    def plot_mesh(self):
        if self.mesh is None:
            return
        with plt.ioff():
            fig, ax = plt.subplots()
            self.mesh.plot_grid_to_ax(ax)
        with self.output_mesh:
            self.output_mesh.clear_output()
            display(fig)
