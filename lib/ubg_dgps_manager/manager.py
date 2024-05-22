#!/usr/bin/env python
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import io
import zipfile

import pandas as pd
import ipywidgets as widgets

from ipywidgets import GridBox, Layout

import cartopy.crs as ccrs
from cartopy.io.img_tiles import OSM
import geojson
from pyproj.database import query_utm_crs_info
from pyproj.aoi import AreaOfInterest

from pyproj import CRS

from pyproj import Transformer

from IPython.display import display

from .electrode_manager import electrode_manager
from .crtomo_mesh_manager import crtomo_mesh_mgr

mpl.rcParams['text.usetex'] = False


class importer_leica_csv(object):
    """Provides means to import a .csv file exported by our Leica dGPS:

    # Example data

        FID;VID;X;Y;Z;Bild;Genauigkeit;
        1;1;7.70933165124028;53.780542967611;1.352;;0.069;
        2;1;7.70929314596666;53.7805037669251;1.487;;0.066;
        3;1;7.70925422148883;53.7804656675215;1.592;;0.072;
    """
    def __init__(self, callback_upload=None):
        self.label = '.Leica CSV'

        # everything will be rendered in here
        self.output = widgets.Output()

        self.widgets = {}

        self._build_gui()

        # this callback will be called when we process the upload
        # two parameters will be provided:
        # 1) the newly imported data
        # 2) a bool value that indicates if all previous data should be deleted
        self.callback_upload = callback_upload

    def _build_gui(self):
        self.widgets = {
            'upload': widgets.FileUpload(
                # accept=['.txt', '.csv'],
                accept='',
                multiple=False
            ),
            'but_upload': widgets.Button(
                description='Load Leica .csv file',
            ),
            'check_delete_prev': widgets.Checkbox(
                value=False,
                description='Delete Previous Data',
                disabled=False,
                indent=False
            ),
        }
        self.vbox = widgets.VBox([
            self.widgets['upload'],
            self.widgets['check_delete_prev'],
            self.widgets['but_upload'],
        ])

        self.widgets['but_upload'].on_click(
            self.process_upload
        )

        with self.output:
            display(self.vbox)

    def process_upload(self, button):
        upload = self.widgets['upload'].value
        if len(upload) != 1:
            print('We need a file to import in order to work. Doing nothing..')
            # do nothing
            return

        if upload[0]['size'] == 0:
            print('Error: Empty file')
            return

        buffer = io.BytesIO(upload[0].content)
        self.buffer = buffer

        self.data = pd.read_csv(
            buffer,
            encoding='utf-16',
            sep=';',
            float_precision='high'
        )

        # header = buffer.readline().decode('iso-8859-1')
        # print(header)
        # if not header.startswith('FID;VID;X;Y;Z;'):
        #     print('ERROR Loading Leica .csv file')
        #     return

        # self.data = np.loadtxt(buffer, sep=';')

        self.wgs84_points = self.data[['X', 'Y', 'Z']].values

        print(self.wgs84_points)
        delete_prev_data = self.widgets['check_delete_prev'].value

        if self.callback_upload is not None:
            print('Calling callback:', self.callback_upload)
            self.callback_upload(self.wgs84_points, delete_prev_data)


class importer_geojson(object):
    """Provides means to import a geojson file
    """
    def __init__(self, callback_upload=None):
        self.label = '.GeoJSON'

        # everything will be rendered in here
        self.output = widgets.Output()

        self.widgets = {}

        self._build_gui()

        # this callback will be called when we process the upload
        # two parameters will be provided:
        # 1) the newly imported data
        # 2) a bool value that indicates if all previous data should be deleted
        self.callback_upload = callback_upload

    def _build_gui(self):
        self.widgets = {
            'upload': widgets.FileUpload(
                accept='.geojson',
                multiple=False
            ),
            'but_upload': widgets.Button(
                description='Load GeoJSON file',
            ),
            'check_delete_prev': widgets.Checkbox(
                value=False,
                description='Delete Previous Data',
                disabled=False,
                indent=False
            ),
        }
        self.vbox = widgets.VBox([
            self.widgets['upload'],
            self.widgets['check_delete_prev'],
            self.widgets['but_upload'],
        ])

        self.widgets['but_upload'].on_click(
            self.process_upload
        )

        with self.output:
            display(self.vbox)

    def process_upload(self, button):
        upload = self.widgets['upload'].value
        if len(upload) != 1:
            print('We need a file to import in order to work. Doing nothing..')
            # do nothing
            return

        if upload[0]['size'] == 0:
            print('Error: Empty file')
            return

        buffer = io.BytesIO(upload[0].content)
        try:
            self.data_geojson = geojson.load(buffer)
        except Exception as e:
            print('There was an error trying to parse the input file')
            print(e)
            return

        self.wgs84_points = np.array(
            list(
                geojson.utils.coords(
                    self.data_geojson['features']
                )
            )
        )
        print(self.wgs84_points)
        delete_prev_data = self.widgets['check_delete_prev'].value

        if self.callback_upload is not None:
            print('Calling callback:', self.callback_upload)
            self.callback_upload(self.wgs84_points, delete_prev_data)


class importer_geojson_zip(object):
    """Provides means to import a geojson file from a .zip file
    """
    def __init__(self, callback_upload=None):
        self.label = '.GeoJSON ZIP'

        # everything will be rendered in here
        self.output = widgets.Output()

        self.widgets = {}

        self._build_gui()

        # this callback will be called when we process the upload
        # two parameters will be provided:
        # 1) the newly imported data
        # 2) a bool value that indicates if all previous data should be deleted
        self.callback_upload = callback_upload

    def _build_gui(self):
        self.widgets = {
            'upload': widgets.FileUpload(
                accept='.zip',
                multiple=False
            ),
            'gj_selector': widgets.RadioButtons(
                description='.geojson file to load:',
                disabled=False,
            ),
            'but_load_gjfile': widgets.Button(
                description='Load .geojson file',
                disabled=True,
            ),
            'check_delete_prev': widgets.Checkbox(
                value=False,
                description='Delete Previous Data',
                disabled=False,
                indent=False
            ),
        }
        self.vbox = widgets.VBox([
            self.widgets['upload'],
            self.widgets['gj_selector'],
            self.widgets['check_delete_prev'],
            self.widgets['but_load_gjfile'],
        ])

        # self.widgets['but_upload'].on_click(
        #     self.process_upload
        # )

        self.widgets['upload'].observe(self._zip_uploaded)
        self.widgets['gj_selector'].observe(self._selection_changed)
        self.widgets['but_load_gjfile'].on_click(self.process_upload)

        with self.output:
            display(self.vbox)

    def _analyze_zfile(self):
        if self.zfile is None:
            return
        self.gjfiles = []
        for info in self.zfile.filelist:
            if info.file_size > 0 and info.filename.endswith('.geojson'):
                self.gjfiles += [info]

    def _update_gui(self):
        """Update the file-uploader part of the gui

        """
        if self.zfile is None:
            # nothing to do
            return
        self._analyze_zfile()
        options = [(x.filename, index) for index, x in enumerate(self.gjfiles)]
        self.widgets['gj_selector'].options = options
        self.widgets['gj_selector'].value = None

    def _zip_uploaded(self, change):
        if change['name'] == 'value':
            print(change['new'])
            cdict = change['new'][0]
            if cdict['type'] == 'application/zip' and cdict['size'] > 0:
                # try to load this as a zip
                self.zfile = zipfile.ZipFile(
                    io.BytesIO(cdict['content'])
                )
                # print('done loading .zip file')
                self._update_gui()

    def _selection_changed(self, change):
        if change['name'] == 'value':
            new_value = change['new']
            if new_value >= 0:
                self.widgets['but_load_gjfile'].disabled = False

    def process_upload(self, button):
        # extract file into a buffer
        finfo = self.gjfiles[self.widgets['gj_selector'].value]

        # load primary data
        buffer = io.BytesIO()
        buffer.write(self.zfile.read(finfo))
        buffer.seek(0)

        try:
            self.data_gj = geojson.load(buffer)
        except Exception as e:
            print('There was an error trying to parse the input file')
            print(e)
            return

        self.wgs84_points = np.array(
            list(
                geojson.utils.coords(
                    self.data_gj['features']
                )
            )
        )
        delete_prev_data = self.widgets['check_delete_prev'].value

        if self.callback_upload is not None:
            print('Calling callback:', self.callback_upload)
            self.callback_upload(self.wgs84_points, delete_prev_data)


class gui(object):
    def __init__(self, filename=None):
        self.widgets = {}
        self.filename = filename
        if filename is not None:
            print('loading .zip file')
            self.zfile = zipfile.ZipFile(filename)
        else:
            self.zfile = None

        # this holds the filename to the geojson file
        self.file_gj = None
        # data from the geojson file
        self.data_gj = None

        # we import wgs84 coordinates
        self.wgs84_points = None

        # utm coordinates are the coordinates we work with
        # note that all modifications are applied to the utm coords only
        # resetting is thus done by reconverting the wgs84 points to utm, if
        # required
        self.utm_coords = None
        self.utm_active_indices = None

        self.el_manager = None

        self.gui = None
        self._build_gui()

    def _build_importer_tab(self):
        self.importers = {
            'geojson_zip': importer_geojson_zip(self._add_to_gps_coords),
            'geojson': importer_geojson(self._add_to_gps_coords),
            'leica': importer_leica_csv(self._add_to_gps_coords),
        }

        self.import_tab = widgets.Tab()
        children = []
        titles = []
        for key, item in self.importers.items():
            children += [item.output]
            titles += [item.label]
        self.import_tab.children = children
        self.import_tab.titles = titles

    def _build_gui(self):
        self._build_importer_tab()
        self.widgets = {
            'output1': widgets.Output(),
            'output2': widgets.Output(),
            'output_utm_points': widgets.Output(),
            'but_reverse_electrodes': widgets.Button(
                description='Reverse el. order',
                disabled=True,
            ),
            'but_sort_dist_to_first': widgets.Button(
                description='Sort wrt dist. to first',
                disabled=True,
            ),
            'but_show_el_manager': widgets.Button(
                description='Show Electrode Manager',
                disabled=True,
            ),
            'help_el_manager': widgets.HTML(
                value="<h2>Electrode manager</h2>" +
                "This section is used to:<br />" +
                "a) arange electrode order<br />" +
                "b) adjust electrode heights (e.g., by interpolation)<br />",
                # placeholder='Some HTML',
                # description='Some HTML',
            ),
            'but_show_gps_coordinates': widgets.Button(
                description='Show GPS Coordinates',
                disabled=True,
            ),
            'output_gps_coords': widgets.Output(),
            'output_el_manager': widgets.Output(),
            'but_show_grid_creator': widgets.Button(
                description='Show Grid Creator',
                disabled=True,
            ),
            'help_grid_creator': widgets.HTML(
                value="<h2>Grid Creator</h2> " +
                "Create a CRTomo mesh",
                # placeholder='Some HTML',
                # description='Some HTML',
            ),
            'output_grid_creator': widgets.Output(),
        }

        self.gui = widgets.VBox([
            self.import_tab,
            GridBox(
                children=[
                    self.widgets['output1'],
                    self.widgets['output2'],
                ],
                layout=Layout(
                    width='100%',
                    grid_template_columns='auto 700px',
                    grid_template_rows='auto auto',
                    grid_gap='5px 10px'
                 )
            ),
            widgets.HTML(
                value="<h2>X-Y Table</h2>" +
                "Delete, arange, or modify x-y positions of electrodes",
            ),
            self.widgets['output_utm_points'],
            widgets.HBox([
                self.widgets['but_reverse_electrodes'],
                self.widgets['but_sort_dist_to_first'],
            ]),
            self.widgets['but_show_gps_coordinates'],
            self.widgets['output_gps_coords'],
            self.widgets['but_show_el_manager'],
            self.widgets['help_el_manager'],
            self.widgets['output_el_manager'],
            self.widgets['but_show_grid_creator'],
            self.widgets['help_grid_creator'],
            self.widgets['output_grid_creator'],
        ])

        self.widgets['but_show_gps_coordinates'].on_click(
            self.print_gps_coordinates
        )

    def print_gps_coordinates(self, button):
        """Print WGS84 Coordinates of the final points
        """
        self.widgets['output_gps_coords'].clear_output()
        with self.widgets['output_gps_coords']:
            print('#WGS84 Coordinates (EPSG:4326)')
            print('#lat lon height[m]')
            # print('# UTM Zone: {}'.format(self.utm_zone))
            for active, row in zip(
                    self.utm_active_indices, self.utm_coords):
                if active:
                    wgs = self.transformer_utm_to_wgs.transform(
                        row[0], row[1]
                    )
                    print(
                        '{:.6f} {:.6f} {:.6f}'.format(
                            *wgs, row[2]
                        )
                    )

    def _prepare_utm_transformer(self):

        # First we need to get the UTM zone for our data points
        lon_min = self.wgs84_points[:, 0].min()
        lon_max = self.wgs84_points[:, 0].max()
        # xdiff = xmax - xmin

        lat_min = self.wgs84_points[:, 1].min()
        lat_max = self.wgs84_points[:, 1].max()

        utm_crs_list = query_utm_crs_info(
            datum_name="WGS 84",
            area_of_interest=AreaOfInterest(
                west_lon_degree=lon_min,
                south_lat_degree=lat_min,
                east_lon_degree=lon_max,
                north_lat_degree=lat_max,
            ),
        )
        print('CODE', utm_crs_list[0].code)
        self.utm_crs = CRS.from_epsg(utm_crs_list[0].code)
        print(self.utm_crs.to_wkt(pretty=True))

        # From SW Maps Homepage: All data recorded and exported by SW Maps is
        # in the WGS84 geographic
        # coordinate system (EPSG:4326).
        self.wgs84_crs = CRS.from_string("EPSG:4326")

        self.transformer = Transformer.from_crs(self.wgs84_crs, self.utm_crs)
        self.transformer_utm_to_wgs = Transformer.from_crs(
            self.utm_crs,
            self.wgs84_crs,
        )

    def _recompute_utm_distances(self):

        coords = self.utm_coords.copy()
        self.xy_distances_to_first = np.hstack((
            np.sqrt(
                (coords[:, 0] - coords[0, 0]) ** 2 +
                (coords[:, 1] - coords[0, 1]) ** 2
            )
        ))

        for i in range(3):
            coords[:, i] -= coords[:, i].min()
        self.xy_distances_rel = np.hstack((
            0,
            np.sqrt(
                (coords[1:, 0] - coords[0:-1, 0]) ** 2 +
                (coords[1:, 1] - coords[0:-1, 1]) ** 2
            )
        ))
        print('Distances:')
        print(self.xy_distances_rel)

    def _to_utm(self, force_utm_conversion=False):
        if not force_utm_conversion and self.utm_coords is not None:
            # work was already done
            return
        self._prepare_utm_transformer()
        utm_coords = []
        for point in self.wgs84_points:
            utm_tmp = self.transformer.transform(point[1], point[0])
            utm_coords += [[utm_tmp[0], utm_tmp[1], point[2]], ]

        self.utm_coords = np.array(utm_coords)
        # by default, activate all electrodes
        self.utm_active_indices = np.ones(self.utm_coords.shape[0])

        self._recompute_utm_distances()

    def _clear_advanced_steps(self):
        """Clear the output of later steps.
        Call this function if you reload a file, or changed the order/selection
        of electrodes
        """
        for wname in ('output_el_manager', ):
            if wname in self.widgets:
                self.widgets[wname].clear_output()

        if self.el_manager is not None:
            del (self.el_manager)
            self.el_manager = None

    def clear_gps_coordinates(self):
        # clean up any previous data
        self.wgs84_points = None
        self.utm_coords = None
        self.utm_active_indices = None

    def _add_to_gps_coords(self, new_wgs84_data, delete_prev_data):
        print('Add new WGS84 data')
        if delete_prev_data:
            print('Clearing previous data')
            self.clear_gps_coordinates()

        if self.wgs84_points is None:
            self.wgs84_points = new_wgs84_data
        else:
            self.wgs84_points = np.vstack((
                self.wgs84_points,
                new_wgs84_data,
            ))
            print('New number of data points:', self.wgs84_points.shape)

        # clean up GUI elements dependent on the gps coordinates
        self._clear_advanced_steps()

        self._to_utm(force_utm_conversion=True)
        self.point_widgets = None
        self._build_map_xy_widgets()
        self._update_map_xy_widgets()

    def __load_gjfile(self, button):

        # clean up GUI elements dependent on the gps coordinates
        self._clear_advanced_steps()

        # extract file into a buffer
        finfo = self.gjfiles[self.widgets['gj_selector'].value]

        # load primary data
        buffer = io.BytesIO()
        buffer.write(self.zfile.read(finfo))
        buffer.seek(0)

        self.data_gj = geojson.load(buffer)

        self.wgs84_points = np.array(
            list(
                geojson.utils.coords(
                    self.data_gj['features']
                )
            )
        )
        self._to_utm()

        self.point_widgets = None
        self._build_map_xy_widgets()
        self._update_map_xy_widgets()

    def _build_map_xy_widgets(self):
        """Some widgets (namely, the utm table) need to be adapted to the
        number of data points. Those widgets are stored in self.point_widgets
        If that variable is None, then this function recreates those widgets.



        """
        if self.point_widgets is not None:
            # nothing to do
            return
        self.widgets['output_utm_points'].clear_output()

        print('Building map-xy-widgets')
        self.point_widgets = []
        print('nr of utm coords:', self.utm_coords.shape)
        print(self.utm_coords)
        for index, point in enumerate(self.utm_coords):
            label_nr = widgets.Label('')
            label_x = widgets.Label('')
            label_y = widgets.Label('')
            label_z = widgets.Label('')
            label_dist_prev = widgets.Label('')
            label_dist_first = widgets.Label('')

            checkbox = widgets.Checkbox(
                value=True,
                description='Use as electrode',
                disabled=False,
                indent=False
            )

            button1 = widgets.Button(description='ignore all before')
            button1.on_click(
                lambda button, eindex=index: self.ignore_all_before(
                    eindex, button
                )
            )

            button2 = widgets.Button(description='ignore all after')
            button2.on_click(
                lambda button, eindex=index: self.ignore_all_after(
                    eindex, button
                )
            )

            items = [
                # 0
                label_nr,
                # 1
                label_x,
                # 2
                label_y,
                # 3
                label_z,
                # 4
                label_dist_prev,
                # 5
                label_dist_first,
                # 6
                checkbox,
                # 7
                button1,
                button2
            ]
            self.point_widgets += items

        print('done creating the widgets')

        self.xyz_header = [
            widgets.HTML('<b>El-Nr (1:)</b>'),
            widgets.HTML('<b>x [m]</b>'),
            widgets.HTML('<b>y [m]</b>'),
            widgets.HTML('<b>z [m]</b>'),
            widgets.HTML('<b>distance [m]</b>'),
            widgets.HTML('<b>distance abs [m]</b>'),
            widgets.HTML(' '),
            widgets.HTML(' '),
            widgets.HTML(' '),
        ]
        # check just to make sure we added all headers
        print('pre check 1')
        assert len(self.xyz_header) == len(items), 'Not enough header items'
        print('post check 1')

        self.gridbox_xyz_points = GridBox(
            children=self.xyz_header + self.point_widgets,
            layout=Layout(
                width='100%',
                grid_template_columns=' '.join((
                    # label:nr
                    '70px',
                    # label:x
                    '70px',
                    # label:y
                    '80px',
                    # label:z
                    '70px',
                    # label:distance
                    '90px',
                    # label:distance abs
                    '115px',
                    # checkbox use el
                    '130px',
                    # button: ignore all before
                    '100px',
                    # button: ignore all after
                    '100px',
                )),
                # grid_template_rows='80px auto 80px',
                # grid_gap='5px 10px'
            )
        )
        print('done creating')
        # vbox = widgets.VBox(self.point_widgets)
        with self.widgets['output_utm_points']:
            display(self.gridbox_xyz_points)

    def _update_map_xy_widgets(self):
        self._recompute_utm_distances()

        # update the GUI
        # Note: From here on we only work on the UTM coordinates
        #       self.utm_coords. We may modify this array
        self.widgets['output1'].clear_output()
        with self.widgets['output1']:
            with plt.ioff():
                fig, ax = self.plot_utm_to_map()
            display(fig)

        self.widgets['output2'].clear_output()
        with self.widgets['output2']:
            with plt.ioff():
                fig, ax = self.plot_utm_topography(relative=True)
            display(fig)

        # show table with utm-datapoints

        mean_electrode_distance = np.mean(self.xy_distances_rel)

        el_nr = 1
        # self.point_widgets = []
        for index, point in enumerate(self.utm_coords):
            nr_i = len(self.xyz_header)
            items = self.point_widgets[index * nr_i: (index + 1) * nr_i]
            if self.utm_active_indices[index]:
                el_label = '{}'.format(el_nr)
                el_nr += 1
            else:
                el_label = '-'
            # nr:
            items[0].value = el_label

            items[1].value = '{:.2f}'.format(point[0])
            items[2].value = '{:.2f}'.format(point[1])
            items[3].value = '{:.2f}'.format(point[2])
            items[4].value = '{:.2f}'.format(self.xy_distances_rel[index])
            if index > 0 and self.xy_distances_rel[index] <= 0.15:
                items[4].style.background = 'red'
            elif index > 0 and self.xy_distances_rel[
                    index] <= mean_electrode_distance / 3:
                items[4].style.background = 'orange'
            else:
                items[4].style.background = 'white'
            items[5].value = '{:.2f}'.format(self.xy_distances_to_first[index])

            items[6].unobserve_all()
            if self.utm_active_indices[index] == 0:
                # disable this line
                for item in items:
                    item.disabled = True

                # disable checkbox
                items[6].value = 0
                items[6].disabled = False
            else:
                for item in items:
                    item.disabled = False
                items[6].value = 1
                items[6].disabled = False

            items[6].observe(
                lambda change, eindex=index: self.set_status_use_as_electrode(
                    eindex, change
                ),
                names='value'
            )

        self.widgets['but_reverse_electrodes'].disabled = False
        self.widgets['but_reverse_electrodes'].on_click(
            self.reverse_electrode_order
        )

        self.widgets['but_sort_dist_to_first'].disabled = False
        self.widgets['but_sort_dist_to_first'].on_click(
            self.sort_utm_dist_to_first
        )

        self.widgets['but_show_gps_coordinates'].disabled = False

        # now we can show the electrode manager
        self.widgets['but_show_el_manager'].disabled = False

        self.widgets['but_show_el_manager'].on_click(
            self._show_electrode_manager
        )

        self.widgets['but_show_grid_creator'].on_click(
            self._show_grid_creator
        )

        self.widgets['output_gps_coords'].clear_output()

    def sort_utm_dist_to_first(self, button):
        self._clear_advanced_steps()

        indices = np.argsort(self.xy_distances_to_first)
        self.utm_coords = self.utm_coords[indices, :]
        self.utm_active_indices = self.utm_active_indices[indices]
        self._recompute_utm_distances()
        self._update_map_xy_widgets()

    def reverse_electrode_order(self, button):
        print('Reversing electrode positions:')
        # print('before')
        # print(self.utm_coords)
        self._clear_advanced_steps()
        self.utm_coords = self.utm_coords[::-1, :]
        self.utm_active_indices = self.utm_active_indices[::-1]
        # print(self.utm_coords)
        self._update_map_xy_widgets()

    def ignore_all_before(self, eindex, button):
        print('Ignoring all electrodes all before', eindex)
        for i in range(0, eindex):
            self.utm_active_indices[i] = 0
        self._update_map_xy_widgets()

    def ignore_all_after(self, eindex, button):
        print('Ignoring all electrodes all before', eindex)
        for i in range(eindex + 1, len(self.utm_active_indices)):
            self.utm_active_indices[i] = 0
        self._update_map_xy_widgets()

    def set_status_use_as_electrode(self, index, change):
        print('xy widgets: set as active electrode')
        self._clear_advanced_steps()
        self.utm_active_indices[index] = int(change['new'])
        self._update_map_xy_widgets()

    def _show_grid_creator(self, button):
        self.grid_creator = crtomo_mesh_mgr(
            self.el_manager.get_electrode_positions(),
            output=self.widgets['output_grid_creator'],
        )
        self.grid_creator.show()

    def get_active_electrode_positions(self):
        positions = []
        for active, position in zip(self.utm_active_indices, self.utm_coords):
            if active:
                positions.append(position)
                print(positions)
        return np.vstack(positions)

    def _show_electrode_manager(self, button):
        coords = self.get_active_electrode_positions()
        xy_distances_to_first = np.hstack((
            np.sqrt(
                (coords[:, 0] - coords[0, 0]) ** 2 +
                (coords[:, 1] - coords[0, 1]) ** 2
            )
        ))

        print(xy_distances_to_first.shape)
        print(np.zeros(self.utm_coords.shape[0]).shape)
        print(
            (self.utm_coords[:, 2] - self.utm_coords[:, 2].min()).shape
        )

        electrode_positions = np.vstack((
            # np.cumsum(self.xy_distances_rel),
            xy_distances_to_first,
            # we ignore y basically...
            np.zeros(xy_distances_to_first.size),
            # relative heights
            coords[:, 2] - coords[:, 2].min(),
        )).T
        if self.el_manager is not None:
            del (self.el_manager)
        self.el_manager = electrode_manager(
            electrode_positions,
            output=self.widgets['output_el_manager'],
        )
        self.el_manager.show()
        self.widgets['but_show_grid_creator'].disabled = False

    def show(self):
        display(self.gui)

    def plot_utm_to_map(self):
        imagery = OSM(
            cache=True,
        )
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(
            1, 1, 1,
            projection=imagery.crs,
        )

        # skip the last character, which we implicitly assume to be N
        self.utm_zone = int(self.utm_crs.utm_zone[:-1])
        utm_projection = ccrs.UTM(self.utm_zone)

        # try to find some sane extents based on the loaded data points
        # longitude (east-west, x-axis)
        xmin = self.utm_coords[:, 0].min()
        xmax = self.utm_coords[:, 0].max()
        xdiff = xmax - xmin

        ymin = self.utm_coords[:, 1].min()
        ymax = self.utm_coords[:, 1].max()
        ydiff = ymax - ymin

        extent = [
            self.utm_coords[:, 0].min() - xdiff / 2,
            self.utm_coords[:, 0].max() + xdiff / 2,
            self.utm_coords[:, 1].min() - ydiff / 2,
            self.utm_coords[:, 1].max() + ydiff / 2,
        ]

        ax.set_extent(extent, utm_projection)

        # Add the imagery to the map.
        detail = 19
        ax.add_image(imagery, detail)

        for index, point in enumerate(self.utm_coords):
            ax.scatter(
                point[0],
                point[1],
                transform=utm_projection,
                s=100,
                color='r',
            )
            ax.scatter(
                point[1],
                point[0],
                transform=utm_projection,
                s=100,
                color='g',
            )
            ax.annotate(
                '{}'.format(int(index)),
                xy=(point[0], point[1]),
                transform=utm_projection,
                color='r',
                fontsize=26,
                bbox=dict(boxstyle="round", fc="0.8"),
                textcoords='offset pixels',
                xytext=(10, 25),
            )

        # line from first to end electrode
        # pstart = self.wgs84_points[0]
        # pend = self.wgs84_points[-1]

        # ax.plot(
        #     [pstart[0], pend[0]],
        #     [pstart[1], pend[1]],
        #     '-',
        #     transform=ccrs.PlateCarree(),
        # )

        return fig, ax

    def plot_utm_topography(self, relative=False):
        coords = self.utm_coords.copy()
        if relative:
            for i in range(3):
                coords[:, i] -= coords[:, i].min()
        xy_distances = self.xy_distances_rel

        fig, axes = plt.subplots(1, 2)
        ax = axes[0]
        ax.plot(
            np.cumsum(xy_distances),
            coords[:, 2],
            '.-',
        )
        ax.grid()
        ax.set_xlabel('distance [m]')
        ax.set_ylabel('height [m]')
        if relative:
            ax.set_title('Relative heights', loc='left', fontsize=7)
        ax.set_title('Topography', loc='right')

        ax = axes[1]
        ax.plot(
            range(1, xy_distances.size),
            xy_distances[1:],
            '.-',
        )
        ax.grid()
        ax.set_ylabel('Distances to prev. el. [m]')
        ax.set_xlabel('Electrode Nr')
        return fig, ax


if __name__ == '__main__':
    gjsel = gui('fake-ert GeoJSON.zip')
    gjsel.show()
