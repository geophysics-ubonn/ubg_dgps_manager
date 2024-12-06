"""The electrode_manager class takes 3D (gps) coordinates and produces 2D
coordinates suitable for ERT/TDIP/EIT modelling/inversions.

Correspondingly, it only deals in x-z coordinates

Coordinates can be modified in various ways, see the code for details.
An incomplete list:
    * remove electrodes
    * rearrange electrodes
    * interpolate heights between two existing electrodes

"""
import logging

import numpy as np
import ipywidgets as widgets
from ipywidgets import GridBox, Layout
import matplotlib.pyplot as plt
from IPython.display import display
from scipy.interpolate import PchipInterpolator

from .log_helper import ListHandler

# import matplotlib.pylab as plt


def get_resampled_positions(data_x_raw, data_z_raw, requested_spacing):
    """
    Parameters
    ----------
    data_x: numpy.ndarray of size N
        X-positions
    data_z: numpy.ndarray of size N
        Z-positions
    requested_spacing: float
        Requested spacing of electrodes

    Returns
    -------
    x_new: numpy.ndarray
        New x coordinates
    z_new: numpy.ndarray
        New z coordinates
    N: int
        Number of new electrodes
    """
    data_x = np.atleast_1d(data_x_raw)
    data_z = np.atleast_1d(data_z_raw)

    # compute distance between the input coordinates
    xy_dist_orig = np.cumsum(
        np.hstack(
            (
                0,
                np.sqrt(np.diff(data_x) ** 2 + np.diff(data_z) ** 2)
            )
        )
    )
    max_dist_orig = xy_dist_orig.max()
    N_approx = np.floor(max_dist_orig / requested_spacing)
    print('N_approx:', N_approx)

    interp = PchipInterpolator(data_x, data_z)
    # evaluate the spline at 4 times the density of the approximate requested
    # spacing
    x_val = np.linspace(
        data_x.min(), data_x.max(), int(np.ceil(N_approx * 4))
    )
    y_interp = interp(x_val)

    xy_dist = np.cumsum(
        np.hstack(
            (
                0,
                np.sqrt(np.diff(x_val) ** 2 + np.diff(y_interp) ** 2)
            )
        )
    )
    print('xy_dist', xy_dist)
    # this is the
    max_dist = int(
        requested_spacing * (int(xy_dist.max() / requested_spacing)) + 1
    )
    print('max_dist', max_dist)

    # compute x values
    s_reg = np.linspace(0, max_dist, int(max_dist / requested_spacing + 1))
    print('s_reg', s_reg)

    x_reg = np.interp(s_reg, xy_dist, x_val)
    print('x_reg', x_reg)

    y_reg = interp(x_reg)

    return x_reg, y_reg, x_reg.size


class electrode_manager(object):
    def __init__(self, electrode_positions, output=None):
        self.log = logging.Logger(
            name='electrode_manager',
            level=logging.INFO,
        )
        self.log_handler = ListHandler()
        self.log.addHandler(self.log_handler)
        self.log.info("Remember: Electrode indices start at 0!")

        self.el_coords_orig = electrode_positions
        self.electrode_positions = np.hstack((
            self.el_coords_orig[:, :],
            # active/disabled column
            np.ones(electrode_positions.shape[0])[:, np.newaxis],
        ))

        self.log.info(
            'Got {} electrodes'.format(self.electrode_positions.shape[0])
        )
        self.vbox = None

        self.widgets = {}

        # we render the whole manager in here
        self.output = output

    def get_resampler_widgets(self):
        """The resampler interpolates all points between two electrodes and
        replaces all electrodes in this range with a certain Number of equally
        spaced electrodes (along the interpolated line).

        This function generates the GUI elements of the resampler.

        Returns
        -------
        resample_widgets: list
            A list of gui widgets
        gbox: ipywidgets.GridBox
            The GridBox containing the widgets
        """
        resample_widgets = []

        resample_widgets += [
            widgets.Label(
                'Replace Electrode range (including start and end):'
            ),
            # dummy to enforce a nicer layout
            widgets.Label(''),
            widgets.BoundedIntText(
                value=0,
                description='Start Electrode:',
                style={'description_width': 'initial'},
                disabled=False,
                min=0,
                max=self.electrode_positions.shape[0] - 1,
            ),
            widgets.BoundedIntText(
                value=0,
                description='End Electrode:',
                style={'description_width': 'initial'},
                disabled=False,
                min=0,
                max=self.electrode_positions.shape[0] - 1,
            ),
            widgets.BoundedFloatText(
                value=0,
                description='Requested electrode spacing:',
                style={'description_width': 'initial'},
                disabled=False,
                min=0,
            ),
            widgets.Button(description='Replace electrodes'),
        ]

        gbox = GridBox(
            children=resample_widgets,
            layout=Layout(
                width='100%',
                grid_template_columns='50% 50%',
                border='solid',
            )
        )
        return resample_widgets, gbox

    def resample_points(self, button):
        start_electrode = self.resample_widgets[2].value
        end_electrode = self.resample_widgets[3].value
        if end_electrode <= start_electrode:
            print('ERROR: end electrode must be larger than start electrode')
            return

        nr_electrodes = self.resample_widgets[4].value
        if nr_electrodes <= 0:
            print('ERROR: requested electrode spacing must be > 0 m!')
            return

        el_ids = range(start_electrode, end_electrode + 1)
        # def get_resampled_positions(data_x_raw, data_z_raw,
        # requested_spacing):
        actives = np.where(self.electrode_positions[:, 3])
        active_els = self.electrode_positions[actives, 0:3].squeeze()

        data_x = active_els[el_ids, 0]
        data_z = active_els[el_ids, 2]
        print('number of electrodes:', nr_electrodes)
        new_x, new_z, N = get_resampled_positions(
            data_x, data_z, nr_electrodes
        )

        print('data_x')
        print(data_x)
        print('data_z')
        print(data_z)

        print('new positions:')
        print(new_x)
        print(new_z)

        print('----------_')

        # splice the new electrode in
        self.electrode_positions = np.vstack((
            self.electrode_positions[0:actives[0][start_electrode], :],
            np.vstack((
                new_x,
                np.ones_like(new_x),
                new_z,
                np.ones_like(new_x),
            )).T,
            self.electrode_positions[actives[0][end_electrode]+1:, :]
        ))

        while len(self.el_widgets) < self.electrode_positions.shape[0]:
            # add new widget rows
            row = self._get_electrode_widgets_row()
            self.el_widgets += [row]
            self.widgets['gridbox'].children += tuple(row)

        # update max-nrs
        self.resample_widgets[2].max = self.electrode_positions.shape[0] - 2
        self.resample_widgets[3].max = self.electrode_positions.shape[0] - 1
        self._update_widgets()

    def set_status_use_as_electrode(self, index, change):
        self.electrode_positions[index, 3] = int(change['new'])
        self._update_widgets()
        self.log.info(
            'Changing active-status of electrode index {} to {}'.format(
                index, change['new']
            )
        )

    def _get_electrode_widgets_row(self):
        widget_row = [
            widgets.Label('_'),
            widgets.Label('x'),
            widgets.Label('z'),
            widgets.Button(description='Move down'),
            widgets.Button(description='Move up'),
            widgets.Checkbox(
                value=True,
                description='Use as electrode',
                disabled=False,
                indent=False
            ),
        ]
        return widget_row

    def _build_widgets(self):
        el_widgets = []

        for index, electrode in enumerate(self.electrode_positions):
            # items = []
            items = self._get_electrode_widgets_row()

            items[3].on_click(
                lambda x, eindex=index: self.move_down(x, eindex))
            items[4].on_click(
                lambda x, eindex=index: self.move_up(x, eindex))

            items[5].observe(
                lambda change, eindex=index: self.set_status_use_as_electrode(
                    eindex, change),
                names='value'
            )

            el_widgets += [items]

        flat_items = []
        for row_of_items in el_widgets:
            flat_items += row_of_items
        self.el_widgets = el_widgets

        self.widgets['button_print'] = widgets.Button(
            description='Print Electrode Coordinates',
            style={'description_width': 'initial'},
            disabled=False,
        )
        self.widgets['output_print'] = widgets.Output()
        self.widgets['button_print'].on_click(
            self.print_electrode_coordinates
        )

        self.widgets['button_show_log'] = widgets.Button(
            description='Show LOG',
            disabled=False,
        )
        self.widgets['output_log'] = widgets.Output()
        self.widgets['button_show_log'].on_click(
            self.print_log
        )

        self.widgets['output_points'] = widgets.Output()

        #
        self.widgets['label_interp'] = widgets.Label(
            'Interpolate heights between'
        )
        self.widgets['int_interp_from'] = widgets.BoundedIntText(
            value=0,
            description='Electrode:',
            disabled=False,
            min=0,
            max=self.electrode_positions.shape[0] - 1,
        )
        self.widgets['label_interp_to'] = widgets.Label(
            'to'
        )
        self.widgets['int_interp_to'] = widgets.BoundedIntText(
            value=0,
            description='Electrode:',
            disabled=False,
            min=0,
            max=self.electrode_positions.shape[0] - 1,
        )

        self.widgets['button_interp'] = widgets.Button(
            description='Interpolate',
        )
        self.widgets['button_interp'].on_click(
            self.interpolate_between_points
        )
        self.widgets['hbox_interp'] = widgets.HBox([
            self.widgets['label_interp'],
            self.widgets['int_interp_from'],
            self.widgets['label_interp_to'],
            self.widgets['int_interp_to'],
            self.widgets['button_interp'],
        ])

        self.resample_widgets, resample_gbox = self.get_resampler_widgets()
        self.resample_widgets[5].on_click(
            self.resample_points
        )

        self.xz_header = [
            widgets.HTML('<b>El-Nr (1:)</b>'),
            widgets.HTML('<b>x [m]</b>'),
            widgets.HTML('<b>z [m]</b>'),
            # widgets.HTML('<b>distance [m]</b>'),
            # widgets.HTML('<b>distance abs [m]</b>'),
            widgets.HTML(' '),
            widgets.HTML(' '),
            widgets.HTML(' '),
        ]

        self.widgets['gridbox'] = GridBox(
                children=self.xz_header + flat_items,
                layout=Layout(
                    width='100%',
                    grid_template_columns=' '.join((
                        # el-nr
                        '80px',
                        # x
                        '60px',
                        # z
                        '60px',
                        '150px',
                        '150px',
                        '180px',
                    )),
                    grid_template_rows='auto',
                    grid_gap='5px 10px',
                 )
        )

        vbox = widgets.VBox([
            self.widgets['gridbox'],
            self.widgets['hbox_interp'],
            resample_gbox,
            self.widgets['output_points'],
            self.widgets['button_print'],
            self.widgets['output_print'],
            self.widgets['button_show_log'],
            self.widgets['output_log'],
        ])
        self.vbox = vbox
        self._update_widgets()

    def interpolate_between_points(self, button):
        print('Interpolating between electrodes')
        el_ids = np.sort([
            self.widgets['int_interp_from'].value,
            self.widgets['int_interp_to'].value,
        ])
        self.log.info(
            'Linear interpolation between electrode indices {} and {}'.format(
                *el_ids
            )
        )
        print('Electrode ids:', el_ids)
        if el_ids[1] - el_ids[0] < 2:
            print('Returning')
            return

        actives = np.where(self.electrode_positions[:, 3])
        print(actives)
        active_els = self.electrode_positions[actives, 0:3].squeeze()
        print('active_els:')
        print(active_els.shape, active_els)

        p = np.polyfit(
            [active_els[el_ids[0], 0], active_els[el_ids[1], 0]],
            [active_els[el_ids[0], 2], active_els[el_ids[1], 2]],
            deg=1,
        )
        print('p', p)

        replace_ids = actives[0][
            el_ids[0] + 1:el_ids[1],
        ]
        print('replace ids:', replace_ids)
        print('replace ids.shape:', replace_ids.shape)
        z_new = np.polyval(p, active_els[replace_ids, 0])
        print('Evaluating at:')
        print(active_els[1:-1, 0])
        print('z_new', z_new)
        self.electrode_positions[replace_ids, 2] = z_new

        self._update_widgets()

    def print_log(self, button):
        with self.widgets['output_log']:
            print(self.log_handler.get_str_formatting())

    def print_electrode_coordinates(self, button):
        self.widgets['output_print'].clear_output()
        with self.widgets['output_print']:
            print('#x[m] y[m] z[m]')
            for position in self.electrode_positions:
                if position[3] == 1:
                    print(
                        '{:.6f} {:.6f} {:.6f}'.format(*position[0:3])
                    )

    def _plot_points(self):
        self.widgets['output_points'].clear_output()

        with plt.ioff():
            fig, ax = plt.subplots()
            for position in self.electrode_positions:
                if position[3] == 1:
                    ax.scatter(
                        position[0],
                        position[2],
                        s=50,
                        color='k',
                    )
            ax.set_xlabel('x [m]')
            ax.set_ylabel('z [m]')
            ax.set_title(
                'Mesh topography (z-axis relative to lowest electrode)'
            )
            ax.grid()

        with self.widgets['output_points']:
            display(fig)

    def _update_widgets(self):
        active_electrode_index = 0

        for index, electrode in enumerate(self.electrode_positions):
            line = self.el_widgets[index]
            # inactive electrode
            if electrode[3] == 0:
                line[0].value = 'Electrode -'
                line[1].value = '{:.3f}'.format(electrode[0])
                line[2].value = '{:.3f}'.format(electrode[2])
                # move down button
                line[3].disabled = True
                # move up button
                line[4].disabled = True
                # use-as-electrode checkbox
                line[5].value = False
            else:
                # activate electrode
                line[0].value = 'Electrode {}'.format(active_electrode_index)
                line[1].value = '{:.3f}'.format(electrode[0])
                line[2].value = '{:.3f}'.format(electrode[2])
                # move-down button
                line[3].disabled = False
                # move up button
                line[4].disabled = False
                # use-as-electrode checkbox
                line[5].value = True
                active_electrode_index += 1

        nr_active_electrodes = np.where(self.electrode_positions[:, 3])[0].size
        self.widgets['int_interp_from'].max = nr_active_electrodes - 1
        self.widgets['int_interp_to'].max = nr_active_electrodes - 1

        self._plot_points()

    def set_active_state(self, index, state):
        print('set activate state')
        pass

    def move_down(self, button, index):
        print('Moving down {} -> {}'.format(index, index + 1))
        self.log.info(
            'Moving electrode down {} -> {}'.format(index, index + 1)
        )
        new_position = index + 1
        if new_position >= self.electrode_positions.shape[0]:
            print('doing nothing')
            return
        self.electrode_positions = np.vstack((
            self.electrode_positions[0:index, :],
            self.electrode_positions[index + 1, :],
            self.electrode_positions[index, :],
            self.electrode_positions[index + 2:, :],
        ))
        self._update_widgets()
        self.show()

    def move_up(self, button, index):
        print('Moving up {} -> {}'.format(index, index - 1))
        self.log.info(
            'Moving electrode up {} -> {}'.format(index, index + 1)
        )
        new_position = index - 1
        if new_position < 0:
            print('doing nothing')
            return
        self.electrode_positions = np.vstack((
            self.electrode_positions[0:index - 1, :],
            self.electrode_positions[index, :],
            self.electrode_positions[index - 1, :],
            self.electrode_positions[index + 1:, :],
        ))
        self._update_widgets()
        self.show()

    def show(self):
        if self.vbox is None:
            self._build_widgets()

        if self.output is not None:
            self.output.clear_output()
            with self.output:
                display(self.vbox)
        else:
            display(self.vbox)

    def get_electrode_positions(self):
        indices = np.where(self.electrode_positions[:, 3])
        return np.vstack((
            self.electrode_positions[indices, 0],
            self.electrode_positions[indices, 2],
        )).T
