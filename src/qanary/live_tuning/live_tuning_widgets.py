from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import ipywidgets as widgets
from IPython.display import display
from qcodes.parameters import Parameter

if TYPE_CHECKING:
    from qanary.live_tuning import LiveTuning


class LiveTuningWidgets:
    """
    Create and manage ipywidgets for LiveTuning controls.

    The widgets:
        - are created automatically from the registered controls,
        - send new values through LiveTuning.set(),
        - are enabled while a measurement is running,
        - are disabled when no measurement is running.
    """

    def __init__(
        self,
        tuning: LiveTuning,
        controls: Sequence[Parameter],
        ranges: Sequence[tuple[float, float]],
        *,
        step: float = 0.01,
        continuous_update: bool = True,
        auto_display: bool = True,
    ) -> None:

        if len(controls) != len(ranges):
            raise ValueError(
                "controls and control_ranges must have the same length. "
                f"Received {len(controls)} controls and "
                f"{len(ranges)} ranges."
            )

        if step <= 0:
            raise ValueError("Widget step must be > 0.")

        self.tuning = tuning
        self.controls = list(controls)
        self.ranges = list(ranges)

        self.step = float(step)
        self.continuous_update = continuous_update

        self.sliders: dict[str, widgets.FloatSlider] = {}

        self._container = self._create_widgets()

        # React to start()/stop().
        self.tuning.add_state_callback(self._update_running_state)

        # Initially no measurement is running.
        self._update_running_state(self.tuning.running)

        if auto_display:
            self.display()

    @property
    def widget(self) -> widgets.Widget:
        """
        Root widget containing all live-tuning controls.
        """
        return self._container

    def _create_widgets(self) -> widgets.Widget:

        slider_widgets = []

        for parameter, value_range in zip(
            self.controls,
            self.ranges,
            strict=True,
        ):
            minimum, maximum = value_range

            minimum = float(minimum)
            maximum = float(maximum)

            if minimum >= maximum:
                raise ValueError(
                    f"Invalid range for control "
                    f"{parameter.name!r}: "
                    f"{value_range}. "
                    "Minimum must be smaller than maximum."
                )

            # Read current hardware/QCoDeS value.
            try:
                initial_value = float(parameter())

            except Exception:
                # If reading the parameter fails, use the
                # center of the specified range.
                initial_value = (minimum + maximum) / 2

            # Make sure widget starts inside its allowed range.
            initial_value = max(
                minimum,
                min(maximum, initial_value),
            )

            slider = widgets.FloatSlider(
                value=initial_value,
                min=minimum,
                max=maximum,
                step=self.step,
                description=parameter.label or parameter.name,
                continuous_update=self.continuous_update,
                readout=True,
                readout_format=".3f",
                disabled=True,
                style={
                    "description_width": "initial",
                },
                layout=widgets.Layout(
                    width="600px",
                ),
            )

            # Do not use parameter directly in a lambda without binding it.
            slider.observe(
                self._make_slider_callback(parameter),
                names="value",
            )

            self.sliders[parameter.name] = slider
            slider_widgets.append(slider)

        return widgets.VBox(slider_widgets)

    def _make_slider_callback(
        self,
        parameter: Parameter,
    ):
        """
        Create an observer callback bound to one control parameter.
        """

        def callback(change):
            if change["name"] != "value":
                return

            if not self.tuning.running:
                return

            self.tuning.set(
                parameter,
                change["new"],
            )

        return callback

    def _update_running_state(
        self,
        running: bool,
    ) -> None:
        """
        Enable sliders while acquisition is running and freeze them otherwise.
        """

        for slider in self.sliders.values():
            slider.disabled = not running

    def sync_from_parameters(self) -> None:
        """
        Synchronize slider positions with the actual parameter values.

        Observer notifications are temporarily disabled implicitly because
        this should normally be called while no measurement is running.
        """

        for parameter in self.controls:
            slider = self.sliders[parameter.name]

            try:
                value = float(parameter())
            except Exception:
                continue

            value = max(
                slider.min,
                min(slider.max, value),
            )

            slider.value = value

    def display(self) -> None:
        """
        Display the control widget in a Jupyter/IPython environment.
        """
        display(self._container)
