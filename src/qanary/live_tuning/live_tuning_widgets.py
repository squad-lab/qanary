from __future__ import annotations

from collections.abc import Sequence
from html import escape
from math import isfinite
from typing import TYPE_CHECKING, Any

import ipywidgets as widgets
from IPython.display import display
from qcodes.parameters import Parameter


if TYPE_CHECKING:
    from qanary.live_tuning.live_tuning import LiveTuning


class LiveTuningWidgets:
    """
    Jupyter widgets for LiveTuning controls.

    The initial slider values are taken from the measurement's
    start snapshot, not by independently reading the parameters.

    The start values are also used by the reset button.
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
            raise ValueError(
                "Widget step must be > 0."
            )

        self.tuning = tuning
        self.controls = list(controls)
        self.ranges = list(ranges)

        self.step = float(step)
        self.continuous_update = continuous_update

        self.sliders: dict[str, widgets.FloatSlider] = {}
        self.value_labels: dict[str, widgets.HTML] = {}

        # Values taken from Snapshot Before.
        self._start_values: dict[str, float] = {}

        # Prevent slider callbacks while we update widgets internally.
        self._programmatic_update = False

        self._validate_configuration()

        self.reset_button = widgets.Button(
            description="Reset controls",
            icon="undo",
            tooltip="Reset all controls to their start snapshot values",
            disabled=True,
        )

        self.reset_button.on_click(
            self._reset_controls
        )

        self._container = self._create_widgets()

        # Widgets follow LiveTuning start/stop state.
        self.tuning.add_state_callback(
            self._update_running_state
        )

        self._update_running_state(
            self.tuning.running
        )

        if auto_display and self.controls:
            self.display()

    @property
    def widget(self) -> widgets.Widget:
        return self._container

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _validate_configuration(self) -> None:

        names = [
            parameter.name
            for parameter in self.controls
        ]

        if len(names) != len(set(names)):
            raise ValueError(
                "Live tuning control names must be unique."
            )

        for parameter, value_range in zip(
            self.controls,
            self.ranges,
            strict=True,
        ):

            if len(value_range) != 2:
                raise ValueError(
                    f"Control range for {parameter.name!r} "
                    "must contain exactly (min, max)."
                )

            minimum = float(value_range[0])
            maximum = float(value_range[1])

            if not isfinite(minimum) or not isfinite(maximum):
                raise ValueError(
                    f"Control range for {parameter.name!r} "
                    "must be finite."
                )

            if minimum >= maximum:
                raise ValueError(
                    f"Invalid range for {parameter.name!r}: "
                    f"{value_range}. Minimum must be smaller "
                    "than maximum."
                )

    # ------------------------------------------------------------------
    # Widget creation
    # ------------------------------------------------------------------

    def _create_widgets(self) -> widgets.Widget:

        rows = []

        for parameter, value_range in zip(
            self.controls,
            self.ranges,
            strict=True,
        ):
            minimum = float(value_range[0])
            maximum = float(value_range[1])

            # We do NOT read parameter() here.
            #
            # The real initial value will be loaded later from
            # Snapshot Before.
            temporary_value = (
                minimum + maximum
            ) / 2

            slider = widgets.FloatSlider(
                value=temporary_value,
                min=minimum,
                max=maximum,
                step=self.step,
                description="",
                continuous_update=self.continuous_update,

                # We provide our own value display above the handle.
                readout=False,

                disabled=True,

                layout=widgets.Layout(
                    width="450px",
                ),
            )

            value_label = widgets.HTML(
                value=self._make_empty_value_label(),
                layout=widgets.Layout(
                    width="450px",
                    height="28px",
                ),
            )

            label = widgets.HTML(
                value=(
                    f"<b>{escape(parameter.label or parameter.name)}</b>"
                ),
                layout=widgets.Layout(
                    width="170px",
                ),
            )

            slider.observe(
                self._make_slider_callback(
                    parameter
                ),
                names="value",
            )

            self.sliders[
                parameter.name
            ] = slider

            self.value_labels[
                parameter.name
            ] = value_label

            slider_stack = widgets.VBox(
                [
                    value_label,
                    slider,
                ],
                layout=widgets.Layout(
                    width="450px",
                ),
            )

            row = widgets.HBox(
                [
                    label,
                    slider_stack,
                ],
                layout=widgets.Layout(
                    align_items="flex-end",
                ),
            )

            rows.append(row)

        return widgets.VBox(
            [
                *rows,
                self.reset_button,
            ]
        )

    # ------------------------------------------------------------------
    # Snapshot handling
    # ------------------------------------------------------------------

    def load_start_snapshot(
        self,
        snapshot: dict[str, Any],
    ) -> dict[str, float]:
        """
        Initialize controls from Snapshot Before.

        Validates that every control:
            - exists in the snapshot,
            - has a numeric finite value,
            - lies inside its configured range.

        Returns:
            Dictionary mapping parameter name -> start value.
        """

        parameter_snapshot = snapshot.get(
            "parameters",
            {},
        )

        start_values: dict[str, float] = {}
        errors: list[str] = []

        for parameter, value_range in zip(
            self.controls,
            self.ranges,
            strict=True,
        ):
            name = parameter.name

            if name not in parameter_snapshot:
                errors.append(
                    f"{name!r}: parameter is not present "
                    "in the start snapshot."
                )
                continue

            raw_value = parameter_snapshot[
                name
            ].get("value")

            if raw_value is None:
                errors.append(
                    f"{name!r}: start snapshot value is None."
                )
                continue

            try:
                value = float(raw_value)

            except (TypeError, ValueError):
                errors.append(
                    f"{name!r}: start snapshot value "
                    f"{raw_value!r} is not numeric."
                )
                continue

            if not isfinite(value):
                errors.append(
                    f"{name!r}: start snapshot value "
                    f"{value!r} is not finite."
                )
                continue

            minimum = float(value_range[0])
            maximum = float(value_range[1])

            if not minimum <= value <= maximum:
                errors.append(
                    f"{name!r}: current value {value} "
                    f"{parameter.unit} is outside the allowed "
                    f"range [{minimum}, {maximum}] "
                    f"{parameter.unit}."
                )
                continue

            start_values[name] = value

        if errors:
            message = (
                "Invalid live tuning control values in "
                "the start snapshot:\n\n"
                + "\n".join(
                    f"  - {error}"
                    for error in errors
                )
            )

            raise ValueError(message)

        self._start_values = start_values

        # Updating the sliders must NOT generate tuning.set(...)
        # calls because acquisition has not started yet.
        self._programmatic_update = True

        try:
            for parameter in self.controls:
                value = start_values[
                    parameter.name
                ]

                slider = self.sliders[
                    parameter.name
                ]

                slider.value = value

                self._update_value_label(
                    parameter,
                    value,
                )

        finally:
            self._programmatic_update = False

        self._update_running_state(
            self.tuning.running
        )

        return dict(start_values)

    # ------------------------------------------------------------------
    # Slider callbacks
    # ------------------------------------------------------------------

    def _make_slider_callback(
        self,
        parameter: Parameter,
    ):
        def callback(change):

            if change["name"] != "value":
                return

            value = float(
                change["new"]
            )

            # Always update visual value.
            self._update_value_label(
                parameter,
                value,
            )

            # Internal widget updates must not touch hardware.
            if self._programmatic_update:
                return

            if not self.tuning.running:
                return

            self.tuning.set(
                parameter,
                value,
            )

        return callback

    # ------------------------------------------------------------------
    # Value bubble
    # ------------------------------------------------------------------

    def _update_value_label(
        self,
        parameter: Parameter,
        value: float,
    ) -> None:

        slider = self.sliders[
            parameter.name
        ]

        minimum = float(slider.min)
        maximum = float(slider.max)

        fraction = (
            (value - minimum)
            / (maximum - minimum)
        )

        fraction = max(
            0.0,
            min(1.0, fraction),
        )

        percentage = (
            100.0 * fraction
        )

        # FloatSlider handles have a finite width.
        #
        # This approximates the handle centre rather than using simply
        # left: XX%, which would be slightly wrong close to the ends.
        handle_width = 16.0

        pixel_correction = (
            handle_width * fraction
        )

        unit = escape(
            str(parameter.unit or "")
        )

        self.value_labels[
            parameter.name
        ].value = f"""
        <div style="
            position: relative;
            width: 100%;
            height: 26px;
        ">
            <span style="
                position: absolute;
                left:
                    calc(
                        8px
                        + {percentage:.6f}%
                        - {pixel_correction:.3f}px
                    );
                transform: translateX(-50%);
                padding: 2px 7px;
                border-radius: 6px;
                background: #555;
                color: white;
                font-size: 12px;
                white-space: nowrap;
            ">
                {value:.4g} {unit}
            </span>
        </div>
        """

    @staticmethod
    def _make_empty_value_label() -> str:
        return """
        <div style="
            height: 26px;
            text-align: center;
            color: #888;
            font-size: 12px;
        ">
            waiting for start snapshot
        </div>
        """

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_controls(
        self,
        _button,
    ) -> None:

        if not self.tuning.running:
            return

        if not self._start_values:
            return

        # First update visual widgets without creating observer
        # hardware requests.
        self._programmatic_update = True

        try:
            for parameter in self.controls:

                value = self._start_values[
                    parameter.name
                ]

                slider = self.sliders[
                    parameter.name
                ]

                slider.value = value

                self._update_value_label(
                    parameter,
                    value,
                )

        finally:
            self._programmatic_update = False

        # Then explicitly request all start values.
        #
        # ControlMailbox coalescing means these are applied safely by
        # the acquisition thread between frames.
        for parameter in self.controls:
            self.tuning.set(
                parameter,
                self._start_values[
                    parameter.name
                ],
            )

    # ------------------------------------------------------------------
    # Start / stop state
    # ------------------------------------------------------------------

    def _update_running_state(
        self,
        running: bool,
    ) -> None:

        ready = bool(
            self._start_values
        ) or not self.controls

        for slider in self.sliders.values():
            slider.disabled = not (
                running and ready
            )

        self.reset_button.disabled = not (
            running
            and bool(self._start_values)
        )

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def display(self) -> None:
        display(
            self._container
        )