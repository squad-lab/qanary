# Control several channels as one

`MultiChannelParameter` presents several QCoDeS parameters as one control. A
set operation writes the same value to every channel, which is useful when a
gate spans several outputs or a group of channels must always move together.

The complete example uses `ShellInstrument`, so you can run it without
hardware.

## Create the combined parameter

```python
from qcdrivers.squad import ShellInstrument
from qanary.parameters import MultiChannelParameter

multi_param = MultiChannelParameter(
    param=[dummy.ch1, dummy.ch2],
    name="multi",
    label="Multi channel",
)

multi_param(0.5)  # both channels are set to 0.5 V
```

All channels must belong to the same root instrument. The combined parameter
inherits their unit and can be added to a {class}`~qanary.measure.Station` or
used in a sweep like any other QCoDeS parameter.

Reading `multi_param()` returns the shared value when all channels agree. It
returns `None` when they differ, making an inconsistent channel group easy to
detect. Its snapshot records the underlying channel names, labels, and units.

Full example: [`multi_channel_parameter_example.py`](https://gitlab.com/squad-lab/qanary/-/blob/main/examples/multi_channel_parameter_example.py)
