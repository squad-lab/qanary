# Ganging channels together

`MultiChannelParameter` drives several QCoDeS parameters as one. Setting it
writes every channel; getting it reads them back as a list. Use it when a gate
is physically several DAC channels, or when a set of channels must always move
together.

Nothing here touches hardware -- the example builds its channels on
`ShellInstrument`, so it runs as-is.

```python
from qcdrivers.squad import ShellInstrument
from qcutils.parameters import MultiChannelParameter

multi_param = MultiChannelParameter(
    name="multi",
    label="Multi channel",
    unit="V",
    parameters=[dummy.ch1, dummy.ch2, dummy2.ch1],
)

multi_param(0.5)     # every channel goes to 0.5 V
multi_param()        # -> [0.5, 0.5, 0.5]
```

The parameter carries its own label and unit and produces a QCoDeS snapshot, so
it can go straight into a {class}`~qcutils.measure.Station` and be swept like
any other parameter.

Full example: [`multi_channel_parameter_example.py`](https://gitlab.com/squad-lab/qcutils/-/blob/main/examples/multi_channel_parameter_example.py)
