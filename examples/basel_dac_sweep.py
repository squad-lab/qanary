# %%

import time
from time import sleep

import numpy as np
from qcdrivers.basel.dacs.dacs import BaselDac2
from qcdrivers.buffered.basel import NodeBaselDAC
from qcodes.instrument import Instrument

from qcutils import measure
from qcutils.measure import Station
from qcutils.sweep import Sweep

# %%

# TODO: Keep examples only as/for examples/tutorials, not as tests.
# No tests should be written in examples.


class DummyInstrument(Instrument):
    def __init__(self, name: str):
        super().__init__(name)
        self._counter = 0
        self.buffer_size = 0

        self.add_parameter(
            "counter_number",
            get_cmd=self._get_counter_number,
            set_cmd=lambda x: self._set_counter_number(x),
            initial_value=None,
        )

        self.add_parameter(
            "counter_array",
            get_cmd=self._get_counter_array,
            set_cmd=None,
            initial_value=None,
        )

    def _get_counter_array(self):
        value = self._counter
        self._counter += 1
        return np.arange(0, value, 1)

    def _set_counter_number(self, value):
        self._counter = value

    def _get_counter_number(self):
        return self._counter

    def prepare_buffer(self, n_points: int):
        self._counter = n_points
        self.buffer_size = n_points


# %%

Instrument.close_all()

dac = BaselDac2(
    "dac", "TCPIP0::192.168.0.108::23::SOCKET"
)  # Connection issues: use "Restart Telnet now!" on device
dummy = DummyInstrument("dummy")

# %% setup DAC channels

dac.set_bandwidth_safely([1, 13], high_bw=True)
dac.activate_dac_channels([1, 13])

# %%

st = Station("test")
st.instruments = [dac]

# %%

extra_metadata = {"info": ""}

run_dict = {
    "wafer_id": "test_wafer",
    "device_type": "test_device",
    "sample_name": "test_sample",
    "station": st,
    "fridge_name": "Radler",
    "metadata": extra_metadata,
    "location_return": True,
    "data_location": "C:/Users/s.schreibing/Documents/qimchi_data",  # <-- choose correct folder
}

# %% test buffered basel dac sweep

gate1 = st.add_parameter("gate1", "Dac channel 1", dac.ch1.voltage)
gate2 = st.add_parameter("gate2", "Dac channel 13", dac.ch13.voltage)

# %%


def test_1d_sweep(V_start=0, V_stop=1, num_points=100, delay=0.03 * 1e-3):
    gate1_sweep = Sweep(
        gate1, V_start, V_stop, num=num_points, start_delay=0, delay=delay
    )

    dummy.prepare_buffer(gate1_sweep.num)

    buffered_sweep = {
        "instrument": NodeBaselDAC(inst=dac),
        "sweeps": [gate1_sweep],
        "dependent": [dummy.counter_array],
    }

    run_dict["experiment_name"] = "test buffered Basel DAC 1d"

    gate1(0)
    time.sleep(1)

    measure.run(
        [buffered_sweep],
        dependents=[],
        no_hashing=True,
        **run_dict,
    )


# %%

# hardware wise: the two gates must be on separate dac boards (high and low) - connect Sync out A into trig C and Sync out C into trig A (small soldered adapter on backside)
# ensure high bandwidth on the used buffered channels if you want to use a small delay !


def test_2d_sweep():
    gate1_sweep = Sweep(gate1, 0, 1, num=5, start_delay=0, delay=0.03 * 1e-3)
    gate2_sweep = Sweep(gate2, 0, 1, num=3, start_delay=0, delay=0.03 * 1e-3)

    dummy.prepare_buffer(gate1_sweep.num * gate2_sweep.num)

    buffered_sweep = {
        "instrument": NodeBaselDAC(inst=dac),
        "sweeps": [gate1_sweep, gate2_sweep],
        "dependent": [dummy.counter_array],
    }

    run_dict["experiment_name"] = "test buffered Basel DAC 2d"

    gate1(0)
    gate2(0)
    time.sleep(0.5)

    measure.run(
        [buffered_sweep],
        dependents=[],
        **run_dict,
    )


# %%

# pulse sequence to check if awg is synced up with DAQ module

delay = 30 * 1e-6  # use as standard trigger_delay

number_pulses = (
    10  # set columns = 2*number_pulses + 1, duration = delay * (columns - 1)
)
pulse_heigt = 0.1

waveform = []
for i in range(number_pulses):
    waveform.append(0)
    waveform.append(pulse_heigt)
waveform.append(0)


awg = dac.awga

awg.write_awg_config(
    {
        "channel": 1,
        "cycles": 1,
        "sampling_rate": delay,
        "waveform": np.array(waveform),
    }
)

awg.trigger("disable")
gate1(0)
sleep(0.5)

dac.run_awg_sweep([awg])

# %%

# %%

Instrument.close_all()

# %%
