#%%

from qcodes.instrument import Instrument

import numpy as np
from time import sleep
import time


from drivers.basel.dacs.dacs import BaselDac2

from qcutils.sweep import Sweep, rampdown, sweeper
from qcutils.buffered.instruments import NodeBaselDAC

from qcutils import measure
from qcutils.measure import Station


#%%

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


#%%

Instrument.close_all()

dac = BaselDac2('dac', 'TCPIP0::192.168.0.108::23::SOCKET') #Connection issues: use "Restart Telnet now!" on device
dummy = DummyInstrument("dummy")

#%% setup DAC channels

dac.set_bandwidth_safely([1, 13], high_bw=True)
dac.activate_dac_channels([1,13])

# %%

st = Station("test")
st.instruments = [dac]

#%%

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

#%%

def test_1d_sweep():

    gate1_sweep = Sweep(gate1, 0, 1, num=10, start_delay=0, delay=0.03*1e-3)

    dummy.prepare_buffer(gate1_sweep.num)

    buffered_sweep = {
            "instrument": NodeBaselDAC(inst=dac),
            "sweeps": [gate1_sweep],
            "dependent": [dummy.counter_array],
    }

    run_dict["experiment_name"] = "test buffered Basel DAC 1d"

    gate1(0)
    gate2(0)
    time.sleep(0.5)

    measure.run(
        [buffered_sweep],
        dependents=[],
        **run_dict,
    )

# %%

# harware wise: the two gates must be on separate dac boards (high and low) - connect Sync out A into trig C and Sync out C into trig A (small soldered adapter on backside)
# ensure high bandwidth on the used buffered channels if you want to use a small delay !

def test_2d_sweep():

    gate1_sweep = Sweep(gate1, 0, 1, num=5, start_delay=0, delay=0.03*1e-3)
    gate2_sweep = Sweep(gate2, 0, 1, num=3, start_delay=0, delay=0.03*1e-3)

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

Instrument.close_all()

# %%
