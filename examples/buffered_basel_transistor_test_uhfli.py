#%%

from qcodes.instrument import Instrument
from qcodes import Parameter

import numpy as np
from time import sleep
import time


from drivers.basel.dacs.dacs import BaselDac2
from drivers.squad.helpers.helpers import Lockin

from qcutils.sweep import Sweep, rampdown, sweeper
from qcutils.buffered.instruments import NodeBaselDAC, NodeUHFLI

from qcutils import measure
from qcutils.measure import Station


#%%

Instrument.close_all()

dac = BaselDac2('dac', 'TCPIP0::192.168.0.108::23::SOCKET') #Connection issues: use "Restart Telnet now!" on device
lockin = Lockin(name='uhfli', address='localhost', device="UHFLI", serial='DEV2793', demod_channels=[0,1])

#%% setup DAC channels

dac.set_bandwidth_safely([1], high_bw=True)
dac.activate_dac_channels([1])

# %%

st = Station("test")
st.instruments = [dac, lockin]

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

V_gate = st.add_parameter("V_gate", "V_gate", dac.ch1.voltage)

# %%

lockin_r1 = st.add_parameter("lockin_amp1", "Lockin R1", lockin.R1)
lockin_p1 = st.add_parameter("lockin_p1", "Lockin P1", lockin.P1)

lockin_r2 = st.add_parameter("lockin_amp2", "Lockin R2", lockin.R2)
lockin_p2 = st.add_parameter("lockin_p2", "Lockin P2", lockin.P2)

#%%

min_delay = 0.03 *1e-3

# choose reasonable sampling rate (Sa/s) - around 1/min_delay , else you might observe a fetch error
def sample_rate(x = None, demod_number = 0):
    demod = lockin.core.demods[demod_number]
    if x is None:
        return demod.rate()
    else:
        return demod.rate(x)
    
for demod in [0,1]:
    sample_rate(x = 2 * 1/min_delay, demod_number = demod)


# lockin frequency - should be around 1/min_delay, else you might observe steps
lockin.frequency1(2 * 1/min_delay)

#%%

def get_trigger_channel(basel_dac_object: Parameter | Sweep):
    
    if isinstance(basel_dac_object, Sweep):
        ch = basel_dac_object.parameter[0].instrument._channum
    elif isinstance(basel_dac_object, Parameter):
        ch = basel_dac_object.instrument._channum

    if ch in np.arange(1,13,1):  # Low dac board - contains AWG A and B - connect sync out A to uhfli trigger input 3
        return 3
    elif ch in np.arange(13,25,1):  # # High dac board - contains AWG C and D - connect sync out C to uhfli trigger input 4
        return 4


#%%

def V_gate_sweep_stepped(V_start=0, V_stop=0.1, num=100, delay = 300 *1e-3):

    gate_sweep = Sweep(V_gate, V_start, V_stop, num=num, start_delay=delay, delay=delay)

    run_dict["experiment_name"] = "test stepped Basel DAC 1d with uhfli - transistor"

    V_gate(0)
    time.sleep(0.5)

    measure.run(
        [gate_sweep],
        dependents=[lockin_r1, lockin_p1, lockin_r2, lockin_p2],
        no_hashing=True,
        **run_dict,
    )

#%%

def V_gate_sweep_buffered(V_start=0, V_stop=0.1, num=100, delay = 0.03 *1e-3, trigger_delay = 0 *1e-6, tc_factor = 1):

    gate_sweep = Sweep(V_gate, V_start, V_stop, num=num, start_delay=delay, delay=delay)

    buffered_sweep = {
        "instrument": NodeBaselDAC(inst=dac),
        "sweeps": [gate_sweep],
        "nodes": [
            {
                "instrument": NodeUHFLI(inst=lockin),
                "dependent": [lockin_r1, lockin_p1, lockin_r2, lockin_p2],
                "demod_channels": [0,1],
                "grid_mode": "linear",
                "input_trigger": get_trigger_channel(V_gate),
                "trigger_level": 1,
                "tc_factor": tc_factor,
                "trigger_delay": trigger_delay, # should be delay minus some bit
            }
        ],
    }

    run_dict["experiment_name"] = "test buffered Basel DAC 1d with uhfli - transistor"

    V_gate(0)
    time.sleep(0.5)

    measure.run(
        [buffered_sweep],
        dependents=[],
        no_hashing=True,
        **run_dict,
    )


#%% compare buffered with stepped sweep

V_gate_sweep_stepped(delay = 0.03 *1e-3)
sleep(0.5)

#%%

for trig in [-20, -10, -5, 0, 2, 5, 10, 20, 50]:
    V_gate_sweep_buffered(delay = 30*1e-6, trigger_delay = trig *1e-6)
    sleep(0.5)

#%%

for tc in [1,2,3,4,5]:
    V_gate_sweep_buffered(delay=0.3*1e-3, trigger_delay=7*1e-6, tc_factor=tc)


# %%

Instrument.close_all()

# %%
