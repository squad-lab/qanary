# %%

import time
from time import sleep

import numpy as np
from qcdrivers.basel.dacs.dacs import BaselDac2
from qcdrivers.buffered.basel import NodeBaselDAC
from qcdrivers.buffered.zurich import NodeMFLI
from qcdrivers.squad.helpers.helpers import Lockin
from qcodes.instrument import Instrument
from qcodes.parameters import Parameter

from qanary import measure
from qanary.measure import Station
from qanary.sweep import Sweep

# %%

Instrument.close_all()

dac = BaselDac2(
    "dac", "TCPIP0::192.168.0.108::23::SOCKET"
)  # Connection issues: use "Restart Telnet now!" on device
lockin = Lockin(name="mfli", address="localhost", serial="DEV7945")

# %% setup DAC channels

dac.set_bandwidth_safely([1], high_bw=True)
dac.activate_dac_channels([1])

# %%

st = Station("test")
st.instruments = [dac, lockin]

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

V_gate = st.add_parameter("V_gate", "V_gate", dac.ch1.voltage)

# %%

lockin_r = st.add_parameter("lockin_amp", "Lockin R", lockin.R)
lockin_p = st.add_parameter("lockin_p", "Lockin P", lockin.P)

# %%

min_delay = 0.03 * 1e-3


# choose reasonable sampling rate (Sa/s) - around 1/min_delay , else you might observe a fetch error
def sample_rate(x=None):
    demod = lockin.core.demods[0]
    if x is None:
        return demod.rate()
    else:
        return demod.rate(x)


# lockin frequency - should be around 1/min_delay, else you might observe steps
lockin.frequency(2 * 1 / min_delay)
sample_rate(2 * 1 / min_delay)

# %%


def get_trigger_channel(basel_dac_object: Parameter | Sweep):
    if isinstance(basel_dac_object, Sweep):
        ch = basel_dac_object.parameter[0].instrument._channum
    elif isinstance(basel_dac_object, Parameter):
        ch = basel_dac_object.instrument._channum

    if (
        ch in np.arange(1, 13, 1)
    ):  # Low dac board - contains AWG A and B - connect sync out A to mfli trigger input 1
        return 1
    elif (
        ch in np.arange(13, 25, 1)
    ):  # # High dac board - contains AWG C and D - connect sync out C to mfli trigger input 2
        return 2


# %%


def V_gate_sweep_stepped(V_start=0, V_stop=1, num=100, delay=300 * 1e-3):
    gate_sweep = Sweep(V_gate, V_start, V_stop, num=num, start_delay=delay, delay=delay)

    run_dict["experiment_name"] = "test stepped Basel DAC 1d with mfli - transistor"

    V_gate(0)
    time.sleep(0.5)

    measure.run(
        [gate_sweep],
        dependents=[lockin_r, lockin_p],
        no_hashing=True,
        **run_dict,
    )


# %%


def V_gate_sweep_buffered(
    V_start=0, V_stop=1, num=100, delay=30 * 1e-6, trigger_delay=0 * 1e-6, tc_factor=1
):
    gate_sweep = Sweep(V_gate, V_start, V_stop, num=num, start_delay=delay, delay=delay)

    buffered_sweep = {
        "instrument": NodeBaselDAC(inst=dac),
        "sweeps": [gate_sweep],
        "nodes": [
            {
                "instrument": NodeMFLI(inst=lockin),
                "dependent": [lockin_r, lockin_p],
                "grid_mode": "linear",
                "input_trigger": get_trigger_channel(V_gate),
                "trigger_level": 1,
                "tc_factor": tc_factor,
                "trigger_delay": trigger_delay,  # should be delay minus some bit
            }
        ],
    }

    run_dict["experiment_name"] = "test buffered Basel DAC 1d with mfli - transistor"

    V_gate(0)
    time.sleep(0.5)

    measure.run(
        [buffered_sweep],
        dependents=[],
        no_hashing=True,
        **run_dict,
    )


# %%

# test for sampling rate - use it large enough but nalso not to large (values will disappear)

for x in [100, 1e3, 1e4, 1e6]:
    sample_rate(x)
    print(f"Sampling rate: {x} Sa/s")
    sleep(0.5)
    V_gate_sweep_buffered(delay=0.3 * 1e-3, trigger_delay=2 * 1e-6)

# %%

# test for lock in frequency

for f in [123, 168, 1135, 2657, 10000]:
    lockin.frequency(f)
    print(f"Lockin frequency: {f} Hz")
    sleep(0.5)
    V_gate_sweep_buffered(delay=0.3 * 1e-3, trigger_delay=2 * 1e-6)


# %%

# compare buffered with stepped sweep

V_gate_sweep_stepped(delay=0.3 * 1e-3)
sleep(0.5)

# %%

delay = 0.3 * 1e-3

sample_rate(2 * 1 / delay)
lockin.frequency(2 * 1 / delay)
sleep(0.5)

V_gate_sweep_stepped(delay=delay)

# %%

for trig in [-20, -10, -5, 0, 2, 5, 10, 20, 50]:
    V_gate_sweep_buffered(delay=30 * 1e-6, trigger_delay=trig * 1e-6)
    sleep(0.5)

# %%

for tc in [1, 2, 3, 4, 5]:
    V_gate_sweep_buffered(delay=0.3 * 1e-3, trigger_delay=7 * 1e-6, tc_factor=tc)


# %%

Instrument.close_all()

# %%
