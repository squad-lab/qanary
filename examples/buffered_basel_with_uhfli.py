# %%

import time

import numpy as np
from qcdrivers.basel.dacs.dacs import BaselDac2
from qcdrivers.buffered.basel import NodeBaselDAC
from qcdrivers.buffered.zurich import NodeUHFLI
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
lockin = Lockin(
    name="uhfli",
    address="localhost",
    device="UHFLI",
    serial="DEV2793",
    demod_channels=[0, 1],
)

# %% setup DAC channels

dac.set_bandwidth_safely([1, 13], high_bw=True)
dac.activate_dac_channels([1, 13])

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

gate1 = st.add_parameter("gate1", "Dac channel 1", dac.ch1.voltage)
gate2 = st.add_parameter("gate2", "Dac channel 13", dac.ch13.voltage)

# %%

lockin_r1 = st.add_parameter("lockin_amp1", "Lockin R1", lockin.R1)
lockin_p1 = st.add_parameter("lockin_p1", "Lockin P1", lockin.P1)

lockin_r2 = st.add_parameter("lockin_amp2", "Lockin R2", lockin.R2)
lockin_p2 = st.add_parameter("lockin_p2", "Lockin P2", lockin.P2)

# choose reasonable sampling rate (Sa/s) - around 1/min_delay , else you might observe a fetch error
for demod in [0, 1]:
    lockin.core.demods[demod].rate(50000)

# %%


def get_trigger_channel(basel_dac_object: Parameter | Sweep):
    if isinstance(basel_dac_object, Sweep):
        ch = basel_dac_object.parameter[0].instrument._channum
    elif isinstance(basel_dac_object, Parameter):
        ch = basel_dac_object.instrument._channum

    if (
        ch in np.arange(1, 13, 1)
    ):  # Low dac board - contains AWG A and B - connect sync out A to mfli trigger input 3
        return 3  # 1
    elif (
        ch in np.arange(13, 25, 1)
    ):  # # High dac board - contains AWG C and D - connect sync out C to mfli trigger input 4
        return 4  # 2


# %%


def test_1d_sweep(gate):
    gate_sweep = Sweep(gate, 0, 1, num=10, start_delay=0, delay=0.03 * 1e-3)

    buffered_sweep = {
        "instrument": NodeBaselDAC(inst=dac),
        "sweeps": [gate_sweep],
        "nodes": [
            {
                "instrument": NodeUHFLI(inst=lockin),
                "dependent": [lockin_r1, lockin_p1, lockin_r2, lockin_p2],
                "demod_channels": [0, 1],
                "grid_mode": "exact",
                "input_trigger": get_trigger_channel(gate),
                "trigger_delay": 2 * 1e-6,
            }
        ],
    }

    run_dict["experiment_name"] = "test buffered Basel DAC 1d with uhfli"

    gate(0)
    time.sleep(0.5)

    measure.run(
        [buffered_sweep],
        dependents=[],
        no_hashing=True,
        **run_dict,
    )


# %%

# hardware wise: the two gates must be on separate dac boards (high and low) - connect Sync out A into trig C and Sync out C into trig A (small soldered adapter on backside)
# ensure high bandwidth on the used buffered channels if you want to use a small delay !

# you must keep DAQ closed on lock in and trigger input impedance to 1k Ohm and sampling rate high enough, else fetch will fail!


def test_2d_sweep(gate1, gate2, num1=5, num2=3, delay=0.03 * 1e-3):
    # both delays must be the same, start_delay is ignored for buffered sweep
    gate1_sweep = Sweep(gate1, 0, 1, num=num1, start_delay=0, delay=delay)
    gate2_sweep = Sweep(gate2, 0, 1, num=num2, start_delay=0, delay=delay)

    sweep_list = [
        gate1_sweep,
        gate2_sweep,
    ]  # determine inner and outer sweep order - inner sweep is always the second one!

    buffered_sweep = {
        "instrument": NodeBaselDAC(inst=dac),
        "sweeps": sweep_list,
        "nodes": [
            {
                "instrument": NodeUHFLI(inst=lockin),
                "dependent": [lockin_r1, lockin_p1, lockin_r2, lockin_p2],
                "demod_channels": [0, 1],
                "grid_mode": "exact",
                "input_trigger": get_trigger_channel(
                    sweep_list[-1]
                ),  # must be the inner one!
                "trigger_delay": 2 * 1e-6,
            }
        ],
    }

    run_dict["experiment_name"] = "test buffered Basel DAC 2d with uhfli"

    gate1(0)
    gate2(0)
    time.sleep(0.5)

    measure.run(
        [buffered_sweep],
        dependents=[],
        no_hashing=True,
        **run_dict,
    )


# %%

Instrument.close_all()

# %%
