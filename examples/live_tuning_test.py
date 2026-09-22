# %%

from qcdrivers.buffered.squad import NodeDummyAcquisition, NodeDummySweeper
from qcdrivers.squad.helpers import ShellInstrument

from qanary.live_tuning.live_tuning import LiveTuning
from qanary.measure import Station
from qanary.sweep import Sweep

from qcodes.instrument import Instrument


# %%

dummy_dac = ShellInstrument("dummy_dac", {})
dummy_acquisition = ShellInstrument("dummy_lockin", {})

# %%

dummy_acquisition.add_parameter(
    "gate_x",
    label="Dummy Gate X",
    unit="V",
    initial_value=0,
    get_cmd=None,
    set_cmd=None,
)

dummy_acquisition.add_parameter(
    "gate_y",
    label="Dummy Gate Y",
    unit="V",
    initial_value=0,
    get_cmd=None,
    set_cmd=None,
)

dummy_acquisition.add_parameter(
    "R",
    label="Dummy R",
    unit="V",
    get_cmd=lambda: 0.0,
)

dummy_acquisition.add_parameter(
    "P",
    label="Dummy Phase",
    unit="deg",
    get_cmd=lambda: 0.0,
)

# %%

dummy_dac.add_parameter(
    "V1",
    label="Dummy DAC 1",
    unit="V",
    set_cmd=lambda x: None,
)

dummy_dac.add_parameter(
    "V2",
    label="Dummy DAC 2",
    unit="V",
    set_cmd=lambda x: None,
)

dummy_dac.add_parameter(
    "V3",
    label="Dummy DAC 3",
    unit="V",
    set_cmd=lambda x: None,
)

# %%

st = Station("Test")
st.instruments = [dummy_dac, dummy_acquisition]

# %%

run_dict = {
    "wafer_id": "wafer1",
    "device_type": "QD",
    "sample_name": "device1",
    "station": st,
    "experiment_name": "live_tuning",
    "data_location": r"C:\Users\s.schreibing\Documents\qimchi_data",
    "fridge_name": "Radler",
}

# %%

v1 = st.add_parameter("V1", "DAC 1", dummy_dac.V1)
v2 = st.add_parameter("V2", "DAC 2", dummy_dac.V2)
v3 = st.add_parameter("V3", "DAC 3", dummy_dac.V3)

lockin_r = st.add_parameter("lockin_r", "Lock-in R", dummy_acquisition.R)
lockin_p = st.add_parameter("lockin_p", "Lock-in Phase", dummy_acquisition.P)

dummy_gate_x = st.add_parameter(
    "dummy_gate_x",
    "Dummy Gate X",
    dummy_acquisition.gate_x,
)

dummy_gate_y = st.add_parameter(
    "dummy_gate_y",
    "Dummy Gate Y",
    dummy_acquisition.gate_y,
)



# %%

v1_sweep = Sweep(v1, -1, 1, num=51, delay=0.001, start_delay=0.001)
v2_sweep = Sweep(v2, -1, 1, num=51, delay=0.001, start_delay=0.001)

buffered_sweep = {
    "instrument": NodeDummySweeper(inst=dummy_dac),
    "sweeps": [v1_sweep, v2_sweep],
    "nodes": [
        {
            "instrument": NodeDummyAcquisition(
                inst=dummy_acquisition,
                noise=0.02,
            ),
            "dependent": [
                lockin_p,
                lockin_r,
            ],
        }
    ],
}

# %%

tuning = LiveTuning(
    controls=[
        dummy_gate_x,
        dummy_gate_y,
    ],
    control_ranges=[
        (-1.0, 1.0),
        (-1.0, 1.0),
    ],
    refresh_interval=0.1,
)

# %%

tuning.start(
    [buffered_sweep],
    **run_dict,
    no_hashing=True
)

# %%

tuning.stop()

# %%

Instrument.close_all()

# %%
