# %%

import time

from drivers.squad.helpers.helpers import Lockin
from qcodes.instrument import Instrument

from qcutils import measure
from qcutils.buffered.instruments import NodeUHFLI
from qcutils.measure import Station
from qcutils.sweep import Sweep

# %%

Instrument.close_all()

lockin = Lockin(
    name="uhfli",
    address="localhost",
    device="UHFLI",
    serial="DEV2793",
    demod_channels=[0, 1],
)

# %%

st = Station("test")
st.instruments = [lockin]

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
    "data_location": r"D:\Measurements\Data",  # <-- choose correct folder
}

# %%

lockin_r1 = st.add_parameter("lockin_amp1", "Lockin R1", lockin.R1)
lockin_p1 = st.add_parameter("lockin_p1", "Lockin P1", lockin.P1)

lockin_r2 = st.add_parameter("lockin_amp2", "Lockin R2", lockin.R2)
lockin_p2 = st.add_parameter("lockin_p2", "Lockin P2", lockin.P2)


# %%


def buffered_frequency_sweep():
    # attention bandwidth mode is auto, going to lower frequencies will lead to longer measurements times!
    # sweep.delay or sweep.start_delay are not accessed, use

    f_sweep = Sweep(lockin.frequency1, 1e5, 1e6, num=501, start_delay=0, delay=0)

    buffered_sweep = {
        "instrument": NodeUHFLI(inst=lockin),
        "sweeps": [f_sweep],
        "dependent": [lockin_r1, lockin_p1, lockin_r2, lockin_p2],
        "phase_unwrap": True,
        "settling_inaccuracy": 0.1 * 1e-3,
        "acquisition": "sweeper",
    }

    run_dict["experiment_name"] = "test buffered UHFLI frequency sweep"

    lockin.frequency1(1e5)
    time.sleep(0.5)

    measure.run(
        [buffered_sweep],
        dependents=[],
        no_hashing=True,
        **run_dict,
    )


# %%


def buffered_amplitude_sweep():
    # attention bandwidth mode is auto, going to lower frequencies will lead to longer measurements times!
    # sweep.delay or sweep.start_delay are not accessed, use

    amp_sweep = Sweep(
        lockin.out1_amplitude1, 1e-3, 10e-3, num=101, start_delay=0, delay=0
    )

    buffered_sweep = {
        "instrument": NodeUHFLI(inst=lockin),
        "sweeps": [amp_sweep],
        "dependent": [lockin_r1, lockin_p1, lockin_r2, lockin_p2],
        "phase_unwrap": True,
        "settling_inaccuracy": 0.1 * 1e-3,
        "acquisition": "sweeper",
    }

    run_dict["experiment_name"] = "test buffered UHFLI amplitude sweep"

    lockin.out1_amplitude1(1e-3)
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
