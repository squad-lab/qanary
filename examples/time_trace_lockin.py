# %%

import time

from drivers.squad.helpers.helpers import Delay, Lockin
from qcodes.instrument import Instrument

from qcutils import measure
from qcutils.buffered.instruments import NodeDelay, NodeMFLI, NodeUHFLI
from qcutils.measure import Station
from qcutils.sweep import Sweep

# %%

Instrument.close_all()

clock = Delay("delay_clock")

lockin_RF = Lockin(
    name="uhfli",
    address="localhost",
    device="UHFLI",
    serial="DEV2793",
    demod_channels=[0],
)
lockin_DC = Lockin("lockin_DC", address="localhost", device="MFLI", serial="DEV7984")

# %%

st = Station("test")
st.instruments = [clock, lockin_RF, lockin_DC]

# %%

extra_metadata = {"info": ""}

run_dict = {
    "wafer_id": "test_wafer",
    "device_type": "test_device",
    "sample_name": "test_time_sweep",
    "station": st,
    "fridge_name": "Radler",
    "metadata": extra_metadata,
    "location_return": True,
    "data_location": r"D:\Measurements\Data",  # <-- choose correct folder
}

# %%

lockin_RF_r1 = st.add_parameter("lockin_amp1", "Lockin R1", lockin_RF.R1)
lockin_RF_p1 = st.add_parameter("lockin_p1", "Lockin P1", lockin_RF.P1)

lockin_DC_r = st.add_parameter("lockin_DC_amp", "Lockin DC R", lockin_DC.R)
lockin_DC_p = st.add_parameter("lockin_DC_phase", "Lockin DC Phase", lockin_DC.P)

# %%

t = st.add_parameter("time", "Time", clock.time)

# %%


def UHFLI_time_trace_measurement(target_duration: float = 5, n_samples: int = 1000):
    """
    Measure a time trace of the lockin samples with DAQ using the lockin and delay instrument.

    Args:
        target_duration (float): The desired duration of the time trace in seconds.
        n_samples (int): The number of samples to acquire.

    """

    t_start = time.time()

    # calculate target delay and set sample rate accordingly
    target_sample_rate = n_samples / target_duration
    lockin_RF.sample_rate1(target_sample_rate)

    time.sleep(0.2)  # wait for the lockin to update its sample rate

    # check actual set sample rate and calculate point delay
    actual_sample_rate = lockin_RF.sample_rate1()
    point_delay = 1 / actual_sample_rate

    time_sweep = Sweep(
        t,
        point_delay,
        n_samples * point_delay,
        n_samples,
        start_delay=0,
        delay=point_delay,
    )

    buffered_sweep = {
        "instrument": NodeDelay(inst=clock),
        "sweeps": [time_sweep],
        "nodes": [
            {
                "instrument": NodeUHFLI(inst=lockin_RF),
                "dependent": [lockin_RF_r1, lockin_RF_p1],
                "tc_factor": 1,
                "demod_channels": [0],
                "grid_mode": "exact",
                "force_trigger": True,
            }
        ],
    }

    run_dict["experiment_name"] = "UHFLI lockin time trace measurement"

    measure.run(
        [buffered_sweep],
        dependents=[],
        **run_dict,
    )

    print(
        f"UHFLI lockin time trace measurement completed in {time.time() - t_start:.2f} s."
    )


# %%


def MFLI_time_trace_measurement(target_duration: float = 5, n_samples: int = 1000):
    """
    Measure a time trace of the lockin samples with DAQ using the lockin and delay instrument.

    Args:
        target_duration (float): The desired duration of the time trace in seconds.
        n_samples (int): The number of samples to acquire.

    """

    t_start = time.time()

    # calculate target delay and set sample rate accordingly
    target_sample_rate = n_samples / target_duration
    lockin_DC.sample_rate(target_sample_rate)

    time.sleep(0.2)  # wait for the lockin to update its sample rate

    # check actual set sample rate and calculate point delay
    actual_sample_rate = lockin_DC.sample_rate()
    point_delay = 1 / actual_sample_rate

    time_sweep = Sweep(
        t,
        point_delay,
        n_samples * point_delay,
        n_samples,
        start_delay=0,
        delay=point_delay,
    )

    buffered_sweep = {
        "instrument": NodeDelay(inst=clock),
        "sweeps": [time_sweep],
        "nodes": [
            {
                "instrument": NodeMFLI(inst=lockin_DC),
                "dependent": [lockin_DC_r, lockin_DC_p],
                "tc_factor": 1,
                "grid_mode": "exact",
                "force_trigger": True,
            }
        ],
    }

    run_dict["experiment_name"] = "MFLI lockin time trace measurement"

    measure.run(
        [buffered_sweep],
        dependents=[],
        **run_dict,
    )

    print(
        f"MFLI lockin time trace measurement completed in {time.time() - t_start:.2f} s."
    )


# %%

# stepped lockin time trace - not recommended for long traces, as it will take a long time to complete


def time_trace_measurement(target_duration: float = 1, n_samples: int = 100):
    t_start = time.time()

    # calculate delay
    point_delay = target_duration / n_samples

    time_sweep = Sweep(
        t,
        point_delay,
        n_samples * point_delay,
        n_samples,
        start_delay=point_delay,
        delay=point_delay,
    )

    run_dict["experiment_name"] = "Lockin stepped time trace measurement"

    measure.run(
        [time_sweep],
        dependents=[lockin_DC_r, lockin_DC_p, lockin_RF_r1, lockin_RF_p1],
        **run_dict,
    )

    print(
        f"Lockin stepped time trace measurement completed in {time.time() - t_start:.2f} s."
    )


# %%

# %%

Instrument.close_all()

# %%
