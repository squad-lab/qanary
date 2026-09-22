#%%

#from qcdrivers.buffered.qdevil import NodeQdac2
#from qcdrivers.buffered.zurich import NodeMFLI

from qcdrivers.squad.helpers import Delay
from qcdrivers.buffered.squad import NodeDelay, NodeDummyAcquisition

from qanary.live_tuning import LiveTuning
from qanary.measure import Measurement, Station
from qanary.sweep import Sweep

from qcodes.instrument import Instrument

#%%

class DummyAcquisition(Instrument):
    """
    Minimal QCoDeS dummy Acquisition Instrument (like a lock-in or dmm).

    R and P are ordinary QCoDeS parameters so Qanary can use them
    as dependents and attach metadata to the xarray Dataset.
    """

    def __init__(self, name: str):
        super().__init__(name)

        self.add_parameter(
            "R",
            label="Dummy R",
            unit="V",
            get_cmd=lambda: 0.0,
        )

        self.add_parameter(
            "P",
            label="Dummy Phase",
            unit="deg",
            get_cmd=lambda: 0.0,
        )

    def get_idn(self):
        return {
            "vendor": "SQUAD",
            "model": "Dummy Lockin",
            "serial": "DUMMY",
            "firmware": None,
        }


#%%

clock = Delay("clock")
dummy_acquisition = DummyAcquisition("dummy_lockin")

#%%

st = Station("Test")
st.instruments = [clock, dummy_acquisition]

#%%

run_dict = {
    "wafer_id": "wafer1",
    "device_type": "QD",
    "sample_name": "device1",
    "experiment_name": "live_tuning",
    "data_location": r"C:\Users\s.schreibing\Documents\qimchi_data",
    "fridge_name": "Radler",
}

#%%

t = st.add_parameter("t", "Time", clock.time)

lockin_r = st.add_parameter("lockin_r", "Lock-in R", dummy_acquisition.R)
lockin_p = st.add_parameter("lockin_p", "Lock-in Phase", dummy_acquisition.P)

#%%


time_sweep = Sweep(t, 0, 1, num=101, delay=0.01, start_delay=0.01)

buffered_sweep = {
    "instrument": NodeDelay(inst=clock),
    "sweeps": [
        time_sweep,
    ],
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

#%%

tuning = LiveTuning(
    station=st,
    controls=[],
    refresh_interval=1,
)

#%%

tuning.start(
    [buffered_sweep],
    dependents=[],
    **run_dict,
    no_hashing=True
)

#%%

tuning.stop()

#%%

Instrument.close_all()

# %%
