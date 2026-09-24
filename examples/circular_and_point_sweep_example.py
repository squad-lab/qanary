from qcodes.instrument import Instrument

from qanary import measure
from qanary.measure import Station
from qanary.sweep import CircularSweep, PointSweep, Sweep

counter = {"value": 0}


def get_signal():
    counter["value"] += 1
    return counter["value"]


class ShellInstrument(Instrument):
    def __init__(self, name: str, parameters: dict, **kwargs) -> None:
        """
        Shell instrument class for defining qcodes parameters from instruments with oddly behaving parameters, or to define new instruments with custom parameters

        Args:
            name: name of the qcodes instrument
            parameters (dict): dictionary of parameters to be added to the instrument, kwargs to qcodes.Instrument.add_parameter()
        """

        super().__init__(name, **kwargs)

        for key, parameter in parameters.items():
            self.add_parameter(key, **parameter)

    def get_idn(self) -> dict[str, str | None]:
        return {
            "vendor": "SQUAD Lab",
            "model": "Shell Instrument",
            "serial": None,
            "firmware": None,
        }


### dac like dummy ###
dummy = ShellInstrument("dummy", {})

v1 = None
v2 = None

dummy.add_parameter(
    "ch1",
    label="Channel 1",
    unit="V",
    set_cmd=lambda value: globals().update(v1=value),
    get_cmd=lambda: None,
)

dummy.add_parameter(
    "ch2",
    label="Channel 2",
    unit="V",
    set_cmd=lambda value: globals().update(v2=value),
    get_cmd=lambda: None,
)

####################


### lock in or dmm like dummy###
dummy2 = ShellInstrument("dummy2", {})

dummy2.add_parameter(
    "signal",
    label="Signal",
    unit="V",
    get_cmd=get_signal,
    snapshot_get=False,
)

################################

st = Station("test")
st.instruments = [dummy, dummy2]


run_dict = {
    "wafer_id": "test",
    "device_type": "test",
    "sample_name": "test",
    "station": st,
    "experiment_name": "",
    "fridge_name": "",
    "location_return": True,
    "data_location": r"C:\Users\s.schreibing\Documents\qimchi_data",  # <-- choose correct folder
}

dummy_ch1 = st.add_parameter("ch1", "Dummy channel 1", dummy.ch1)
dummy_ch2 = st.add_parameter("ch2", "Dummy channel 2", dummy.ch2)

dummy2_signal = st.add_parameter("signal", "Dummy signal", dummy2.signal)


point_sweep = PointSweep(dummy_ch1, points=[1, 2, 1, 2], delay=0.1, start_delay=0.1)
print(vars(point_sweep))

run_dict["experiment_name"] = "PointSweep_test"

counter = {"value": 0}
measure.run([point_sweep], dependents=[dummy2_signal], no_hashing=True, **run_dict)


circ_sweep = CircularSweep(
    dummy_ch2, start=1, stop=4, num=4, delay=0.1, start_delay=0.1
)
print(vars(circ_sweep))

run_dict["experiment_name"] = "CircularSweep_test"

counter = {"value": 0}
measure.run([circ_sweep], dependents=[dummy2_signal], no_hashing=True, **run_dict)


sweep = Sweep(dummy_ch2, start=1, stop=7, num=7, delay=0.1, start_delay=0.1)
print(vars(sweep))

run_dict["experiment_name"] = "Sweep_test"

counter = {"value": 0}
measure.run([sweep], dependents=[dummy2_signal], no_hashing=True, **run_dict)


Instrument.close_all()
