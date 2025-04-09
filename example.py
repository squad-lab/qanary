# %%
from qcodes.instrument_drivers.mock_instruments import (
    DummyInstrument,
    DummyInstrumentWithMeasurement,
)

from qcodes.instrument import Instrument
from qcutils.sweep import Sweep
from qcutils import measure
from qcutils.measure import Station

# %%
Instrument.close_all()
# A dummy signal generator with two parameters ch1 and ch2
dac = DummyInstrument("dac", gates=["ch1", "ch2"])
dmm = DummyInstrumentWithMeasurement("dmm", setter_instr=dac)

# %%
st = Station("test_station")
st.instruments = [dac, dmm]
dac_ch1 = st.add_parameter("dac_ch1", "Voltage 1", dac.ch1)
dac_ch2 = st.add_parameter("dac_ch2", "Voltage 2", dac.ch2)
dmm_v = st.add_parameter("dmm_v", "Voltmeter", dmm.v1)

# %%
gate_sweep1 = Sweep(dac_ch1, 0, 1, num=40, delay=1e-6, start_delay=0)
gate_sweep2 = Sweep(dac_ch2, 0, 1, num=50, delay=1e-6, start_delay=0)

extra_metadata = {"test": "test", "test2": 2}
run_dict = {
    "dependents": [dmm_v],
    "wafer_id": "test_wafer",
    "device_type": "test_device",
    "sample_name": "test_sample",
    "experiment_name": "test_measurement",
    "station": st,
    "metadata": extra_metadata,
    "location_return": True,
    "data_location": "D:/Measurement/Data/",
}
# %%
measure.run(
    [gate_sweep1, gate_sweep2],
    **run_dict,
)
# %%
