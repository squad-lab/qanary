# %%
from qcdrivers.buffered.qdevil import NodeQDAC2
from qcdrivers.buffered.zurich import NodeMFLI
from qcdrivers.squad.helpers.helpers import Lockin
from qcodes.instrument import Instrument
from qcodes.instrument_drivers.mock_instruments import (
    DummyInstrument,
    DummyInstrumentWithMeasurement,
)
from qcodes_contrib_drivers.drivers.QDevil.QDAC2 import QDac2

from qanary import measure
from qanary.measure import Station
from qanary.sweep import Sweep

# %%
Instrument.close_all()
# A dummy signal generator with two parameters ch1 and ch2
dac_dummy = DummyInstrument("dac_dummy", gates=["ch1", "ch2"])
dmm = DummyInstrumentWithMeasurement("dmm", setter_instr=dac_dummy)
mfli1 = Lockin(name="mfli1", address="192.168.0.104", serial="DEV7128")
dac = QDac2("dac", "TCPIP0::qdevil_dac_1.lab.squad-lab.org::5025::SOCKET")

# %%
st = Station("test_station")
st.instruments = [dac, dac_dummy, dmm]
dummy_dac_ch1 = st.add_parameter("dac_ch1", "Voltage 1", dac_dummy.ch1)
dummy_dac_ch2 = st.add_parameter("dac_ch2", "Voltage 2", dac_dummy.ch2)
dac_ch24 = st.add_parameter("dac_ch24", "Channel 24", dac.ch24.dc_constant_V)
dac_ch23 = st.add_parameter("dac_ch23", "Channel 23", dac.ch23.dc_constant_V)
mfli_r = st.add_parameter("mfli_r", "Lockin Amplitude", mfli1.R)
mfli_p = st.add_parameter("mfli_p", "Lockin Phase", mfli1.P)

dmm_v = st.add_parameter("dmm_v", "Voltmeter", dmm.v1)

# %%
gate_sweep_dummy1 = Sweep(dummy_dac_ch1, 0, 1, num=40, delay=1e-6, start_delay=0)
gate_sweep_dummy2 = Sweep(dummy_dac_ch2, 0, 1, num=50, delay=1e-6, start_delay=0)
gate_sweep1 = Sweep(dac_ch24, start=0, stop=0.1, num=51, delay=0.01)
gate_sweep2 = Sweep(dac_ch23, start=0, stop=0.1, num=51, delay=0.01)

buffered_sweep = {
    "instrument": NodeQDAC2(inst=dac),
    "sweeps": [gate_sweep1, gate_sweep2],
    "output_trigger": 1,
    "trigger_type": "step",
    "nodes": [
        {
            "instrument": NodeMFLI(inst=mfli1),
            "dependent": [mfli_r, mfli_p],
            "input_trigger": 1,
        }
    ],
}

# %%
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
    "data_location": "D:/Measurements/Data/",
}
# %%
measure.run(
    [gate_sweep_dummy1, gate_sweep_dummy2, buffered_sweep],
    **run_dict,
)
# %%
