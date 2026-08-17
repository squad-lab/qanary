# Our own drivers are needed to prevent timing issues with some of the keysight DMMs. 34410a seemingly supports sample_time for ramped measurements, but 34461a does not. This driver removes all timing features so that the driver can be used without throwing errors in the DMM.
from drivers.keysight.dmms import Keysight34461A
from qcodes.instrument import Instrument
from qcodes_contrib_drivers.drivers.QDevil.QDAC2 import QDac2

from qcutils import measure
from qcutils.buffered.instruments import NodeKeysightDMM, NodeQDAC2
from qcutils.measure import Station
from qcutils.sweep import Sweep

Instrument.close_all()
dmm1 = Keysight34461A(
    "dmm1", "TCPIP0::rwth_keysight_dmm_1.lab.squad-lab.org::inst0::INSTR"
)
dmm2 = Keysight34461A(
    "dmm2", "TCPIP0::rwth_keysight_dmm_2.lab.squad-lab.org::inst0::INSTR"
)
dac = QDac2("dac", "TCPIP0::qdevil_dac_1.lab.squad-lab.org::5025::SOCKET")


st = Station("test_station")
st.instruments = [dac, dmm1, dmm2]
dac_ch23 = st.add_parameter("dac_ch23", "Channel 23", dac.ch23.dc_constant_V)
dac_ch24 = st.add_parameter("dac_ch24", "Channel 24", dac.ch24.dc_constant_V)

dmm_v1 = st.add_parameter("dmm_v3", "Voltmeter 1", dmm1.volt)  # connected to ch23
dmm_v2 = st.add_parameter("dmm_v4", "Voltmeter 2", dmm2.volt)  # connected to ch24

# These values may be treated as the absolute maximum speed these measurements can be done at. The actual speed is limited by the DMM
gate_sweep1 = Sweep(dac_ch23, start=0, stop=0.5, num=100, delay=0.005)
gate_sweep2 = Sweep(dac_ch24, start=0, stop=0.5, num=100, delay=0.005)

buffered_sweep_dmm = {
    "instrument": NodeQDAC2(inst=dac),
    "sweeps": [gate_sweep1, gate_sweep2],
    "output_trigger": 4,
    "trigger_type": "step",
    "trigger_width": 1e-3,
    "nodes": [
        {
            "instrument": NodeKeysightDMM(inst=dmm1),
            "dependent": [dmm_v1],
            "input_trigger": 1,
        },
        {
            "instrument": NodeKeysightDMM(inst=dmm2),
            "dependent": [dmm_v2],
            "input_trigger": 1,
        },
    ],
}
run_dict = {
    "dependents": [],
    "wafer_id": "test_wafer",
    "device_type": "test_device",
    "sample_name": "test_sample",
    "experiment_name": "test_measurement",
    "station": st,
    "location_return": True,
    "data_location": "D:/Measurements/Data/",
}

for _ in range(10):
    measure.run(
        [buffered_sweep_dmm],
        **run_dict,
    )
