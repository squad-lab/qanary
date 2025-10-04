# %%

from qcodes.instrument import Instrument
from qcodes_contrib_drivers.drivers.QDevil.QDAC2 import QDac2

from drivers.squad.helpers.helpers import Lockin
from qcutils.buffered.instruments import NodeMFLI, NodeQDAC2
from qcutils.buffered.sweep import arm_instruments, fetch_results
from qcutils.sweep import Sweep

# %%
Instrument.close_all()
mfli1 = Lockin(name="mfli1", address="192.168.0.104", serial="DEV7128")
dac = QDac2("dac", "TCPIP0::qdevil_dac_1.lab.squad-lab.org::5025::SOCKET")

sw = Sweep(dac.ch24.dc_constant_V, start=0, stop=0.1, num=101, delay=0.01)
buffered_sweep = {
    "instrument": NodeQDAC2(inst=dac),
    "sweep": sw,
    "output_trigger": 1,
    "trigger_type": "step",  # single, ramp or every
    "nodes": [
        {
            "instrument": NodeMFLI(inst=mfli1),
            "dependent": ["demods/0/sample.r"],
            "input_trigger": 1,
        }
    ],
}

# %%
toplevel = arm_instruments(buffered_sweep)
# %%
toplevel.run_sweep()
# %%
fetch_results(buffered_sweep)
# %%
