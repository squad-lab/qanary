import warnings
from time import sleep
from typing import Any, Optional

import numpy as np
from qcodes import Instrument
from qcodes import validators as vals
from qcodes.instrument import VisaInstrument

warnings.warn(
    "Please use https://git.squad-lab.org/squad-lab/measurements/drivers instead",
    DeprecationWarning,
    stacklevel=2,
)


class ShellInstrument(Instrument):
    def __init__(self, name: str, parameters: dict, **kwargs) -> None:
        """Shell instrument class for defining qcodes parameters from instruments with oddly behaving parameters, or to define new instruments with custom parameters

        Args:
            name: name of the qcodes instrument
            parameters (dict): dictionary of parameters to be added to the instrument, kwargs to qcodes.Instrument.add_parameter()
        """

        super().__init__(name, **kwargs)

        for key, parameter in parameters.items():
            self.add_parameter(key, **parameter)


class Lockin(Instrument):
    """Wrapper class for lockin amplifiers, currently only supports SR830 and MFLI

    Args:
        name: name of the qcodes instrument
        address: address of the lockin amplifier (localhost or GPIB address)
        device: device type
        serial: serial number of the lockin amplifier, only required for MFLI
    """

    def __init__(
        self, name, address, device="MFLI", serial=None, *args, **kwargs
    ) -> None:
        super().__init__(f"wrapper_{name}", **kwargs)
        if serial:
            try:
                pass
            except ImportError:
                raise ImportError(f"Please install zhinst-qcodes to use the {device}")

            if device == "MFLI":
                from zhinst.qcodes import MFLI

                self.core = MFLI(
                    name=name,
                    host=address,
                    interface="1GbE",
                    serial=serial,
                    *args,
                    **kwargs,
                )
                self.frequency = self.core.oscs[0].freq
                self.amplitude = self.core.sigouts[0].amplitudes[1].value
                self.on = self.sigouts[0].on

                self.add = self.core.sigouts[0].add
                self.diff = self.core.sigouts[0].diff

                self.sinc = self.core.demods[0].sinc
                self.harmonic = self.core.demods[0].harmonic

                self.ac = self.core.sigins[0].ac
                self.tc = self.core.demods[0].timeconstant
                self.order = self.core.demods[0].order

                self.autosigout = self.core.sigouts[0].autorange
                self.autovoltin = self.core.sigins[0].autorange
                self.autocurrin = self.core.sigins[0].autorange

                self.add_parameter(
                    "R",
                    label="R",
                    get_parser=float,
                    get_cmd=self.r_val,
                )

                self.add_parameter(
                    "P",
                    label="P",
                    get_parser=float,
                    get_cmd=self.p_val,
                    unit="deg",
                )

            elif device == "UHFLI":
                # Still have to add all of the parameters here
                from zhinst.qcodes import UHFLI

                self.core = UHFLI(
                    name=f"{name}_core",
                    host=address,
                    interface="1GbE",
                    serial=serial,
                    *args,
                    **kwargs,
                )

                for demod in range(len(self.core.demods)):
                    self.add_parameter(
                        f"R{demod}",
                        label=f"R{demod}",
                        get_parser=float,
                        get_cmd=self.r_val,
                        demods=demod,
                    )

                    self.add_parameter(
                        f"P{demod}",
                        label=f"P{demod}",
                        get_parser=float,
                        get_cmd=self.p_val,
                        demods=demod,
                        unit="deg",
                    )

        else:
            # Still have to add all the parameters here
            from qcodes.instrument_drivers.stanford_research import SR830

            device = "SR830"
            self.core = SR830(f"{name}_core", address, *args, **kwargs)
            self.sinc = self.core.sync_filter
            self.tc = self.core.time_constant
            self.order = self.filter_slope

    def delay(self, order, tc) -> float:
        filter_settling = {
            1: 3 * tc,
            2: 4.7 * tc,
            3: 6.3 * tc,
            4: 7.8 * tc,
            5: 9.2 * tc,
            6: 11 * tc,
            7: 12 * tc,
            8: 13 * tc,
        }
        return filter_settling[int(order)]

    def r_val(self, demods=0) -> float:
        return abs(
            self.core.demods[demods].sample()["x"][0]
            + 1j * self.core.demods[demods].sample()["y"][0]
        )

    def p_val(self, demods=0) -> float:
        return self.core.demods[demods].sample()["phase"][0]

    def get_idn(self) -> dict:
        return self.core.get_idn()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return self.core.__getattr__(name)


class Conductance(Instrument):
    def __init__(
        self,
        name,
        current,
        voltage,
        curr_ampl,
        volt_ampl,
        volt_divider=1.0,
        delay=0.1,
        resistance=0.0,
    ):
        """Conductance instrument class for calculating the conductance from a current and voltage parameter

        Args:
            name: name of the qcodes instrument
            current: current parameter or float
            voltage: voltage parameter or float
            curr_ampl: current amplification
            volt_ampl: voltage amplification
            volt_divider: voltage divider
            delay: delay at each measurement
            resistance: resistance of the line
        """

        super().__init__(name)

        from scipy.constants import physical_constants

        self.cond_quantum = physical_constants["conductance quantum"][0]
        self.line_resistance = resistance
        self.delay = delay
        self.curr_ampl = curr_ampl
        self.volt_ampl = volt_ampl / volt_divider

        self.current = current
        self.voltage = voltage

        self.add_parameter(
            "value",
            label="G",
            get_parser=float,
            get_cmd=self.get_conductance,
            unit="G0",
        )

    def get_current(self):
        if isinstance(self.current, (float, int)):
            return self.current / self.curr_ampl
        else:
            return self.current() / self.curr_ampl

    def get_voltage(self):
        if isinstance(self.voltage, (float, int)):
            return self.voltage / self.volt_ampl
        else:
            return self.voltage() / self.volt_ampl

    def get_resistance(self):
        return (self.get_voltage() / self.get_current()) - self.line_resistance

    def get_conductance(self):
        sleep(self.delay)
        return (1 / self.get_resistance()) * (1 / self.cond_quantum)

    def get_idn(self) -> dict:
        idn_dict = {
            "vendor": "Conductance Wrapper",
            "model": "1.0",
            "serial": "1.0",
            "firmware": 1,
        }
        return idn_dict


class Delay(Instrument):
    def __init__(self, name, delay=0.1):
        super().__init__(name)

        self.num_time = 0
        self.delay = delay

        self.add_parameter(
            "time",
            label="Time Delay",
            set_cmd=self.set_delay,
            unit=f"x ({delay}s)",
        )

    def set_delay(self, number):
        self.num_time += 1
        sleep(self.delay)


class BaselSP1004a(VisaInstrument):
    """
    A driver for Basel Preamp's (SP1004a) Remote Instrument - Model SP1004a.

    Args:
        name: name for your instrument driver instance
        address: address of the connected remote controller of basel preamp
    """

    def __init__(
        self,
        name: str,
        address: str,
        terminator: str = "\r\n",
        **kwargs: Any,
    ) -> None:
        super().__init__(name, address, terminator=terminator, **kwargs)

        self.connect_message()

        self.add_parameter(
            "gain",
            label="Gain",
            unit="",
            set_cmd=self._set_gain,
            get_cmd=self._get_gain,
            vals=vals.Enum(1e2, 1e3, 1e4),
        )
        self.add_parameter(
            "fcut",
            unit="Hz",
            label="Filter Cut-Off Frequency",
            get_cmd=self._get_filter,
            get_parser=self._parse_filter_value,
            set_cmd=self._set_filter,
            val_mapping={
                100: "100",
                300: "300",
                1000: "1k",
                3000: "3k",
                10e3: "10k",
                30e3: "30k",
                100e3: "100k",
                300e3: "300k",
                1e6: "FULL",
            },
        )
        self.add_parameter(
            "overload_status", label="Overload Status", set_cmd=False, get_cmd="GET O"
        )

        self.add_parameter(
            "vin_offset_compensated",
            label="Overload Status",
            set_cmd=False,
            get_cmd="GET C",
        )

    def get_idn(self) -> dict[str, Optional[str]]:
        vendor = "Physics Basel"
        model = "SP 1004A"
        serial = None
        firmware = None
        return {
            "vendor": vendor,
            "model": model,
            "serial": serial,
            "firmware": firmware,
        }

    def _set_gain(self, value: float) -> None:
        r = self.ask(f"SET G 1E{int(np.log10(value))}")
        if r != "OK":
            raise ValueError(f"Expected OK return but got: {r}")

    def _get_gain(self) -> float:
        s = self.ask("GET G")
        r = s.split("Gain: ")[1]
        return float(r)

    def _set_filter(self, value: str) -> None:
        r = self.ask(f"SET F {value}")
        if r != "OK":
            raise ValueError(f"Expected OK return but got: {r}")

    def _get_filter(self) -> str:
        s = self.ask("GET F")
        return s.split("Filter: ")[1]

    @staticmethod
    def _parse_filter_value(val: str) -> str:
        if val.startswith("F"):
            return "FULL"
        elif val[-3::] == "kHz":
            return str(int(val[0:-3])) + "k"

        elif val[-2::] == "Hz":
            return str(int(val[0:-2]))
        else:
            raise ValueError(f"Could not interpret result. Got: {val}")
