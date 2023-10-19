from qcodes import Instrument
from time import sleep


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

    def __init__(self, name, address, device="MFLI", serial=None, *args, **kwargs):
        super().__init__(f"wrapper_{name}", **kwargs)
        if serial:
            try:
                import zhinst.qcodes
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
                self.tc_sugg = 5 / self.frequency()

                filter_settling = {
                    1: 3 * self.tc(),
                    2: 4.7 * self.tc(),
                    3: 6.3 * self.tc(),
                    4: 7.8 * self.tc(),
                    5: 9.2 * self.tc(),
                    6: 11 * self.tc(),
                    7: 12 * self.tc(),
                    8: 13 * self.tc(),
                }
                self.t_avg = filter_settling[int(self.order())]

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

    def r_val(self, demods=0):
        return abs(
            self.core.demods[demods].sample()["x"][0]
            + 1j * self.core.demods[demods].sample()["y"][0]
        )

    def p_val(self, demods=0):
        return self.core.demods[demods].sample()["phase"][0]

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
        if type(self.current) == float or type(self.current) == int:
            return self.current / self.curr_ampl
        else:
            return self.current() / self.curr_ampl

    def get_voltage(self):
        if type(self.voltage) == float or type(self.voltage) == int:
            return self.voltage / self.volt_ampl
        else:
            return self.voltage() / self.volt_ampl

    def get_resistance(self):
        return (self.get_voltage() / self.get_current()) - self.line_resistance

    def get_conductance(self):
        sleep(self.delay)
        return (1 / self.get_resistance()) * (1 / self.cond_quantum)


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
