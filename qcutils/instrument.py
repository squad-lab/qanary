from qcodes import Instrument


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
    def __init__(self, name, address, device="MFLI", serial=None, *args, **kwargs):
        super().__init__(f"wrapper_{name}", **kwargs)
        if serial:
            try:
                import zhinst.qcodes
            except ImportError:
                raise ImportError(f"Please install zhinst-qcodes to use the {device}")
            
            if device == "MFLI":
                from zhinst.qcodes import MFLI
                self.core = MFLI(name=name, host=address, interface="1GbE", serial=serial, *args, **kwargs)
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
                
                self.add_parameter('R',
                                   label="R",
                                   get_parser=float,
                                   get_cmd=self.r_val,)
                
                self.add_parameter('P',
                                   label="P",
                                   get_parser=float,
                                   get_cmd=self.p_val,
                                   unit="deg",)

            elif device == "UHFLI":
                # Still have to add all of the parameters here
                from zhinst.qcodes import UHFLI
                self.core = UHFLI(name=f"{name}_core", host=address, interface="1GbE", serial=serial, *args, **kwargs)
                
                for demod in range(len(self.core.demods)):
                    self.add_parameter(f"R{demod}",
                                       label=f"R{demod}",
                                       get_parser=float,
                                       get_cmd=self.r_val,
                                       demods=demod)
                    
                    self.add_parameter(f"P{demod}",
                                       label=f"P{demod}",
                                       get_parser=float,
                                       get_cmd=self.p_val,
                                       demods=demod,
                                       unit="deg",)

        else:
            # Still have to add all the parameters here
            from qcodes.instrument_drivers.stanford_research import SR830
            device = "SR830"
            self.core = SR830(f"{name}_core", address, *args, **kwargs)
            self.sinc = self.core.sync_filter
            self.tc = self.core.time_constant
            self.order = self.filter_slope
            
    def r_val(self, demods=0):
        return abs(self.core.demods[demods].sample()["x"][0] + 1j*self.core.demods[demods].sample()["y"][0])
    
    def p_val(self, demods=0):
        return self.core.demods[demods].sample()["phase"][0]
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return self.core.__getattr__(name)