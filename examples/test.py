from qcodes.instrument import Instrument


class ShellInstrument(Instrument):
    def __init__(
        self,
        name: str,
        parameters: dict,
        **kwargs,
    ) -> None:

        super().__init__(name, **kwargs)

        for key, parameter in parameters.items():
            self.add_parameter(
                key,
                **parameter,
            )

    def get_idn(self) -> dict[str, str | None]:
        return {
            "vendor": "SQUAD Lab",
            "model": "Shell Instrument",
            "serial": None,
            "firmware": None,
        }


Instrument.close_all()

dummy_lockin = ShellInstrument(
    "dummy_lockin",
    parameters={},
)

print(dummy_lockin.get_idn())
print(dummy_lockin.snapshot())
