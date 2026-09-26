# %%

from qcdrivers.squad import ShellInstrument

from qanary.measure import Station

# %%

dummy = ShellInstrument("dummy", {})
dummy2 = ShellInstrument("dummy2", {})

# %%

st = Station("test")
st.instruments = [dummy, dummy2]

# %%

v1 = None
v2 = None
v3 = None

dummy.add_parameter(
    "ch1",
    label="Channel 1",
    unit="V",
    set_cmd=lambda value: globals().update(v1=value),
    get_cmd=lambda: v1,
)

dummy.add_parameter(
    "ch2",
    label="Channel 2",
    unit="V",
    set_cmd=lambda value: globals().update(v2=value),
    get_cmd=lambda: v2,
)

dummy2.add_parameter(
    "ch3",
    label="Channel 3",
    unit="V",
    set_cmd=lambda value: globals().update(v3=value),
    get_cmd=lambda: v3,
)


# %%

multi_param = st.add_parameter(
    param=[dummy.ch1, dummy.ch2],
    name="custom_multi_param_1",
    label="Custom Multi Parameter 1",
)

# %%

dummy_ch1 = st.add_parameter("ch1", "Channel 1", dummy.ch1, param_type="testing")

# %%

# test outputs
multi_param.set(5.0)
print(f"ch1: {dummy.ch1()}, ch2: {dummy.ch2()}")
print(f"multi_param: {multi_param.get()}")

print(multi_param.name)
print(multi_param.label)

print(multi_param.source_parameters)
print(multi_param.param_type)

print(multi_param.snapshot())

# %%
