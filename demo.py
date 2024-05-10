from gradio import TabbedInterface

from demo.delightful_hifi import interfaceDelightfulHifi44100
from demo.fastpitch_hifi import interfaceFastpichHifi

TabbedInterface(
    [interfaceDelightfulHifi44100, interfaceFastpichHifi],
    ["DelightfulHifi_44100", "FastpichHifi"],
).launch()
