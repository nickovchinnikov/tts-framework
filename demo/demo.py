from gradio import TabbedInterface

from .delightful_hifi import interfaceDelightfulHifi44100
from .fastpitch_hifi import interfaceFastpichHifi

TabbedInterface(
    [interfaceDelightfulHifi44100, interfaceFastpichHifi],
    ["Saxophone", "Flute"],
).launch()
