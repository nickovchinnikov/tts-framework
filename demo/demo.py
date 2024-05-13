from gradio import TabbedInterface

from .delightful_hifi import interfaceDelightfulHifi44100
from .delightful_univnet import interfaceDelightfulUnuvnet22050
from .fastpitch_hifi import interfaceFastpichHifi

TabbedInterface(
    [
        interfaceDelightfulUnuvnet22050,
        interfaceDelightfulHifi44100,
        interfaceFastpichHifi,
    ],
    ["DelightfulUnuvnet", "DelightfulHifi", "FastpichHifi"],
).launch(server_port=6006)
