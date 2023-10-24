from pytorch_lightning.core import LightningModule

from model.univnet import UnivNet


class AcousticModule(LightningModule):
    def __init__(self):
        super().__init__()
