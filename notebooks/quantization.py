# %%
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


from neural_compressor.config import (
    AccuracyCriterion,
    PostTrainingQuantConfig,
    TuningCriterion,
)
from neural_compressor.quantization import fit

from training.modules import AcousticModule

# %%
checkpoint = "checkpoints/epoch=5371-step=575533.ckpt"
module = AcousticModule.load_from_checkpoint(checkpoint)
module.eval()


# %%
accuracy_criterion = AccuracyCriterion(tolerable_loss=0.01)
tuning_criterion = TuningCriterion(max_trials=600)
conf = PostTrainingQuantConfig(
    approach="static", backend="default", tuning_criterion=tuning_criterion, accuracy_criterion=accuracy_criterion,
)


# %%
# q_model = fit(
#     model=module.acoustic_model,
#     conf=conf,
#     calib_dataloader=module.train_dataloader(),
#     eval_func=eval_func
# )

# q_model.save("./saved_model/")

# %%

# %%

# %%
