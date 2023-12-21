# %%
import torch
path ="./epoch=4677-step=410361.ckpt"

# Load checkpoint
ckpt = torch.load(path, map_location=torch.device('cpu'))
ckpt
                  
# %%
ckpt["optimizer_states"][0]['param_groups'][0]["lr"]

# %%
ckpt["optimizer_states"][0]['param_groups'][0]["initial_lr"]

# %%
path2 = "epoch=121-step=7808.ckpt"
# Load checkpoint 2
ckpt2 = torch.load(path2, map_location=torch.device('cpu'))

# %%
latest_lr = ckpt2["optimizer_states"][0]['param_groups'][0]["lr"]
latest_lr

# %%
ckpt2["optimizer_states"][0]['param_groups'][0]["lr"] = latest_lr * 2000

ckpt2["optimizer_states"][0]['param_groups'][0]["lr"]

# %%
ckpt2["optimizer_states"][0]['param_groups'][0]["initial_lr"]

# %%
path3 = "./epoch=121-step=7808_fixedlr.ckpt"

# Save the new checkpoint
torch.save(ckpt2, path3)

# %%