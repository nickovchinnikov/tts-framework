# %%
import matplotlib.pyplot as plt
import pandas as pd

# Load the data
data = pd.read_json("data_result.json")

# %%
data.iloc[:, 1:]

# %%
# Create a DataFrame with the second and third values as x and y
df = pd.DataFrame(data.iloc[:, 1:])

# Plot the data
df.plot()

# Show the plot
plt.show()

# %%
import glob

import pandas as pd

# Get a list of all JSON files in the directory
files = glob.glob("./versions/*.json")
files

# %%

# %%
