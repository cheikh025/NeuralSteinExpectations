import pandas as pd

import os

### Merge all the .csv files in the experiments folder
# Path to the experiments folder
experiments_folder = 'experiments/GMM_results/'
# List all the files in the experiments folder
files = os.listdir(experiments_folder)
# Filter out the .csv files
files = [file for file in files if (file.endswith('.csv'))]

# Read all the .csv files and concatenate them along the rows
df = pd.concat([pd.read_csv(os.path.join(experiments_folder, file)).drop(columns=["Unnamed: 0"]) for file in files], ignore_index=True)

# Save the merged dataframe to a new .csv file
df.to_csv(experiments_folder + 'merged_GMM_results.csv', index=False)
