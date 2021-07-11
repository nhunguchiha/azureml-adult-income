# Import libraries
import os
import argparse
import pandas as pd
from azureml.core import Run
from sklearn.preprocessing import LabelEncoder

# Get parameters
parser = argparse.ArgumentParser()
parser.add_argument("--input-data", type=str, dest='raw_dataset_id', help='raw dataset')
parser.add_argument('--prepped-data', type=str, dest='prepped_data', default='prepped_data', help='Folder for results')
args = parser.parse_args()
save_folder = args.prepped_data

# Get the experiment run context
run = Run.get_context()

# load the data (passed as an input dataset)
print("Loading Data...")
df = run.input_datasets['raw_data'].to_pandas_dataframe()

#Drop columns that have many null value and __index_level_0__ column
df = df.drop(['workclass','occupation','native-country', '__index_level_0__'], axis = 1)

category_feature = ['education','marital-status','relationship','race','sex']



# one hot coding the categorical columns
df = pd.get_dummies(df, columns = category_feature , drop_first = True)

# encoding the target columns
target = ['income']
le = LabelEncoder()
df[target] = le.fit_transform(df[target])
run.log_list('label', le.classes_)
# Save the prepped data
print("Saving Data...")
os.makedirs(save_folder, exist_ok=True)
save_path = os.path.join(save_folder,'df.csv')
df.to_csv(save_path, index=False, header=True)
# End the run
run.complete()
