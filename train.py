# Import libraries
from azureml.core import Run, Model
import argparse
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# Get parameters
parser = argparse.ArgumentParser()
#Input dataset 
parser.add_argument("--training-data", type=str, dest='training_data', help='training data')
# Hyperparameters
parser.add_argument('--learning_rate', type=float, dest='learning_rate', default=0.1, help='learning rate')
parser.add_argument('--n_estimators', type=int, dest='n_estimators', default=100, help='number of estimators')
# Add arguments to args collection
args = parser.parse_args()
training_data = args.training_data

# Get the experiment run context
run = Run.get_context()

# Log Hyperparameter values
run.log('learning_rate',  np.float(args.learning_rate))
run.log('n_estimators',  np.int(args.n_estimators))

# load the prepared data file in the training folder
print("Loading Data...")
file_path = os.path.join(training_data,'df.csv')
df = pd.read_csv(file_path)
x = df.drop(['income'], axis = 1).values
y = df['income'].values


# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=0)

# Train a Gradient Boosting classification model with the specified hyperparameters
print('Training a classification model')
model = GradientBoostingClassifier(learning_rate=args.learning_rate,
                                   n_estimators=args.n_estimators).fit(X_train, y_train)

# calculate accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
print('Accuracy:', acc)
run.log('Accuracy', np.float(acc))

# calculate AUC
y_scores = model.predict_proba(X_test)
auc = roc_auc_score(y_test,y_scores[:,1])
print('AUC: ' + str(auc))
run.log('AUC', np.float(auc))

# plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])
fig = plt.figure(figsize=(6, 4))
# Plot the diagonal 50% line
plt.plot([0, 1], [0, 1], 'k--')
# Plot the FPR and TPR achieved by our model
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
run.log_image(name = "ROC", plot = fig)
plt.show()
run.log_image(name='ROC curve', plot=fig)

# Save the model in the run outputs
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/adult_income_model.pkl')

run.complete()
