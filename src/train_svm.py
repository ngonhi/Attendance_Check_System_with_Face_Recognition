from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.svm import SVC 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import numpy as np
import argparse
import pickle
import json
import joblib

# Construct the argumet parser and parse the argument
ap = argparse.ArgumentParser()

ap.add_argument("--embeddings", default="outputs/embeddings.pickle",
                help="path to serialized db of facial embeddings")
ap.add_argument("--test_embeddings", default="outputs/test_embeddings.pickle",
                help="path to serialized db of facial embeddings")
ap.add_argument("--model", default="outputs/svm_model.pkl",
                help="path to output trained model")
args = vars(ap.parse_args())

# Load the face embeddings
data = pickle.loads(open(args["embeddings"], "rb").read())
labels = np.array(data['names'])
embeddings = np.array(data["embeddings"])

param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}  

model = SVC()
grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', refit = True, verbose = 3)
print('start fitting')
grid.fit(embeddings, labels)
print(grid.best_score_)
print(grid.best_params_)
joblib.dump(grid.best_estimator_, args['model'])
print('Finished fitting')

# Test
# Load the face embeddings
test_data = pickle.loads(open(args["test_embeddings"], "rb").read())
test_labels = test_data['names']
test_embeddings = np.array(test_data["embeddings"])

#Load model
model = joblib.load(args['model'])
#Evaluate
labels_pred = model.predict(test_embeddings)
print(classification_report(labels_pred, test_labels))