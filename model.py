import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import os

# random seed
seed = 42

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Read original dataset
iris_path = os.path.join(script_dir, "iris.csv")
iris_df = pd.read_csv(iris_path)
iris_df = iris_df.sample(frac=1, random_state=seed)

# selecting features and target variable
X = iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = iris_df['species']

# split data into train and test sets
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed, stratify=y)

# create an instance of KNN classifier
clf = KNeighborsClassifier(n_neighbors=5)

# train the classifier on the training data
clf.fit(X_train, y_train)

# evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy}')

# save the model to disk
model_path = os.path.join(script_dir, "knn_model.sav")
joblib.dump(clf, model_path)
print(f'Model saved to {model_path}')