import joblib
import os

def predict(data):
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "knn_model.sav")
    
    # Load the model
    clf = joblib.load(model_path)
    
    # Make prediction
    return clf.predict(data)