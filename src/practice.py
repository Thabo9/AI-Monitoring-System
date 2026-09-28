import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

class AIModel:
    def __init__(self):
        
        df = pd.read_csv('../Dataset/equipment_anomaly_data.csv')

        
        df['equipment'] = df['equipment'].map({'Turbine': 0, 'Compressor': 1, 'Pump': 2})
        df['location'] = df['location'].map({'Atlanta': 0, 'Chicago': 1, 'San Francisco': 2, 'New York': 3, 'Houston': 4})

        features = ['temperature', 'pressure', 'vibration', 'humidity', 'equipment', 'location']
        X = df[features]
        y = df['faulty']

        #Training and splitting of data, used stratify to preserve propotion 
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        ) 
        #Training the data using the Random Forest Classifier
        self.model_rand = RandomForestClassifier()
        self.model_rand.fit(X_train, y_train)

        #Prediction of output(0 or 1) and model accuracy
        prediction = self.model_rand.predict(X_test)
        self.accuracy = accuracy_score(y_test, prediction)

    #DEfined two methods to use in nlp-config.py for prediction
    def predict(self, input_data):
        return self.model_rand.predict(input_data)

    def predict_proba(self, input_data):
        return self.model_rand.predict_proba(input_data)
