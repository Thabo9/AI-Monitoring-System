import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report,ConfusionMatrixDisplay,confusion_matrix
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('Dataset/equipment_anomaly_data.csv')

df['equipment'] = df['equipment'].map({'Turbine': 0, 'Compressor': 1, 'Pump': 2})
df['location'] = df['location'].map({'Atlanta': 0, 'Chicago': 1, 'San Francisco': 2, 'New York': 3, 'Houston': 4})

features =['equipment',
'pressure',
'vibration',
'humidity',
'location']

X = df[features]

#X.hist(bins=42, figsize=(12, 8))
#plt.show()

y = df['faulty']

X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=.20, random_state= 42, stratify=y
) 

model = RandomForestClassifier()

model = model.fit(X_train, y_train)

prediction = model.predict(X_test)

accuracy = accuracy_score(
  y_test,
  prediction  
)
cm = confusion_matrix(y_test, prediction)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title('Confusion Matrix')
plt.show()
classification_rep = classification_report(y_test, prediction)

#print(accuracy)
#print(classification_rep)