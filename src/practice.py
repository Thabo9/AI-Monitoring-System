import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('../Dataset/ai4i2020.csv')


df['Type'] = df['Type'].map({'L': 0, 'M': 1, 'H': 2})

features = ['Type',
'Air temperature [K]',
'Process temperature [K]',
'Rotational speed [rpm]',
'Torque [Nm]',
'Tool wear [min]']

X = df[features]

X.hist(bins=42, figsize=(12, 8))
plt.show()

y = df['Machine failure']

X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=.20, random_state= 42, stratify=y
) 

model = RandomForestClassifier(n_estimators=100,
                               random_state = 42)

model = model.fit(X_train, y_train)

prediction = model.predict(X_test)

cm =confusion_matrix(y_test, prediction)

print(cm)
print(classification_report(y_test, prediction))