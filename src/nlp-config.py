import nltk
"""nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')"""
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from nltk.tag import pos_tag
import practice
import pandas as pd

from transformers import pipeline
stop_words = set(stopwords.words('english'))
"""
# Tokenization
tokens = word_tokenize(input_text.lower())

cleaned_tokens = [token for token in tokens if token not in string.punctuation]

lemmatizer = WordNetLemmatizer()
# Lemmatization
# Lemmatization is good for linguistic accuracy.
for token in cleaned_tokens:
    lemmatized_token = lemmatizer.lemmatize(token, pos='v')
    #print("Lemmatized Token:", lemmatized_token)

pos_tags = pos_tag(cleaned_tokens)

nouns = [word for word, pos in pos_tags if pos.startswith('NN')]
verbs = [word for word, pos in pos_tags if pos.startswith('VB')] #Can add for adjectives later maybe.
print("Nouns:", nouns)
print("Verbs:", verbs)"""


qa_model = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')

#question = "What is the rotational speed and temperature of the equipment?"

#Object creation for the AIModel class
obj = practice.AIModel()

# var input_raw in same order as var feature_names
input_raw = [[200, 26, 1.8, 52.74, 1, 1]]
feature_names = ['temperature', 'pressure', 'vibration','humidity','equipment','location']
input_data = pd.DataFrame(input_raw, columns=feature_names)
prediction = obj.predict(input_data)
probabilities = obj.predict_proba(input_data)


print("Prediction:", prediction)
if prediction == 1:
    print(f"Probabilities: {probabilities[0][1]}%\nThe model predicts that the equipment is faulty.")
    machine_status = "The machine is faulty."
else:
    print(f"Probabilities: {probabilities[0][0]}\nThe model predicts that the equipment is not faulty.")
    machine_status = "The machine is not faulty."

#Mapping the numerical values back to their original names
equipment_names = {0: 'Turbine', 1: 'Compressor', 2: 'Pump'}
location_names = {0: 'Atlanta', 1: 'Chicago', 2: 'San Francisco', 3: 'New York', 4: 'Houston'}

#Input data to use for context gen
temperature = input_data['temperature'].values[0]
pressure = input_data['pressure'].values[0]
vibration = input_data['vibration'].values[0]
humidity = input_data['humidity'].values[0]
equipment = equipment_names[input_data['equipment'].values[0]]
location = location_names[input_data['location'].values[0]]

#Method providing reason the machine probably failed 
def failure_reason():
    reasons = ["Failure was caused by"]
    if temperature > 90:
        reasons.append(" Overheating")
    if vibration > 7:
        reasons.append(" high vibration, the machine fails rapidly if it exceeds 7mm/s ")
    if pressure > 60:
        reasons.append(" higher pressure bar")
    if humidity > 70: 
        reasons.append(" and higher humidity percentage")

    return reasons

#Feed the context info from the input data
context = f"""The equipment is a {equipment}. The temperature is {temperature} degrees Celsius.
              The pressure is {pressure} bar. The vibration is {vibration} mm/s.
              The humidity is {humidity}%. {machine_status}. The equipment is located in {location}.
              {failure_reason}"""

#A while loop for user to ask questions about the equipment and having an option to close the program
while True:
    question = input("Ask a question about the equipment (type 'exit' to quit): ")
    if question.lower() == 'exit':
        break
    print("#"*60)
    result = qa_model(question=question, context=context)
    print("Answer:", result['answer'])