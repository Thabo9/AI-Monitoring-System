
from ast import While

from transformers import pipeline
from pypdf import PdfReader


reader = PdfReader("AI with Python (2).pdf")
context = ""
for page in reader.pages:
    context += page.extract_text() +"\n"



qa_model = pipeline("question-answering", model = "deepset/minilm-uncased-squad2")

#while True:
question = input("Ask a question (or type 'exit' to quit): ")
   #if question.lower() == 'exit':
     #break
  # else:
answer = qa_model(question=question, context=context)
print(answer)
     










   






              

     

