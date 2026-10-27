#New NLP practice stuff

import ollama

# Build context from the current sensor readings
context = f"""
You are an assistant for a machine predictive maintenance system.

Equipment: {equipment}
Location: {location}

Current sensor readings:
- Temperature: {temperature} degrees Celsius
- Pressure: {pressure} bar
- Vibration: {vibration} mm/s
- Humidity: {humidity}%

Random Forest prediction: {machine_status}

Configured warning checks:
- High temperature: {temperature > 90}
- High pressure: {pressure > 60}
- High vibration: {vibration > 7}
- High humidity: {humidity > 70}

Explain the readings and possible maintenance concerns.
Do not claim that a warning proves the cause of a failure.
Do not invent sensor readings, inspection results, or confirmed causes.
If the evidence is insufficient to identify the cause, say so.
"""

print("\nMachine Maintenance Chatbot")
print("Ask about the machine, or type 'exit' to quit.")

while True:
    question = input("\nYou: ").strip()

    if question.lower() == "exit":
        break

    if not question:
        continue

    response = ollama.chat(
    model="qwen2.5:3b",
    messages=[
      {
        "role": "system",
        "content": (
          "Answer clearly and concisely using the "
          "provided machine data. Distinguish predictions "
          "from confirmed causes."
        )
      },
      {
        "role": "user",
        "content": f"""
Machine information:
{context}

Question: {question}
"""
      }
    ],
    options={"temperature": 0.2}
  )

print("\nAssistant:", response["message"]["content"])