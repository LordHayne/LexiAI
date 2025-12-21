from langchain_ollama import ChatOllama

client = ChatOllama(base_url="http://localhost:11434", model="gemma3:4b-it-qat")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Sag mir etwas Nettes."}
]

response = client.invoke(messages)
print("Antwort:")
print(response)
print("Inhalt:")
print(response.content)
