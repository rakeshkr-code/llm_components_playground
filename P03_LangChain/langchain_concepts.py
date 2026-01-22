from langchain_ollama import ChatOllama
from pprint import pprint

# Use local Llama (FREE, no API key!) [web:145]
llm = ChatOllama(
    model="llama3.2:1b",  # Smallest, fastest
    temperature=0.7
)

# Or use DeepSeek (better for code) [web:153]
llm_code = ChatOllama(
    model="deepseek-r1:1.5b"
)

# Chat
# response = llm.invoke("What is machine learning?")
response = llm.invoke("Capital of India?")

print(type(response), end="\n\n")
pprint(dict(response))
print()
print(response.content)
