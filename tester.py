from transformers import pipeline
from huggingface_hub import login

login(token="key")

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2")
pipe(messages)