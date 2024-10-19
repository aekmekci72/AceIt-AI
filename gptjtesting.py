import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the GPT-J model and tokenizer
model_name = "EleutherAI/gpt-j-6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to generate a response from the model
def generate_response(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.7
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def chat():
    print("Chatbot: Hello! How can I help you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye!")
            break
        response = generate_response(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chat()
