from transformers import GPT2LMHeadModel, GPT2Tokenizer

def setup_model():
    model_name = "gpt2-medium"  # We'll use a larger version of GPT-2 for better performance
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return tokenizer, model

def answer_question(tokenizer, model, question):
    prompt = f"Q: {question}\nA:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate a response
    output = model.generate(input_ids, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)
    
    # Decode and clean up the response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = response.split("A:")[1].strip()
    
    return answer

def main():
    print("Welcome to the AI Question Answering System!")
    print("Ask any question, and I'll do my best to provide an accurate answer.")
    print("Type 'quit' to exit.")

    tokenizer, model = setup_model()

    while True:
        user_input = input("\nYour question: ")
        if user_input.lower() == 'quit':
            print("Thank you for using the AI Question Answering System. Goodbye!")
            break

        try:
            answer = answer_question(tokenizer, model, user_input)
            print("\nAI Response:")
            print(answer)
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try again.")

if __name__ == "__main__":
    main()