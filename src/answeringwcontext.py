from transformers import pipeline

# Load the pre-trained model and tokenizer
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def answer_question(context, question):
    # Use the pipeline to get the answer
    result = qa_pipeline(question=question, context=context)
    return result['answer']

def main():
    print("Welcome to the AI Question Answering System!")
    print("Please provide a context, and then ask questions about it.")
    print("Type 'quit' at any time to exit.")

    context = input("\nPlease enter the context: ")

    while True:
        question = input("\nYour question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            print("Thank you for using the AI Question Answering System. Goodbye!")
            break

        answer = answer_question(context, question)
        print("\nAnswer:", answer)

if __name__ == "__main__":
    main()