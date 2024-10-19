from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Initialize the question generation pipeline
question_generator = pipeline("text2text-generation", model="valhalla/t5-small-qg-hl")

# Initialize the question-answering pipeline
qa_pipeline = pipeline('question-answering', model='valhalla/t5-small-qa-qg-hl')

def generate_qa_pairs(context, num_questions=5):
    # Generate factual question-answer pairs first
    factual_qa_pairs = generate_factual_qa_pairs(context, num_questions)
    
    # Additional types of questions to consider
    other_qa_pairs = []
    for question, answer in factual_qa_pairs:
        # Generate multiple choice, true/false, and free response questions if applicable
        if '?' not in question:  # Skip if it's already a factual question
            continue
        
        # Generate multiple choice question
        mcq = generate_multiple_choice_question(question, context)
        other_qa_pairs.append(mcq)
        
        # Generate true/false question
        tfq = generate_true_false_question(question, context)
        other_qa_pairs.append(tfq)
        
        # Generate free response question
        frq = generate_free_response_question(question, context)
        other_qa_pairs.append(frq)
    
    # Combine all generated QA pairs
    all_qa_pairs = factual_qa_pairs + other_qa_pairs
    
    return all_qa_pairs

def generate_factual_qa_pairs(context, num_questions):
    # Generate factual question-answer pairs
    factual_qa_pairs = []
    generated_questions = set()
    
    while len(factual_qa_pairs) < num_questions:
        question = generate_question(context)
        
        # Skip if the question is already generated to avoid duplicates
        if question in generated_questions:
            continue
        
        answer = generate_factual_answer(question, context)
        factual_qa_pairs.append((question, answer))
        generated_questions.add(question)
    
    return factual_qa_pairs

def generate_question(context):
    # Generate a question using the question generation pipeline
    question = question_generator(context, max_new_tokens=50, num_return_sequences=1)[0]['generated_text'].strip()
    return question

def generate_factual_answer(question, context):
    # Use a QA pipeline to get the answer to factual questions
    answer = qa_pipeline(question=question, context=context)['answer']
    return answer

def generate_multiple_choice_question(question, context):
    # Example implementation for generating a multiple choice question
    options = ["Option A", "Option B", "Option C", "Option D"]
    return (question, options)

def generate_free_response_question(question, context):
    # Example implementation for generating a free response question
    answer = qa_pipeline(question=question, context=context)['answer']
    return (question, answer)

def generate_true_false_question(question, context):
    # Example implementation for generating a true/false question
    answer = qa_pipeline(question=question, context=context)['answer']
    return (question, "True" if answer else "False")

# Example context (text explaining a concept)
concept_context = """
Deviations from the five conditions for Hardy-Weinberg equilibrium can alter allele frequencies in a population. The three main causes of evolutionary change are natural selection, genetic drift, and gene flow. Genetic drift refers to a change in the gene pool of a population due to chance, which is most pronounced in small populations.
"""

# Generate QA pairs
qa_pairs = generate_qa_pairs(concept_context, num_questions=5)

# Print generated QA pairs
print("Generated Question-Answer Pairs:")
for i, (question, answer) in enumerate(qa_pairs, start=1):
    if isinstance(answer, list):
        print(f"Question {i}: {question}")
        for idx, option in enumerate(answer, start=1):
            print(f"Option {idx}: {option}")
    else:
        print(f"Question {i}: {question}")
        print(f"Answer {i}: {answer}")
    print()
