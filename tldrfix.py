from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

def generate_single_sentence_summary(text, max_length=50, min_length=30):
    # Load pre-trained T5 model and tokenizer
    model_name = "t5-small"  # You can use "t5-base" for better quality but slower processing
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Prepare the text input
    preprocess_text = text.strip().replace("\n", "")
    t5_prepared_Text = "summarize: " + preprocess_text

    # Tokenize the input text
    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt", max_length=512, truncation=True)

    # Generate summary
    summary_ids = model.generate(
        tokenized_text,
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Ensure the summary is a single sentence
    summary_sentences = summary.split('. ')
    single_sentence_summary = summary_sentences[0] if summary_sentences else summary

    return single_sentence_summary

# Example usage
text_to_summarize = """
Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Colloquially, the term "artificial intelligence" is often used to describe machines (or computers) that mimic "cognitive" functions that humans associate with the human mind, such as "learning" and "problem-solving". The field of AI research was founded on the claim that human intelligence "can be so precisely described that a machine can be made to simulate it". Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.
"""

summary = generate_single_sentence_summary(text_to_summarize)
print("Summary:", summary)