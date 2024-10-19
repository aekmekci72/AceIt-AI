from rake_nltk import Rake
import re

def extract_keyphrases(text, num_phrases=5):
    # Initialize RAKE with parameters
    r = Rake()

    # Extract keywords from the text
    r.extract_keywords_from_text(text)

    # Get the top keywords by score
    keyphrases = r.get_ranked_phrases()

    # Filter for unique and meaningful keyphrases
    unique_keyphrases = []
    for phrase in keyphrases:
        # Clean up the phrase (remove special characters, extra spaces)
        cleaned_phrase = re.sub(r'\W+', ' ', phrase.lower()).strip()
        
        # Check if the cleaned phrase is not empty and not already in unique_keyphrases
        if cleaned_phrase and cleaned_phrase not in unique_keyphrases:
            unique_keyphrases.append(cleaned_phrase)

        # Stop if we have enough keyphrases
        if len(unique_keyphrases) >= num_phrases:
            break

    return unique_keyphrases[:num_phrases]

# Example text
text = """
Natural language processing (NLP) is a subfield of artificial intelligence concerned with the interaction 
between computers and humans in natural language. TextRank is an extractive summarization technique 
that finds the most informative sentences in a text. It works by building a graph where sentences are nodes 
and edges denote the similarity between sentences.
"""

# Extract keyphrases using refined approach
keyphrases = extract_keyphrases(text)

# Print extracted keyphrases
print("Extracted Keyphrases:")
for phrase in keyphrases:
    print(f"- {phrase.capitalize()}")  # Capitalize first letter for better readability
    # You can add more elaboration or description for each keyphrase here if needed
    print()
