import wikipediaapi
from collections import Counter
import time
import requests

def suggest_learning_path(explorations, max_suggestions=10, max_retries=3):
    wiki_wiki = wikipediaapi.Wikipedia(
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI,
        user_agent='studybuddy-app/1.0'
    )

    concept_counter = Counter()
    learning_keywords = ['concept', 'theory', 'principle', 'method', 'technique', 'algorithm', 'framework', 'model', 'paradigm', 'approach']

    for exploration in explorations:
        for attempt in range(max_retries):
            try:
                page = wiki_wiki.page(exploration)
                if page.exists():
                    # Process sections
                    for section in page.sections:
                        if any(keyword in section.title.lower() for keyword in learning_keywords):
                            concept_counter[section.title] += 2
                    
                    # Process links
                    for link in page.links.values():
                        if any(keyword in link.title.lower() for keyword in learning_keywords):
                            concept_counter[link.title] += 1
                break  # If successful, break the retry loop
            except (requests.exceptions.RequestException, wikipediaapi.WikipediaException) as e:
                if attempt < max_retries - 1:
                    print(f"Error occurred: {e}. Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print(f"Failed to process '{exploration}' after {max_retries} attempts.")

    # Ensure we have at least max_suggestions
    while len(concept_counter) < max_suggestions:
        concept_counter[f"Explore more about {explorations[len(concept_counter) % len(explorations)]}"] += 1

    # Get top suggestions
    suggested_concepts = [concept for concept, _ in concept_counter.most_common(max_suggestions)]

    return suggested_concepts

# Example usage
explorations = [
    "Spanish Inquisition",
    "Irish Potato Famine",
    "Civil War"
]

max_suggestions = 10
try:
    suggested_concepts = suggest_learning_path(explorations, max_suggestions=max_suggestions)
    print(f"Suggested Learning Path (top {max_suggestions} concepts):")
    for i, concept in enumerate(suggested_concepts, 1):
        print(f"{i}. {concept}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")