import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet
import random

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def get_antonym(word):
    antonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.antonyms():
                antonyms.append(lemma.antonyms()[0].name())
    return antonyms[0] if antonyms else None

def generate_tf_questions(text, num_questions=10):
    sentences = sent_tokenize(text)
    questions = []

    for sentence in sentences:
        # Generate a true question
        questions.append((f"True or False: {sentence}", True))

        words = word_tokenize(sentence)
        tagged = pos_tag(words)

        # False question strategies
        strategies = [
            lambda: negate_verb(sentence, tagged),
            lambda: replace_with_antonym(sentence, tagged),
            lambda: change_quantity(sentence, tagged),
        ]

        false_question = None
        while not false_question and strategies:
            strategy = random.choice(strategies)
            false_question = strategy()
            strategies.remove(strategy)

        if false_question:
            questions.append((f"True or False: {false_question}", False))

    random.shuffle(questions)
    return questions[:num_questions]

def negate_verb(sentence, tagged):
    for i, (word, tag) in enumerate(tagged):
        if tag.startswith('VB'):
            if word.lower() in ['is', 'are', 'was', 'were']:
                return sentence[:sentence.index(word)] + word + " not" + sentence[sentence.index(word)+len(word):]
            elif word.lower() == 'have':
                return sentence[:sentence.index(word)] + "do not have" + sentence[sentence.index(word)+len(word):]
            else:
                return sentence[:sentence.index(word)] + "does not " + word + sentence[sentence.index(word)+len(word):]
    return None

def replace_with_antonym(sentence, tagged):
    for word, tag in tagged:
        if tag.startswith('JJ') or tag.startswith('RB'):
            antonym = get_antonym(word)
            if antonym:
                return sentence.replace(word, antonym)
    return None

def change_quantity(sentence, tagged):
    quantity_words = {'all': 'some', 'every': 'some', 'always': 'sometimes', 'never': 'sometimes'}
    for word, _ in tagged:
        if word.lower() in quantity_words:
            return sentence.replace(word, quantity_words[word.lower()])
    return None

# Example usage
notes = """
Recursion is a method of solving a problem where the solution depends on solutions to smaller instances of the same problem. 
A recursive function calls itself during its execution. The process of recursion can be thought of as a loop that repeats until a base condition is met.
"""

questions = generate_tf_questions(notes)

for i, (question, answer) in enumerate(questions, 1):
    print(f"{i}. {question}")
    print(f"Answer: {'True' if answer else 'False'}\n")