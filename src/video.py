import re
import asyncio
import os
import requests
import spacy
from gtts import gTTS
from moviepy.editor import ImageSequenceClip, AudioFileClip, concatenate_videoclips, concatenate_audioclips
from PIL import Image
from duckduckgo_search import DDGS
from sydney import SydneyClient
from dotenv import load_dotenv

load_dotenv()

bing_cookies_key = os.getenv('BING_COOKIES')

if bing_cookies_key is None:
    print("Error: BING_COOKIES environment variable is not set.")
    exit(1)

os.environ["BING_COOKIES"] = bing_cookies_key

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

def generate_audio(text):
    tts = gTTS(text=text, lang='en')
    return tts

def search_images(term, max_images=30):
    print(f"Searching for '{term}'")
    ddgs = DDGS()
    return [result['image'] for result in ddgs.images(keywords=term, max_results=max_images)]


def download_image(url, folder):
    filename = os.path.join(folder, os.path.basename(url))
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            f.write(response.content)
        # Verify the image
        with Image.open(filename) as img:
            img.verify()
        return filename
    except Exception as e:
        print(f"Error downloading or verifying image {url}: {e}")
        return None

def resize_images(image_paths, output_folder, target_size=(1280, 720)):
    resized_paths = []
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"Image file {image_path} does not exist. Skipping.")
            continue
        try:
            image = Image.open(image_path)
            image = image.resize(target_size)
            resized_path = os.path.join(output_folder, os.path.basename(image_path))
            image.save(resized_path)
            resized_paths.append(resized_path)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
    return resized_paths


def extract_concepts_and_summaries(text):
    concepts = {}
    current_concept = None
    current_search_phrases = []
    current_summary = []

    lines = text.strip().split('\n')
    for line in lines:
        line = line.strip()
        print(f"Processing line: {line}")
        if re.match(r'^\d+\.\s\*\*.*\*\*:$', line):  # Match lines like "1. **Creating Functions**:"
            if current_concept:
                concepts[current_concept] = {
                    "search_phrases": current_search_phrases,
                    "summary": ' '.join(current_summary)
                }
            current_concept = line.split('**')[1].strip()
            current_search_phrases = []
            current_summary = []
            print(f"New concept detected: {current_concept}")
        elif line.startswith("- Search phrases:"):
            phrases = re.findall(r'"(.*?)"', line)
            current_search_phrases.extend(phrases)
            print(f"Search phrases found: {current_search_phrases}")
        elif line.startswith("- Summary:"):
            current_summary.append(line.split(":", 1)[1].strip())
            print(f"Summary found: {current_summary}")
        else:
            if current_concept:
                current_summary.append(line.strip())
    
    if current_concept:
        concepts[current_concept] = {
            "search_phrases": current_search_phrases,
            "summary": ' '.join(current_summary)
        }
    
    return concepts

async def main(response) -> None:
    async with SydneyClient() as sydney:
        # Extract concepts and summaries from the response
        concepts = extract_concepts_and_summaries(response)

        # Debugging: Print the extracted concepts and summaries
        print("Extracted Concepts and Summaries:")
        for concept, data in concepts.items():
            print(f"Concept: {concept}")
            print(f"Search Phrases: {data['search_phrases']}")
            print(f"Summary: {data['summary']}")
            print("-----")

        # Exit if no concepts were extracted
        if not concepts:
            print("No concepts were extracted. Exiting.")
            return

        image_paths = []

        for concept, data in concepts.items():
            concept_images = []
            for phrase in data["search_phrases"]:
                images = search_images(phrase, max_images=1)
                if images:
                    concept_images.append(images[0])
            concepts[concept]["images"] = concept_images

            print(f"Downloading images for concept: {concept}")
            concept_folder = os.path.join(os.getcwd(), "images", concept)
            os.makedirs(concept_folder, exist_ok=True)
            for i, image_url in enumerate(data['images']):
                image_path = download_image(image_url, concept_folder)
                if image_path:
                    image_paths.append(image_path)
                    print(f"Downloaded image {i + 1}: {image_path}")

        image_paths = resize_images(image_paths, "resized_images")

        concept_audio_paths = []
        concept_video_clips = []

        for concept, data in concepts.items():
            summary = data["summary"]
            images = data["images"]
            image_durations = []
            image_clips = []

            tts = generate_audio(summary)
            audio_path = f"audio_{concept}.mp3"
            tts.save(audio_path)
            concept_audio_paths.append(audio_path)

            audio_clip = AudioFileClip(audio_path)
            audio_duration = audio_clip.duration

            duration_per_image = audio_duration / len(images)
            for i, image in enumerate(images):
                image_path = os.path.join("resized_images", os.path.basename(image))
                if os.path.exists(image_path):
                    image_clips.append((image_path, duration_per_image))

            for image_path, duration in image_clips:
                img_clip = ImageSequenceClip([image_path], durations=[duration])
                concept_video_clips.append(img_clip)

        final_video_clip = concatenate_videoclips(concept_video_clips)
        final_audio_clip = concatenate_audioclips([AudioFileClip(p) for p in concept_audio_paths])
        final_video_clip = final_video_clip.set_audio(final_audio_clip)

        output_video_path = "output_video.mp4"
        final_video_clip.write_videofile(output_video_path, fps=24)
newinp = '''[1]: https://bing.com/search?q=What+is+equilibrium%3f 
""[2]: https://www.merriam-webster.com/dictionary/equilibrium 
""[3]: https://www.britannica.com/science/equilibrium-physics 
""[4]: https://www.dictionary.com/browse/equilibrium 
""[5]: http://www.oxforddictionaries.com/us/ 
""**Equilibrium** refers to a state of balance between opposing forces or actions. Let's break it down:
1. **Intellectual or Emotional Balance**: Equilibrium can describe a state of intellectual or emotional poise, where someone maintains stability and composure[^1^][2]. For instance, when trying to recover their equilibrium after a challenging situation.
2. **Adjustment Between Opposing Influences**: It also represents a state of adjustment between opposing or divergent influences or elements. Imagine finding equilibrium between commercial development and the conservation of natural treasures[^1^][2].
3. **Physical Balance**: In physics, equilibrium refers to a system where neither its motion nor its internal energy state tends to change over time. This can be either static (as in a balanced body acted upon by forces with a net resultant of zero) or dynamic (as in a reversible chemical reaction with equal rates in both directions)[^2^][3].
4. **Symbolism**: The word "equilibrium" has roots in the Latin word "libra," meaning "weight" or "balance." As a constellation, zodiac symbol, and astrological sign, Libra is often depicted as a set of balance scales, symbolizing fairness, equality, and justice[^1^][2].Remember, equilibrium appears in various contexts, including biology, chemistry, physics, and economics, but the common thread is always the balance of competing influences[^1^][2]. 
If you have any more questions, feel free to ask! 😊'''
oldinp = '''Certainly! Let's break down each concept and identify key search phrases for relevant static images:

        1. **Creating Functions**:
            - Search phrases: "defining functions in Python," "user-defined functions," "function parameters," "return statement."
            - Summary: Functions in programming allow you to encapsulate reusable code. They take input (parameters), perform specific tasks, and optionally return a result. The syntax for defining a function in Python is straightforward: `def function_name(parameters):`. The body of the function contains the code to be executed when the function is called[^1^][1].

        2. **Decomposition**:
            - Search phrases: "Algorithmic decomposition," "structured analysis," "object-oriented decomposition," "functional decomposition."
            - Summary: Decomposition involves breaking down complex problems or systems into smaller, manageable parts. Different types of decomposition exist, including algorithmic decomposition (structured steps), structured analysis (system functions and data entities), and object-oriented decomposition (classes or objects). Functional decomposition replaces a system's functional model with subsystem models[^2^][5].

        3. **While Loops**:
            - Search phrases: "While loop in programming," "entry-controlled loops," "condition-based repetition."
            - Summary: While loops execute a block of code repeatedly as long as a specified condition remains true. They evaluate the condition before each iteration, execute the code block if the condition is true, and terminate when the condition becomes false. Useful for uncertain or dynamically changing situations[^3^][9].

        4. **For Loops**:
            - Search phrases: "For loop in programming," "iterating over a sequence," "fixed number of iterations."
            - Summary: For loops execute a set of statements repetitively based on a specified condition. They are commonly used when you know how many times you want to execute a block of code. The syntax includes initialization, condition, and increment (or decrement) components. Useful for iterating over sequences or performing a fixed number of tasks[^4^][13].

        Feel free to use these search phrases to find relevant static images that enhance student comprehension!
'''
if __name__ == "__main__":
    asyncio.run(main(newinp))