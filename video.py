from huggingface_hub import login
from langchain_community.llms import HuggingFaceEndpoint
from gtts import gTTS
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, TextClip, CompositeVideoClip
import requests
from duckduckgo_search import DDGS
import re
import os
from PIL import Image, ImageDraw, ImageFont
from moviepy.video.fx import resize, crop
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


# Log in to Hugging Face with your API token
login(token="key")

# Initialize the endpoint
list_llm = [
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "google/gemma-7b-it",
    "google/gemma-2b-it",
    "HuggingFaceH4/zephyr-7b-beta",
    "HuggingFaceH4/zephyr-7b-gemma-v0.1",
    "meta-llama/Llama-2-7b-chat-hf",
    "microsoft/phi-2",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "mosaicml/mpt-7b-instruct",
    "tiiuae/falcon-7b-instruct",
    "google/flan-t5-xxl"
]
default_llm_index = 1

llm_model = list_llm[default_llm_index]
llm = HuggingFaceEndpoint(
    repo_id=llm_model,
    temperature=0.5,
    max_new_tokens=2048,
    top_k=20
)
def generate_video_script(text):
    prompt = f"""
    Create a detailed script for a video that explains the following concepts in an engaging and informative manner:
    
    {text}
    
    The script should include an introduction, explanation of key concepts, and examples. Prefix all of the spoken text in the script with Narrator: 
    Be sure to include a title at the beginning for the video, prefixed with Title:
    Whenever relevant, include a search term for a relevant graphic inside of square brackets preceding the corresponding chunk of text. Try to make the descriptions of the graphics fairly detailed so that the corresponding images that will be found are relevant to the content.

    """
    response = llm(prompt)
    return response
def get_title(video_script):
    lines = video_script.splitlines()

    for line in lines:
        if 'Title:' in line:
            narrator_text = line.split('Title:')[1].strip()
            return narrator_text
    return 'Explanation Video'    
def extract_narrator_script(video_script):
    narrator_lines = []
    lines = video_script.splitlines()

    for line in lines:
        if 'Title:' in line:
            narrator_text = line.split('Title:')[1].strip()
            narrator_lines.append(narrator_text)
        if 'Narrator:' in line:
            narrator_text = line.split('Narrator:')[1].strip()
            narrator_lines.append(narrator_text)
        if 'Narrator (voiceover):' in line:
            narrator_text = line.split('Narrator (voiceover):')[1].strip()
            narrator_lines.append(narrator_text)
        if 'Narrator (Voiceover):' in line:
            narrator_text = line.split('Narrator (Voiceover):')[1].strip()
            narrator_lines.append(narrator_text)
    return ' '.join(narrator_lines)

def generate_audio(text, output_file):
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save(output_file)
    print(f"Audio content written to {output_file}")

def extract_graphics_cues(video_script):
    # Extracts the content inside square brackets
    graphic_cues = []
    for line in video_script.splitlines():
        if '[' in line and ']' in line:
            cue = line[line.index('[') + 1:line.index(']')].strip()
            graphic_cues.append(cue)
    return graphic_cues
def extract_timed_sections(video_script):
    # Extract text sections and their timings
    sections = []
    for line in video_script.splitlines():
        match = re.match(r'\[Start: (\d+)\] \[(End: (\d+))\]', line)
        if match:
            start_time, end_time = int(match.group(1)), int(match.group(3))
            sections.append((start_time, end_time))
    return sections
def download_image(query, output_file, retries=10):
    with DDGS() as ddgs:
        for i in range(retries):
            results = ddgs.images(query)
            for result in results:
                image_url = result['image']
                try:
                    img_data = requests.get(image_url, timeout=10).content
                    with open(output_file, 'wb') as f:
                        f.write(img_data)
                except requests.exceptions.RequestException as e:
                    print(f"An error occurred: {e}")
                try:
                    # Attempt to open the image to ensure it's valid
                    img = Image.open(output_file)
                    img.verify()  # Verify that it is, in fact, an image
                    print(f"Image downloaded and verified: {output_file}")
                    return
                except (IOError, SyntaxError) as e:
                    print(f"Image {output_file} is corrupted or cannot be opened, trying another image. Error: {e}")
                    continue
            
            print(f"Retrying download for query '{query}' ({i+1}/{retries})")
        
        # If all retries fail
        raise ValueError(f"Could not find a valid image for query '{query}' after {retries} attempts.")


def calculate_wpm(narrator_script, audio_duration):
    total_words = len(narrator_script.split())
    wpm = total_words / (audio_duration / 60)  # Convert duration to minutes
    return wpm

def create_title_image(title_text, output_file):
    img = Image.new('RGB', (1280, 720), color='black')
    
    # Initialize ImageDraw
    d = ImageDraw.Draw(img)
    
    # Load a font
    try:
        # You can specify a TTF font file if you have one
        font = ImageFont.truetype("arial.ttf", 70)
    except IOError:
        # Use a default font if the specified font is not available
        font = ImageFont.load_default()
    
    # Wrap text
    max_width = img.width - 40  # Max width for text, with some padding
    lines = []
    words = title_text.split()
    current_line = ""

    for word in words:
        # Check the width of the current line with the next word
        test_line = current_line + word + " "
        text_width, _ = d.textsize(test_line, font=font)
        
        if text_width <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word + " "
    
    # Append the last line
    if current_line:
        lines.append(current_line)
    
    # Calculate total text height
    total_text_height = sum(d.textsize(line, font=font)[1] for line in lines)
    
    # Calculate starting y position
    text_y = (img.height - total_text_height) / 2
    
    # Draw each line
    for line in lines:
        text_width, text_height = d.textsize(line, font=font)
        text_x = (img.width - text_width) / 2
        d.text((text_x, text_y), line, font=font, fill=(255, 255, 255))
        text_y += text_height
    
    # Save the image
    img.save(output_file)
    print(f"Title image created: {output_file}")

def calculate_reading_time(text, wpm):
    words = len(text.split())
    return words / wpm * 60

def create_video(image_files, audio_file, script, output_file):
    audio = AudioFileClip(audio_file)
    audio_duration = audio.duration
    print(audio_duration)
    narrator_script = extract_narrator_script(script)
    wpm = calculate_wpm(narrator_script, audio_duration)
    
    clips = []
    current_time = 0

    # Extract graphics cues and corresponding text
    graphics_cues = extract_graphics_cues(script)
    script_lines = script.splitlines()
    
    # Title image and its duration
    title_text = get_title(script)
    title_duration = calculate_reading_time(title_text, wpm)

    # Add the title image as the first clip
    title_clip = ImageClip(image_files[0]).set_duration(title_duration).resize(height=720)
    title_clip = title_clip.set_audio(audio.subclip(0, title_duration))
    clips.append(title_clip)
    
    current_time += title_duration
    
    # Now process each graphic cue and corresponding text
    for index in range(len(graphics_cues)):
        # Find the script text after the current graphic cue until the next one
        text_after_cue = ""
        start_collecting = False
        for line in script_lines:
            if graphics_cues[index] in line:
                start_collecting = True
                continue
            if start_collecting:
                if index < len(graphics_cues) - 1 and graphics_cues[index + 1] in line:
                    break
                text_after_cue += line + " "
        
        # Calculate the reading time for the text after the current graphic cue using WPM
        reading_time = calculate_reading_time(text_after_cue, wpm)
        
        # Set the duration for the current image
        clip_start = current_time
        clip_end = clip_start + reading_time
        if(index==0):
            clip_end-=title_duration
        
        # Debugging output
        print(f"Processing image {image_files[index+1]}: {clip_start} to {clip_end}")

        clip = ImageClip(image_files[index+1]).set_duration(clip_end - clip_start).resize(height=720)
        clip = clip.set_audio(audio.subclip(clip_start, clip_end))
        clips.append(clip)
        
        # Update the current time for the next clip
        current_time = clip_end
    
    if clips:
        last_clip = clips[-1]
        remaining_duration = audio.duration - sum(clip.duration for clip in clips)
        if remaining_duration > 0:
            extended_clip = last_clip.set_duration(last_clip.duration + remaining_duration)
            clips[-1] = extended_clip
    video = concatenate_videoclips(clips, method="compose")
    video.write_videofile(output_file, fps=5, codec='libx264', preset='fast')
    print(f"Video created: {output_file}")
def main():
    user_notes = """
    The Spanish Inquisition
Introduction:
The Spanish Inquisition was a judicial institution established in 1478 by the Catholic Monarchs Ferdinand II of Aragon and Isabella I of Castile. Its primary goal was to maintain Catholic orthodoxy in their kingdoms and to identify and punish heresy. This institution is infamous for its role in the persecution and suppression of non-Catholic communities in Spain, including Jews and Muslims.

Historical Context:

Preceding Events: The Spanish Inquisition emerged during a period of religious and political consolidation in Spain. The Reconquista, the Christian reconquest of the Iberian Peninsula from Muslim rule, had been completed in 1492 with the fall of Granada. The Catholic Monarchs sought to unify their realms under a single religion as part of their broader effort to strengthen centralized power.
Motivations: The Inquisition was driven by a desire to ensure religious conformity and to reinforce political unity. Ferdinand and Isabella aimed to create a unified Spanish identity, which they believed was necessary for political stability and the effective consolidation of their reign.
Structure and Function:

Organization: The Spanish Inquisition was administered by the Suprema, a central governing body based in Madrid. The Suprema was responsible for overseeing the operations of local tribunals throughout Spain and its territories.
Local Tribunals: The Inquisition established local tribunals in various Spanish cities, including Seville, Valladolid, and Barcelona. These tribunals were responsible for conducting trials, interrogations, and investigations into suspected heresy.
Methods: The Inquisition used various methods to identify heretics, including denunciations, confessions, and torture. Individuals could be accused by informers or denounce themselves voluntarily. Torture was employed to extract confessions, although the extent and use of torture varied over time and across different regions.
Key Events and Figures:

Tomás de Torquemada: The first Grand Inquisitor, Tomás de Torquemada, was a central figure in the early years of the Inquisition. His tenure saw a significant increase in the number of trials and executions. Torquemada was known for his zealous enforcement of orthodoxy and his role in the expulsion of Jews from Spain in 1492.
The Expulsion of the Jews: In 1492, Ferdinand and Isabella issued the Alhambra Decree, which ordered the expulsion of all Jews from Spain. This decree was partly motivated by the Inquisition's desire to eliminate what it perceived as a source of religious and moral corruption.
The Moriscos: Muslims who converted to Christianity, known as Moriscos, were also targeted by the Inquisition. Despite their conversion, many Moriscos faced suspicion and persecution. The Inquisition's scrutiny of the Moriscos intensified, leading to their eventual expulsion from Spain in the early 17th century.
Impact and Legacy:

Social and Cultural Effects: The Inquisition had a profound impact on Spanish society and culture. It contributed to the homogenization of Spanish religious and cultural identity, often at the expense of religious and ethnic diversity. The fear of persecution and the climate of suspicion had a chilling effect on intellectual and artistic expression.
Decline: The power and influence of the Spanish Inquisition began to wane in the 18th century, especially with the rise of Enlightenment ideas advocating for religious tolerance and human rights. The Inquisition was formally abolished in 1834.
Historical Perspective: The Spanish Inquisition remains a controversial and debated subject. While some view it as a symbol of religious intolerance and persecution, others argue that its impact has been exaggerated and that it was a product of its time, reflecting broader trends in early modern European history.
Conclusion:
The Spanish Inquisition represents a dark chapter in the history of religious intolerance and persecution. Its legacy is a reminder of the dangers of absolute authority and the importance of safeguarding religious and cultural diversity. The lessons learned from this period continue to resonate in contemporary discussions about religious freedom and human rights.
"""
    # Generate video script
    video_script = generate_video_script(user_notes)
    print("Generated Script:\n", video_script)
    
    # Extract only the Narrator's lines for audio generation, including the title
    narrator_script = extract_narrator_script(video_script)
    print("Narrator Script:\n", narrator_script)
    
    # Generate audio from the narrator script
    audio_file = "output_audio.mp3"
    generate_audio(narrator_script, audio_file)
    
    # Extract graphics cues from the script
    graphic_cues = extract_graphics_cues(video_script)
    print("Graphic Cues:\n", graphic_cues)
    
    # Generate title image
    title_text = get_title(video_script)
    title_image_file = "title_image.jpg"
    create_title_image(title_text, title_image_file)
    
    # Download relevant images based on the graphics cues
    image_files = [title_image_file]  # Start with the title image
    for i, cue in enumerate(graphic_cues):
        image_file = f"image_{i}.jpg"
        download_image(cue, image_file)
        image_files.append(image_file)
    
    # Create the video with WPM-based timing
    video_file = "output_video.mp4"
    create_video(image_files, audio_file, video_script, video_file)

if __name__ == '__main__':
    main()