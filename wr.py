import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import pandas as pd

# Download NLTK resources (run once)
nltk.download('punkt')
nltk.download('stopwords')

def fetch_web_content(url):
    try:
        # Send request to the URL and get HTML content
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract text from the HTML content
        text = soup.get_text()
        return text
    except Exception as e:
        print(f"Error fetching content: {e}")
        return None

def process_text(text):
    # Tokenize text into sentences and words
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    
    # Remove stopwords (common words like 'the', 'is', 'and')
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    
    return sentences, filtered_words

def analyze_data(words):
    # Create a DataFrame to analyze word frequency
    word_freq = nltk.FreqDist(words)
    df = pd.DataFrame(list(word_freq.items()), columns=['Word', 'Frequency'])
    df = df.sort_values(by='Frequency', ascending=False).reset_index(drop=True)
    return df

def main():
    url = 'https://en.wikipedia.org/wiki/Python_(programming_language)'
    print(f"Fetching content from {url}...")
    web_content = fetch_web_content(url)
    
    if web_content:
        sentences, filtered_words = process_text(web_content)
        
        # Example analysis: word frequency
        print("\nTop 10 Most Frequent Words:")
        word_frequency_df = analyze_data(filtered_words)
        print(word_frequency_df.head(10))
        
        # Example: print sentences
        print("\nSample Sentences:")
        for idx, sentence in enumerate(sentences[:3]):
            print(f"{idx+1}. {sentence}")
    else:
        print("Failed to fetch web content.")

if __name__ == "__main__":
    main()
