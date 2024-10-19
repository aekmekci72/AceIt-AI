from stackapi import StackAPI
import requests
from bs4 import BeautifulSoup

# Initialize StackAPI client for Stack Overflow
SITE = StackAPI('stackoverflow')

def search_stackoverflow(question):
    try:
        # Search for questions related to the input question
        response = SITE.fetch('search', intitle=question, sort='relevance')
        if response['items']:
            # Return the link to the top answer/question on Stack Overflow
            return response['items'][0]['link']
        else:
            return "No relevant answer found on Stack Overflow."
    except Exception as e:
        print(f"Error: {e}")
        return None

def get_page_content(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Extract the question and answer contents
            question_content = soup.find('div', class_='s-prose js-post-body')
            answers = soup.find_all('div', class_='s-prose js-post-body')
            content = ""
            if question_content:
                content += "Question:\n" + question_content.get_text() + "\n\n"
            if answers:
                for idx, answer in enumerate(answers):
                    content += f"Answer {idx + 1}:\n" + answer.get_text() + "\n\n"
            return content.strip()
        else:
            return "Failed to retrieve the page content."
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    question = input("Enter your programming question: ")
    response = search_stackoverflow(question)
    if response:
        print("Top answer/question on Stack Overflow:")
        print(response)
        page_content = get_page_content(response)
        if page_content:
            print("\nPage Content:\n")
            print(page_content)
        else:
            print("Failed to fetch the page content.")
    else:
        print("Failed to fetch an answer from Stack Overflow.")

if __name__ == "__main__":
    main()
