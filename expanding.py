import requests

def get_wikipedia_summary(concept):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "titles": concept
    }

    response = requests.get(url, params=params)
    data = response.json()
    pages = data["query"]["pages"]
    page_id = next(iter(pages))
    summary = pages[page_id]["extract"]

    return summary

# Example usage
concept = "Natural language processing"
summary = get_wikipedia_summary(concept)
print(f"Summary for '{concept}':")
print(summary)
