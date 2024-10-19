import requests

def check_profanity(text):
    url = "https://www.purgomalum.com/service/containsprofanity"
    params = {"text": text}
    response = requests.get(url, params=params)
    return response.text == 'true'

text = "Your text to check"
if check_profanity(text):
    print("Profanity detected")
else:
    print("No profanity detected")
