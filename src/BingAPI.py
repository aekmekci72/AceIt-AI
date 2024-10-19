import asyncio
import os
from sydney import SydneyClient
from dotenv import load_dotenv

import json

load_dotenv()

bing_cookies_key = os.getenv('BING_COOKIES')

if bing_cookies_key is None:
    print("Error: BING_COOKIES environment variable is not set.")
    exit(1)

os.environ["BING_COOKIES"] = bing_cookies_key


async def main() -> None:
    async with SydneyClient() as sydney:
        question = 'Given these notes: "Review (basics): #Creating functions def my_functionQ: pass #Decomposition * Breaking down big ideas to small helper functions (think back to library exercise we did last time) #While loops - indefinite loops (till something is false) while condition: pass #For loops- definite loops (run a fixed number of times) for iin range(10): pass ". Identify key search phrases that I could use to generate relevant static images to accompany each concept. These images will serve as visual aids to enhance student comprehension. Consider the context and keywords associated with each concept to ensure the images are directly related and supportive of the provided summary'
        response = await sydney.ask(question, citations=True)
        print("Sydney:", response.encode('ascii', 'ignore').decode('ascii'))
    # async with SydneyClient() as sydney:
    #     response, suggested_responses = await sydney.compose(
    #         prompt="Why Python is a great language", format="ideas", suggestions=True,
    #     )

    #     response, suggested_responses = await sydney.compose(
    #         prompt=suggested_responses[0], format="ideas", suggestions=True
    #     )

    #     print(response)

if __name__ == "__main__":
    asyncio.run(main())