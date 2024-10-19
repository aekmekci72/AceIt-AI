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
        recent = ["Equilibrium", "Trig", "Molarity", "Osmosis"]
        public = ["Geometry", "Napoleon", "Functions - Python", "Diffusion", "Enthalpy", "Atoms", "Watercolor", "Cybersecurity 101"]

        question = "Given a user's most recently studied sections and public sections, select the top 3 public sections the user may be interested in. Here's user's recently studied sets: {}. Here's all public study sets: {}.".format(recent, public)
        response = await sydney.ask(question, citations=True)
        print("Sydney:", response.encode('ascii', 'ignore').decode('ascii'))

if __name__ == "__main__":
    asyncio.run(main())