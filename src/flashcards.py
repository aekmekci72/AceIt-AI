import asyncio
import os
from sydney import SydneyClient
from dotenv import load_dotenv

load_dotenv()

bing_cookies_key = os.getenv('BING_COOKIES')

if bing_cookies_key is None:
    print("Error: BING_COOKIES environment variable is not set.")
    exit(1)

os.environ["BING_COOKIES"] = bing_cookies_key

async def main() -> None:
    notes = '''The circumference of a circle is its "perimeter," or the distance around its edge. If we broke the circle and bent it into one flat line, the length of this line would be its circumference:
The diameter of a circle is a line segment from one point on the edge of the circle to another point on the edge, passing through the center of the circle. It is the longest line segment that cuts across the circle from one point to another. There are many different diameters, but they all have the same length:


Diameters of a Circle
The radius of a circle is a line segment from the center of the circle to a point on the edge of the circle. It is half of a diameter, and thus its length is half the length of the diameter. Again, there are many radii, but they all have the same length. In the following diagram, a, b, and c are all radii:


Radii of a Circle
The area of a circle is the total number of square units that fill the circle. The area of the following circle is about 13 units. Note that we count fractional units inside the circle as well as whole units.


Example: Max is building a house. The first step is to drill holes and fill them with concrete.
The holes are 0.4 m wide and 1 m deep, how much concrete should Max order for each hole?

 

circle auger
The holes are circular (in cross section) because they are drilled out using an auger.

The diameter is 0.4m, so the Area is: 0.126 m2
'''

    async with SydneyClient() as sydney:
        question = f'''
        Task: Generate 10 flashcards based on the following notes. The target is to help the student learn and remember key aspects of the notes.

        Notes:
        {notes}

        Format: 
        Question: [Your question here]
        Answer: [Your answer here]
        
        '''
        response = await sydney.ask(question, citations=True)
        print("Sydney:", response.encode('ascii', 'ignore').decode('ascii'))

if __name__ == "__main__":
    asyncio.run(main())