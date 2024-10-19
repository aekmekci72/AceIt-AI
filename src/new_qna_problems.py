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

async def generate_questions(subject: str) -> None:
    # Define notes and question formats for different subjects
    subjects = {
        "computer science": {
            "notes": '''
                Recursion in computer science is a method where the solution to a problem depends on solutions to smaller instances of the same problem (as opposed to iteration). Recursive algorithms have two cases: a recursive case and base case. Any function that calls itself is recursive.

    Examples of recursive functions:

    Factorial: n! = n x (n -1) x (n-2) x ... x 1
    Fibonacci: 1,1,2,3,5,8 ,...
    Multiplication (3 x 2): 3 + 3
    Multiplication (2 x 3): 2 + 2 + 2
    Summations from i = 1 to 5: 1 + 2 + 3 + 4 + 5
    n^2 + (n-1) ^2 + (n-2)^2 + ... + 1
    1 + 10 + 100 + 1000 + 10000 + .....
    1 + 1 + 1 + ... + 1
    0 + 1 + 2 + 3 + ... + n
    func( 0 ) = 0 , func(n) = func(n-1) + n
    Recursion is useful for tasks that can be defined in terms of similar subtasks, for example search, sort , and traversal problems often have simple recursive solutions. At some point the function encounters a subtask that it can perform without calling itself.

    Directly recursive: method that calls itself
    Indirectly recursive: method that calls another method and eventually results in the original method call
    Tail recursive method: recursive method in which the last statement executed is the recursive call
    Infinite recursion: case where every recursive call results in another recursive call
    Factorial Numbers
    Factorial numbers defined recursively example:
    factorial(0) = 1;
    factorial(n) = n * factorial(n-1);
    Examples
    0! = 1
    1! = 1 * 1 = 1
    2! = 2 * 1 * 1 = 2
    3! = 3 * 2 * 1 * 1 = 6
    4! = 4 * 3 * 2 * 1 * 1 = 24

    Tower of Hanoi recursive program (Javascript)
    /*
    *This is a Tower of hanoi recursive program
    * written in javascript
    *This program will tell you the correct
    * moves to make for any number of disks 'n'
    * in this case n = 4
    */

    function move(n, a, b, c) {
    if (n > 0) {
        move(n-1, a, c, b);
        console.log("Move disk from " + a + " to " + c);
        move(n-1, b, a, c);
    }
    }
    move(4, "A", "B", "C");

    Recursion function to multiply two numbers
    Multiplication can be thought of as a recursive function. Multiplication is simply adding the number 'X' 'Y' times or vice versa. For example if I multiplied 5 by 3 (e.g. 5 * 3) the way multiplication works, we get 5 + 5 + 5 = 15 or 3 + 3 +3+ 3+ 3= 15 both are correct ways to do multiplication. This works perfectly for positive integers, but what if the we wanted to multiply 5 * 0 = 0 or 0 * 5 =0 and 5 * 1 = 5 or 1 * 5 = 5, that will be our base case also known as the stopping or non-recursive case.

    So what will the recursive program look like ? Base case if input X or input Y is 0 we will return 0, if X is 1 then we return Y , if Y is 1then we return X. Both X and Y are our input parameter variables. You look at the multiplication function next to this paragraph to see how we would multiply two positive numbers recursively. Note: you cannot use this function for negative values.

            ''',
            "question_format": '''
            Question Format: Write a recursive function to [solve a specific problem/task].
            Question: [Your question here]
            Options: [None, as it's a coding or tracing question]
            Answer: [Correct code solution]

            [Explanation of the problem and how the recursive function works]
            '''
        },
        "math": {
            "notes": '''The circumference of a circle is its "perimeter," or the distance around its edge. If we broke the circle and bent it into one flat line, the length of this line would be its circumference:
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
            ''',
            "question_format": '''
            Question Format: [Specify question format for math]
            Question: [Your math question here]
            Options: [Option 1, Option 2, Option 3, Option 4] (for multiple choice questions)
            Answer: [Correct answer]

            [Explanation of the math concept/problem]
            '''
        },
        # Add more subjects as needed
    }

    # Fetch notes and question format based on the subject
    if subject.lower() not in subjects:
        print("Error: Subject not supported.")
        return

    async with SydneyClient() as sydney:
        notes = subjects[subject.lower()]["notes"]
        question_format = subjects[subject.lower()]["question_format"]

        question = f'''
        Task: Generate practice test questions based on the following notes for {subject}.

        Notes:
        {notes}

        {question_format}

        Citations: [Include citations where necessary]
        '''
        response = await sydney.ask(question, citations=True)
        print("Sydney:", response.encode('ascii', 'ignore').decode('ascii'))

if __name__ == "__main__":
    # Specify the subject for which you want to generate questions
    subject = "computer science"  # Change this to any supported subject
    asyncio.run(generate_questions(subject))
