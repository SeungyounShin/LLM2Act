"""
    This code is adapted from 
    https://github.com/GammaTauAI/reflexion-human-eval/blob/main/generators/generator_utils.py#L11
"""


import os
import gzip
import json
import openai
from tenacity import (
    retry,
    stop_after_attempt, # type: ignore
    wait_random_exponential, # type: ignore
)

from typing import Union, List, Optional

openai.api_key = os.getenv("OPENAI_API_KEY")

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def gpt_completion(
        model: str,
        prompt: Union[str, List[str]],
        max_tokens: int = 512,
        stop_strs: Optional[List[str]] = None,
        temperature: float = 0.0,
    ) -> Union[str, List[str]]:
    # check if batched or not
    is_batched = isinstance(prompt, list)
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop_strs,
    )
    if is_batched:
        res: List[str] = [""] * len(prompt)
        for choice in response.choices: # type: ignore
            res[choice.index] = choice.text
        return res
    return response.choices[0].text # type: ignore

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def gpt_chat(
        model: str,
        system_message: str = "ChatGpt",
        user_message: str = "hi",
        max_tokens: int = 612,
        temperature: float = 0.0,
        message: List = [],
    ) -> str:
    
    if len(message) == 0:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
    else:
        response = openai.ChatCompletion.create(
            model=model,
            messages=message,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )

    return response.choices[0].message.content # type: ignore

def parse_body(text):
    lines = text.split('\n')
    for i in range(len(lines)-1, -1, -1):
        if 'return' in lines[i]:
            return '\n'.join(lines[:i+1])
    return text

if __name__=='__main__':
    system_message ="""
    You are CodeGPT, an AI model that solve competitive programming problems.
    """

    user1 = """Instruction : Write a function to find squares of individual elements in a list using lambda function.

Test Case : 
assert square_nums([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])==[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
assert square_nums([10,20,30])==[100,400,900]
assert square_nums([12,15])==[144,225]"""

    asst1 = """
    Think : To square each element in a given list, I can use Python's built-in map function, which applies a given function to each item of an iterable (such as a list). As the function to apply, I can use a lambda function that takes one argument and returns its square. The map function returns a map object, which I will convert to a list using the list function.
Action : [Generate Python Code]
```python
def square_nums(nums):
    return list(map(lambda x: x ** 2, nums))
```"""

    user1_2 = """Observation : 
No Error"""

    asst1_2 = """Think : I think I successfully finished the code. I need to submit the code.
Action : [FINISH]"""

    user2 = """Instruction : Write a function to find the number of ways to fill it with 2 x 1 dominoes for the given 3 x n board.

Test Case : 
assert count_ways(2) == 3
assert count_ways(8) == 153
assert count_ways(12) == 2131"""
  
    
    msg = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user1 },
        {"role": "assistant", "content": asst1},
        {"role": "user", "content": user2},
    ]

    # test gpt_chat
    print(gpt_chat('gpt-4', message = msg))