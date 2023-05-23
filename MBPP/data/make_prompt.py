import json
from colorama import Fore, Style

# Load promptset from JSON
with open('promptset.json', 'r') as prompt_file:
    prompt_list = json.load(prompt_file)

# Define color codes for formatting
BOLD_GREEN = Fore.GREEN + Style.BRIGHT
LIGHT_BLUE = Fore.CYAN + Style.NORMAL
GRAY = Fore.LIGHTBLACK_EX + Style.NORMAL

for prompt in prompt_list:
    text = prompt['text']
    code = prompt['code']
    test_list = prompt['test_list']

    print(BOLD_GREEN + text + Style.RESET_ALL)
    print(LIGHT_BLUE + code + Style.RESET_ALL)
    print(GRAY + str(test_list) + Style.RESET_ALL)
