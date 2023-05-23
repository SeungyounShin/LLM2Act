import os,json 
from colorama import Fore, Style

path = "/home/seungyoun/LLM2Act/MBPP/history/llama_react_test.json"

# Load promptset from JSON
with open(path, 'r') as hist_file:
    hist_list = json.load(hist_file)

# Define color codes for formatting
BOLD_GREEN = Fore.GREEN + Style.BRIGHT
LIGHT_BLUE = Fore.CYAN + Style.NORMAL
GRAY = Fore.LIGHTBLACK_EX + Style.NORMAL

for prompt in hist_list:
    traj = prompt['traj']

    print(traj)
    print('======')

    #print(BOLD_GREEN + text + Style.RESET_ALL)
    #print(LIGHT_BLUE + code + Style.RESET_ALL)
    #print(GRAY + str(test_list) + Style.RESET_ALL)
