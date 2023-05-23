import os,random,sys, time
import json,re
import traceback
import linecache
from transformers import LlamaForCausalLM, LlamaTokenizer,GenerationConfig

history = dict()

fewshot_path = "/home/seungyoun/LLM2Act/MBPP/data/prompt_gpt4.txt"
llama_weight_path = '/home/seungyoun/stanford_alpaca/ckpt/7B/llama-7b'
tokenizer = LlamaTokenizer.from_pretrained("/home/seungyoun/stanford_alpaca/ckpt/7B/tokenizer")
print(f'+ Loaded tokenizer')
#model = LlamaForCausalLM.from_pretrained('/home/seungyoun/stanford_alpaca/ckpt/alpaca_7B/checkpoint-38000',device_map="auto")
model = LlamaForCausalLM.from_pretrained(llama_weight_path,device_map="auto")
model = model.eval()
#model = LlamaForCausalLM.from_pretrained('decapoda-research/llama-13b-hf',device_map="auto")
print(f'+ Loaded model')

def save_dictionary_to_json(file_path, dictionary):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=4)

generation_config = GenerationConfig(
        top_p=1.0,
)

def llm(prompts, input=None, generation_config = None):
    start = time.time()
    prompts = prompts.replace('Think :\nThink :','Think :')
    inputs = tokenizer(prompts, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    
    results = model.generate(
          input_ids=input_ids,
          generation_config=generation_config,
          return_dict_in_generate=True,
          output_scores=True,
          max_new_tokens=300,
          use_cache=True
    )

    returns = list()
    gen_list = list()
    for i,s in enumerate(results.sequences):
        gen = tokenizer.decode(s)
        
        given = prompts
        gen = gen.split(given)[-1]
        gen_list.append(gen)

        print('\033[34m' + prompts+ '\033[0m' + '\033[32m' + gen+ '\033[0m')
        print("\n==================================\n")

    end = time.time()
    #print(f'LLM took {end-start:.2f} seconds')
    return random.choice(gen_list)

import json
from colorama import Fore, Style

# Load promptset from JSON
with open('promptset.json', 'r') as prompt_file:
    prompt_list = json.load(prompt_file)

# Define color codes for formatting
BOLD_GREEN = Fore.GREEN + Style.BRIGHT
LIGHT_BLUE = Fore.CYAN + Style.NORMAL
GRAY = Fore.LIGHTBLACK_EX + Style.NORMAL

inp = """Instruction : Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix cost[][] and a position (m, n) in cost[][].

Think : I think I need to use dynamic programming to find minimum cost to get to the d[i][j] from d[0][0]
Action : [Generate Python Code]
```python
def"""

for prompt in prompt_list:
    text = prompt['text']
    code = prompt['code']
    test_list = prompt['test_list']
    
    #gen = llm(f'{text}\ndef')
    gen = llm(inp)

    print(BOLD_GREEN + text + Style.RESET_ALL)
    print(LIGHT_BLUE + code + Style.RESET_ALL)
    print(GRAY + str(test_list) + Style.RESET_ALL)
    print(gen)
    print("--------")
