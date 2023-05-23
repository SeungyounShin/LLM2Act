import os,random,sys, time
import json
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer,GenerationConfig,AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead

model_name = 'llama'

green_color = "\033[92m"
blue_color = "\033[94m"
gray_color = "\033[90m"
reset_color = "\033[0m"

COLOR_RED = "\033[91m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_END = "\033[0m"

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

CHAT_JSON_PATH = '/home/seungyoun/LLM2Act/FEVER/history/fever_chat_dict.json'

with open(CHAT_JSON_PATH, 'r') as f:
    chat_dict_list = json.load(f)

chat_dict = {}

for item in chat_dict_list:
    instruction = item['instruction']
    chat = item['chat']
    chat_dict[instruction] = f'User: {chat}'

history = dict()

if model_name=='gpt2':
    tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2-large',)
    print(f'+ Loaded tokenizer')

    model = transformers.GPT2LMHeadModel.from_pretrained('gpt2-large', pad_token_id=tokenizer.eos_token_id)
    model = model.eval().cuda()
    print(f'+ Loaded model {model_name}')
    #tokenizer.pad_token_id = 50257 # hard-coded

elif model_name=='llama':
    #llama_weight_path = '/home/seungyoun/stanford_alpaca/ckpt/webshop_full_onlysucess_chat_full/checkpoint-264'
    llama_weight_path = '/home/seungyoun/stanford_alpaca/ckpt/7B/llama-7b'
    tokenizer = LlamaTokenizer.from_pretrained("/home/seungyoun/stanford_alpaca/ckpt/7B/tokenizer")
    print(f'+ Loaded tokenizer')
    #model = LlamaForCausalLM.from_pretrained('/home/seungyoun/stanford_alpaca/ckpt/alpaca_7B/checkpoint-38000',device_map="auto")
    model = LlamaForCausalLM.from_pretrained(llama_weight_path,device_map="auto")
    #model = LlamaForCausalLM.from_pretrained('decapoda-research/llama-13b-hf',device_map="auto")
    print(f'+ Loaded model {model_name}')

def save_dictionary_to_json(file_path, dictionary):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=4)

generation_config = GenerationConfig(
        temperature=0.2,
        top_p=0.8,)

def llm(prompts, input=None, generation_config = None, stop=None):
    start = time.time()
    prompts = prompts.replace('  ', '')

    claim_in_prompt = [i.split('\n')[0] for i in prompts.split('Claim: ') if i.split('\n')[0] is not '']
    for c in claim_in_prompt:
        try:
            to_chat = chat_dict[c]
            prompts = prompts.replace(c, to_chat)
        except:
            pass 
    
    prompts = prompts.replace('Claim: User:', 'Claim\nUser:')

    inputs = tokenizer(prompts, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()

    results = model.generate(
          input_ids=input_ids,
          generation_config=generation_config,
          return_dict_in_generate=True,
          output_scores=True,
          max_new_tokens=64,
          use_cache=True
    )

    returns = list()
    gen_list = list()
    for i,s in enumerate(results.sequences):
        gen = tokenizer.decode(s)
        
        given = prompts
        gen = gen.split(given)[-1].split(stop[0])[0].split('Claim')[0]
        gen_list.append(gen)

        #print('\033[34m' + prompts+ '\033[0m' + '\033[32m' + gen+ '\033[0m')
        #print("\n==================================\n")

    end = time.time()
    #print(f'LLM took {end-start:.2f} seconds')
    return random.choice(gen_list)


import wikienv, wrappers
env = wikienv.WikiEnv()
env = wrappers.FeverWrapper(env, split="dev")
env = wrappers.LoggingWrapper(env)

def step(env, action):
    attempts = 0
    while attempts < 10:
        try:
            return env.step(action)
        except requests.exceptions.Timeout:
            attempts += 1

import json
import sys

folder = './prompts/'
prompt_file = 'fever.json'
with open(folder + prompt_file, 'r') as f:
    prompt_dict = json.load(f)

webthink_prompt = prompt_dict['webthink_simple3']

def webthink(idx=None, prompt=webthink_prompt, to_print=True):
    question = env.reset(idx=idx)
    if to_print:
        print(idx, question)
    prompt += question + "\n"
    n_calls, n_badcalls = 0, 0
    for i in range(1, 8):
        n_calls += 1
        llm_input_prompt = prompt + f"Thought {i}:"
        if model_name == 'gpt2':
            llm_input_prompt = llm_input_prompt[-1024:]
        thought_action = llm(llm_input_prompt, stop=[f"\nObservation {i}:"])
        try:
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
        except:
            print('ohh...', thought_action)
            n_badcalls += 1
            n_calls += 1
            thought = thought_action.strip().split('\n')[0]
            llm_input_prompt = prompt + f"Thought {i}: {thought}\nAction {i}:"
            if model_name == 'gpt2':
                llm_input_prompt = llm_input_prompt[-1024:]
            action = llm(llm_input_prompt, stop=[f"\n"]).strip()
        
        print(f"{COLOR_BLUE}Action {i}:{COLOR_END} {COLOR_GREEN}{action}{COLOR_END}")

        try:
            obs, r, done, info = step(env, action[0].lower() + action[1:])
        except:
            continue 

        obs = obs.replace('\\n', '')
        step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
        prompt += step_str
        if to_print:
            print(f"\033[92mThought {i}:\033[0m {thought}\n\033[94mAction {i}:\033[0m {action}\n\033[90mObservation {i}:\033[0m {obs}\n")
        if done:
            break
    if not done:
        obs, r, done, info = step(env, "finish[]")
    if to_print:
        print(info, '\n')
    info.update({'n_calls': n_calls, 'n_badcalls': n_badcalls, 'traj': prompt})
    return r, info

import random
import time
idxs = list(range(7405))
random.Random(233).shuffle(idxs)

traj_list = list()
rs = []
infos = []
old_time = time.time()
for i in idxs[:500]:
    if int(i) != 6837:
        continue
    r, info = webthink(i, to_print=True)
    rs.append(info['em'])
    infos.append(info)
    print(sum(rs), len(rs), sum(rs) / len(rs), (time.time() - old_time) / len(rs))
    print('-----------')
    traj_list.append(info['traj'])
    print()

    # save traj_list to json 
    if len(rs)%10 == 0:
        with open(f'./history/{model_name}_react_chat_fewshot_tmp.json', 'w') as file:
            json.dump(traj_list, file)
        print('Save trajectory')

with open(f'./history/{model_name}_react_chat_fewshot_tmp.json', 'w') as file:
    json.dump(traj_list, file)