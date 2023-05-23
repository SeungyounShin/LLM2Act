import os,random,sys, time
import json,re
import traceback
import linecache
from transformers import LlamaForCausalLM, LlamaTokenizer,GenerationConfig

history = dict()

fewshot_path = "/home/seungyoun/LLM2Act/MBPP/data/prompt_gpt4.txt"

def save_dictionary_to_json(file_path, dictionary):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=4)

generation_config = GenerationConfig(
        top_p=1.0,
)

import openai
 
openai.api_key = os.environ["OPENAI_API_KEY"]

def llm(prompt, stop=["Observation"]):
    response = openai.Completion.create(
      model="text-davinci-002",
      prompt=prompt,
      temperature=0,
      max_tokens=350,
      top_p=1,
      frequency_penalty=0.0,
      presence_penalty=0.0,
      stop=stop
    )
    return response["choices"][0]["text"]


import json
from colorama import Fore, Style

# Load promptset from JSON
with open('/home/seungyoun/LLM2Act/MBPP/data/trainset.json', 'r') as prompt_file:
    prompt_list = json.load(prompt_file)

def load_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return data

fewshot_prompt = load_file(fewshot_path).strip()

# Define color codes for formatting
BOLD_GREEN = Fore.GREEN + Style.BRIGHT
GREEN = Fore.GREEN + Style.NORMAL
LIGHT_BLUE = Fore.CYAN + Style.NORMAL
GRAY = Fore.LIGHTBLACK_EX + Style.NORMAL

def mbpp_run(instruciton, code, test_list, max_step=5, to_print=True):
    test_assertion = '\n'.join(test_list)
    prompt = f'{fewshot_prompt}\n\nInstruction : {instruciton}\n\nTest Case :\n{test_assertion}'

    traj = f'Instruction : {instruciton}\n\nTest Case :\n{test_assertion}'

    for step in range(max_step):
        prompt += '\n\nThink :'
        traj += '\n\nThink :'
        gen = llm(prompt)
        # process
        gen = gen.split('Observation :')[0].strip()
        if 'Action : [FINISH]' in gen:
            if to_print:
                print('Think : ' + GREEN + gen + Style.RESET_ALL)
            traj += f'\nThink : {gen}'
            break 
        # parse code 
        match = re.search(r'```python\s+(.*?)\s+```', gen, re.DOTALL)
        code,function_name = '',''
        if match:
            code = match.group(1)
            match_def = re.search(r'def\s+([^\s(]+)', code)

            if match_def:
                function_name = match_def.group(1)

                modified_assertions = []
                for assertion in test_list:
                    modified_assertion = re.sub(r'(\bassert\s+)([^\s(]+)', fr'\1{function_name}', assertion)
                    modified_assertions.append(modified_assertion)
                
                test_assertion = '\n'.join(modified_assertions)

        if to_print:
            print('Think : ' + GREEN + gen + Style.RESET_ALL)
        traj += gen
        prompt += gen
        
        # code exec
        error_msg = None
        error_string = None
        try:
            exec(f'{code}\n{test_assertion}')
        except Exception as e:
            error_msg = e 
            #error_class = error_msg.__class__.__name__
            #detail = error_msg.args[0]
            _, _, tb = sys.exc_info()
            tb_now = tb
            error_string = ""
            while tb_now is not None:
                filename = tb_now.tb_frame.f_code.co_filename
                line_number = tb_now.tb_lineno
                if filename == "<string>":
                    line_of_code = (code + "\n" + test_assertion).split('\n')[line_number-1]
                    error_string = f'Error in file {filename} on line {line_number}\n' + \
                                f'Code at line {line_number}: {line_of_code}'
                tb_now = tb_now.tb_next
        
        # make Observation
        prompt += '\n'
        traj += '\n'

        if error_msg is None: 
            prompt += 'Observation : \nNo Error'
            traj += 'Observation : \nNo Error'
            if to_print:
                print('Observation : \n' + LIGHT_BLUE + 'No Error' + Style.RESET_ALL)
        else:
            prompt += f'Observation : \n{error_string}'
            traj +=  f'Observation : \n{error_string}'
            if to_print:
                print('Observation : \n' + LIGHT_BLUE + error_string + Style.RESET_ALL)
        
    # eval again

    # code exec
    error_msg = None
    try:
        exec(f'{code}\n\n{test_assertion}')
    except Exception as e:
        error_msg = e 
        #error_class = error_msg.__class__.__name__
        #detail = error_msg.args[0]
        _, _, tb = sys.exc_info()
        traceback_info = ''.join(traceback.format_exception(None, error_msg, tb))

    print('=================')
    print(GRAY + traj + Style.RESET_ALL)
    print('=================')

    if error_msg is not None:
        return {
            'traj' : traj, 'success' : False 
        } 
    return {
            'traj' : traj, 'success' : True 
    } 

def print_stat(hist):
    success_cnt = 0
    
    for h in hist:
        if h['success']:
            success_cnt += 1
    
    SR = success_cnt/len(hist)

    print(f' DEMO : {len(hist)} | SR : {SR}')

    # save hist 
    with open('/home/seungyoun/LLM2Act/MBPP/history/gpt3_react_train.json', 'w') as file:
        json.dump(hist, file)


hist = list()
for prompt in prompt_list:
    text = prompt['text']
    code = ''#prompt['code']
    test_list = prompt['test_list']

    output_dict = mbpp_run(text , code , test_list)
    hist.append(output_dict)

    print_stat(hist)