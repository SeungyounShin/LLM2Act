import os,random,sys, time
import json,re
import traceback
import torch 
import linecache
from transformers import LlamaForCausalLM, LlamaTokenizer,GenerationConfig, StoppingCriteria, StoppingCriteriaList

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

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_ids:list):
        self.keywords = keywords_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids[0][-1] in self.keywords:
            return True
        return False

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
    
    stop_words = ['Observation :','==========']
    stop_ids = [tokenizer.encode(w)[0] for w in stop_words]
    stop_criteria = KeywordsStoppingCriteria(stop_ids)


    results = model.generate(
          input_ids=input_ids,
          generation_config=generation_config,
          return_dict_in_generate=True,
          output_scores=True,
          max_new_tokens=512,
          stopping_criteria=StoppingCriteriaList([stop_criteria]),
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
with open('/home/seungyoun/LLM2Act/MBPP/data/testset.json', 'r') as prompt_file:
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

                #modified_assertions = []
                #for assertion in test_list:
                #    modified_assertion = re.sub(r'(\bassert\s+)([^\s(]+)', fr'\1{function_name}', assertion)
                #    modified_assertions.append(modified_assertion)
                #
                #test_assertion = '\n'.join(modified_assertions)

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
                    lines = (code + "\n" + test_assertion).split('\n')
                    line_of_code = lines[min(line_number-1, len(lines))-1]
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
    if code=='':
        return {
            'traj' : traj, 'success' : False 
        }
    try:
        exec(f'{code}\n\n{test_assertion}')
    except Exception as e:
        error_msg = e 
        #error_class = error_msg.__class__.__name__
        #detail = error_msg.args[0]
        _, _, tb = sys.exc_info()
        traceback_info = ''.join(traceback.format_exception(None, error_msg, tb))

    if error_msg is not None:
        print('=================')
        print(GRAY + traj + Style.RESET_ALL)
        print('========[FAIL]=========')

        return {
            'traj' : traj, 'success' : False 
        } 

    print('=================')
    print(GRAY + traj + Style.RESET_ALL)
    print('========[SUCCESS]=========')

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
    with open('/home/seungyoun/LLM2Act/MBPP/history/llama_react_test.json', 'w') as file:
        json.dump(hist, file)

    if len(hist) >=100:
        exit()


hist = list()
for prompt in prompt_list[1:]:
    text = prompt['text']
    code = ''
    test_list = prompt['test_list']

    output_dict = mbpp_run(text , code , test_list)
    hist.append(output_dict)

    print_stat(hist)