import os,random,sys, time
import json
from transformers import LlamaForCausalLM, LlamaTokenizer,GenerationConfig

history = dict()

llama_weight_path = '/home/seungyoun/stanford_alpaca/ckpt/webshop_full_onlysucess/checkpoint-132'
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
        temperature=0.2,
        top_p=0.8,
)

def llm(prompts, input=None, generation_config = None):
    start = time.time()
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
        gen = gen.split(given)[-1]
        gen_list.append(gen)

        #print('\033[34m' + prompts+ '\033[0m' + '\033[32m' + gen+ '\033[0m')
        #print("\n==================================\n")

    end = time.time()
    #print(f'LLM took {end-start:.2f} seconds')
    return random.choice(gen_list)

def generate_new_instruction():
  insturction_path = '/home/seungyoun/webshop/demo_files/webshop_demos_instruction_only.json'
  with open(insturction_path) as f:
    instructions = json.load(f)
  
  # select randomly 2 instructions
  instruction = random.choices(instructions, k=3)

  instruction_prompt = f"Instruction:\n{instruction[0]['instruction']}\n\nInstruction:\n{instruction[1]['instruction']}\n\nInstruction:\n{instruction[2]['instruction']}\n\nInstruction:"

  generation_config = GenerationConfig(
        temperature=0.2,
        top_p=0.8,
  )

  self_gen_instruction = llm(instruction_prompt, generation_config=generation_config)
  self_gen_instruction = self_gen_instruction.split('\n')[1]

  return self_gen_instruction
  

import requests
from bs4 import BeautifulSoup
from bs4.element import Comment

WEBSHOP_URL = "http://3.83.245.205:3000"
ACTION_TO_TEMPLATE = {
    'Description': 'description_page.html',
    'Features': 'features_page.html',
    'Reviews': 'review_page.html',
    'Attributes': 'attributes_page.html',
}

def clean_str(p):
  return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


def tag_visible(element):
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return (
        element.parent.name not in ignore and not isinstance(element, Comment)
    )


def webshop_text(session, page_type, query_string='', page_num=1, asin='', options={}, subpage='', **kwargs):
    if page_type == 'init':
      url = (
          f'{WEBSHOP_URL}/{session}'
      )
    if page_type == 'search':
      url = (
          f'{WEBSHOP_URL}/search_results/{session}/'
          f'{query_string}/{page_num}'
      )
    elif page_type == 'item':
      url = (
          f'{WEBSHOP_URL}/item_page/{session}/'
          f'{asin}/{query_string}/{page_num}/{options}'
      )
    elif page_type == 'item_sub':
      url = (
          f'{WEBSHOP_URL}/item_sub_page/{session}/'
          f'{asin}/{query_string}/{page_num}/{subpage}/{options}'
      )
    elif page_type == 'end':
      url = (
          f'{WEBSHOP_URL}/done/{session}/'
          f'{asin}/{options}'
      )
    # print(url)
    html = requests.get(url).text
    html_obj = BeautifulSoup(html, 'html.parser')
    texts = html_obj.findAll(text=True)
    visible_texts = list(filter(tag_visible, texts))
    # visible_texts = [str(text).strip().strip('\\n') for text in visible_texts]
    # if page_type == 'end': import pdb; pdb.set_trace()
    if False:
        # For `simple` mode, return just [SEP] separators
        return ' [SEP] '.join(t.strip() for t in visible_texts if t != '\n')
    else:
        # Otherwise, return an observation with tags mapped to specific, unique separators
        observation = ''
        option_type = ''
        options = {}
        asins = []
        cnt = 0
        prod_cnt = 0
        just_prod = 0
        for t in visible_texts:
            if t == '\n': continue
            if t.replace('\n', '').replace('\\n', '').replace(' ', '') == '': continue
            # if t.startswith('Instruction:') and page_type != 'init': continue
            # print(t.parent.name, t)
            if t.parent.name == 'button':  # button
                processed_t = f'\n[{t}] '
            elif t.parent.name == 'label':  # options
                if f"'{t}'" in url:
                    processed_t = f'[[{t}]]'
                    # observation = f'You have clicked {t}.\n' + observation
                else:
                    processed_t = f'[{t}]'
                options[str(t)] = option_type
                # options[option_type] = options.get(option_type, []) + [str(t)]
            elif t.parent.get('class') == ["product-link"]: # product asins
                processed_t = f'\n[{t}] '
                if prod_cnt >= 3:
                  processed_t = ''
                prod_cnt += 1
                asins.append(str(t))
                just_prod = 0
            else: # regular, unclickable text
                processed_t =  '\n' + str(t) + ' '
                if cnt < 2 and page_type != 'init': processed_t = ''
                if just_prod <= 2 and prod_cnt >= 4: processed_t = ''
                option_type = str(t)
                cnt += 1
            just_prod += 1
            observation += processed_t
        info = {}
        if options:
          info['option_types'] = options
        if asins:
          info['asins'] = asins
        if 'Your score (min 0.0, max 1.0)' in visible_texts:
          idx = visible_texts.index('Your score (min 0.0, max 1.0)')
          info['reward'] = float(visible_texts[idx + 1])
          observation = 'Your score (min 0.0, max 1.0): ' + (visible_texts[idx + 1])
        return clean_str(observation), info

class webshopEnv:
  def __init__(self):
    self.sessions = {}
  
  def step(self, session, action):
    done = False
    observation_ = None
    if action == 'reset':
      self.sessions[session] = {'session': session, 'page_type': 'init'}
    elif action.startswith('think['):
      observation = 'OK.'
    elif action.startswith('search['):
      assert self.sessions[session]['page_type'] == 'init'
      query = action[7:-1]
      self.sessions[session] = {'session': session, 'page_type': 'search',
                                'query_string': query, 'page_num': 1}
    elif action.startswith('click['):
      button = action[6:-1]
      if button == 'Buy Now':
        assert self.sessions[session]['page_type'] == 'item'
        self.sessions[session]['page_type'] = 'end'
        done = True
      elif button == 'Back to Search':
        assert self.sessions[session]['page_type'] in ['search', 'item_sub', 'item']
        self.sessions[session] = {'session': session, 'page_type': 'init'}
      elif button == 'Next >':
        assert False # ad hoc page limitation
        assert self.sessions[session]['page_type'] == 'search'
        self.sessions[session]['page_num'] += 1
      elif button == '< Prev':
        assert self.sessions[session]['page_type'] in ['search', 'item_sub', 'item']
        if self.sessions[session]['page_type'] == 'search':
          assert False
          self.sessions[session]['page_num'] -= 1
        elif self.sessions[session]['page_type'] == 'item_sub':
          self.sessions[session]['page_type'] = 'item'
        elif self.sessions[session]['page_type'] == 'item':
          self.sessions[session]['page_type'] = 'search'
          self.sessions[session]['options'] = {}
      elif button in ACTION_TO_TEMPLATE:
        assert self.sessions[session]['page_type'] == 'item'
        self.sessions[session]['page_type'] = 'item_sub'
        self.sessions[session]['subpage'] = button
      else:
        if self.sessions[session]['page_type'] == 'search':
          assert button in self.sessions[session].get('asins', [])  # must be asins
          self.sessions[session]['page_type'] = 'item'
          self.sessions[session]['asin'] = button
        elif self.sessions[session]['page_type'] == 'item':
          assert 'option_types' in self.sessions[session]
          assert button in self.sessions[session]['option_types'], (button, self.sessions[session]['option_types'])  # must be options
          option_type = self.sessions[session]['option_types'][button]
          if not 'options' in self.sessions[session]:
            self.sessions[session]['options'] = {}
          self.sessions[session]['options'][option_type] = button
          observation_ = f'You have clicked {button}.'
    else:
      assert False
    observation, info = webshop_text(**self.sessions[session])
    if observation_:
      observation = observation_
    self.sessions[session].update(info)
    reward = info.get('reward', 0.0)
    return observation, reward, done
env = webshopEnv()

def webshop_run(idx, prompt, to_print=True):
  action = 'reset'
  init_prompt = prompt
  prompt = ''
  for i in range(15):
    try:
      res = env.step(idx, action)
      observation = res[0]
    except AssertionError:
      observation = 'Invalid action!'

    if action.startswith('think'):
      observation = 'OK.'


    if to_print:
      # print action in green 
      print(f'\033[92mAction: {action}\033[0m')
      # print observation in grey
      print(f'\033[90mObservation: {observation}\033[0m')
      #print(f'Action: {action}\nObservation: {observation}\n')
      sys.stdout.flush()
    if i:
      prompt += f' {action}\nObservation: {observation}\n\nAction:'
    else:
      #prompt += f'{observation}\n\nAction:'
      new_insturction = generate_new_instruction()
      prompt += f'Action: reset\nObservation:\nWebshop\nInstruction:\n{new_insturction}\n\nAction:'
    
    if res[2]:  
      # print in blue
      print(f'\033[94m{prompt}\033[0m')
      history[idx] = prompt
      save_dictionary_to_json('history_selfplay.json', history)
      return res[1]

    action = llm(init_prompt + prompt[-(6400-len(init_prompt)):], generation_config = generation_config).split('\n')[0].lstrip(' ')
  
  # print in red  
  print(f'\033[91m{prompt}\033[0m')
  history[idx] = prompt
  save_dictionary_to_json('history_selfplay.json', history)
  return 0

def run_episodes(prompt, n=50):
  rs = []
  cnt = 0
  for i in range(n):
    st = time.time()
    print('-----------------')
    print(i)
    try:
      r = webshop_run(f'fixed_{i+600}', prompt, to_print=True)
    except AssertionError:
      r = 0
      cnt += 1
    rs.append(r)
    end = time.time()
    if (i+1) % 1 == 0:
      r, sr, fr = sum(rs) / len(rs), len([_ for _ in rs if _ == 1]) / len(rs), cnt / len(rs)
      print(i+1, r, sr, fr)
      print(f'----------------- {end-st} seconds')
  r, sr, fr = sum(rs) / len(rs), len([_ for _ in rs if _ == 1]) / n, cnt / n
  print(r, sr, fr)
  return rs

#res1 = run_episodes(prompt1, 50)
res1 = run_episodes('', 5000)

