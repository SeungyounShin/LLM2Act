import json, time, torch
from transformers import LlamaForCausalLM, LlamaTokenizer,GenerationConfig

from webshop_lite import dict_to_fake_html
from predict_help import (
    Page, convert_dict_to_actions, convert_html_to_text,
    parse_results_amz, parse_item_page_amz,
    parse_results_ws, parse_item_page_ws,
    parse_results_ebay, parse_item_page_ebay,
    WEBSHOP_URL, WEBSHOP_SESSION
)

ENVIRONMENTS = ['amazon', 'webshop', 'ebay']

llama_weight_path = '/home/seungyoun/stanford_alpaca/ckpt/webshop_fever_react'
tokenizer = LlamaTokenizer.from_pretrained("/home/seungyoun/stanford_alpaca/ckpt/7B/tokenizer")
print(f'+ Loaded llama tokenizer')
model = LlamaForCausalLM.from_pretrained(llama_weight_path,device_map="auto")
model = model.eval()
print(f'+ Loaded llama model')

def process_str(s):
    s = s.lower().replace('"', '').replace("'", "").strip()
    s = s.replace('[sep]', '[SEP]')
    return s

def process_goal(state):
    state = state.lower().replace('"', '').replace("'", "")
    state = state.replace('amazon shopping game\ninstruction:', '').replace('webshop\ninstruction:', '')
    state = state.replace('\n[button] search [button_]', '').strip()
    if ', and price lower than' in state:
        state = state.split(', and price lower than')[0]
    return state

def data_collator(batch):
    state_input_ids, state_attention_mask, action_input_ids, action_attention_mask, sizes, labels, images = [], [], [], [], [], [], []
    for sample in batch:
        state_input_ids.append(sample['state_input_ids'])
        state_attention_mask.append(sample['state_attention_mask'])
        action_input_ids.extend(sample['action_input_ids'])
        action_attention_mask.extend(sample['action_attention_mask'])
        sizes.append(sample['sizes'])
        labels.append(sample['labels'])
        images.append(sample['images'])
    max_state_len = max(sum(x) for x in state_attention_mask)
    max_action_len = max(sum(x) for x in action_attention_mask)
    return {
        'state_input_ids': torch.tensor(state_input_ids)[:, :max_state_len],
        'state_attention_mask': torch.tensor(state_attention_mask)[:, :max_state_len],
        'action_input_ids': torch.tensor(action_input_ids)[:, :max_action_len],
        'action_attention_mask': torch.tensor(action_attention_mask)[:, :max_action_len],
        'sizes': torch.tensor(sizes),
        'images': torch.tensor(images),
        'labels': torch.tensor(labels),
    }

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

def get_return_value(env, asin, options, search_terms, page_num, product):
    asin_url = None

    # Determine product URL + options based on environment
    if env == 'webshop':
        query_str = "+".join(search_terms.split())
        options_str = json.dumps(options)
        asin_url = (
            f'{WEBSHOP_URL}/item_page/{WEBSHOP_SESSION}/'
            f'{asin}/{query_str}/{page_num}/{options_str}'
        )
    else:
        asin_url = f"https://www.ebay.com/itm/{asin}" if env == 'ebay' else \
            f"https://www.amazon.com/dp/{asin}"
    
    # Extract relevant fields for product
    product_reduced = {k: v for k, v in product.items() if k in ["asin", "Title", "Description", "BulletPoints"]}
    product_reduced["Description"] = product_reduced["Description"][:100] + "..."
    product_reduced["Features"] = product_reduced.pop("BulletPoints")
    product_reduced["Features"] = product_reduced["Features"][:100] + "..."

    # Create HTML to show link to product
    html = """<!DOCTYPE html><html><head><title>Chosen Product</title></head><body>"""
    html += f"""Product Image:<img src="{product["MainImage"]}" height="50px" /><br>""" if len(product["MainImage"]) > 0 else ""
    html += f"""Link to Product:
        <a href="{asin_url}" style="color:blue;text-decoration:underline;" target="_blank">{asin_url}</a>
        </body></html>"""

    return product_reduced, options if len(options) > 0 else "None Selected", html
        

def predict(obs, info):
    """
    Given WebShop environment observation and info, predict an action.
    """
    valid_acts = info['valid']
    return "search[" + llm(process_goal(obs)) + "]"

def run_episode(goal, env, verbose=True):
    """
    Interact with amazon to find a product given input goal.
    Input: text goal
    Output: a url of found item on amazon.
    """
    env = env.lower()
    if env not in ENVIRONMENTS:
        print(f"[ERROR] Environment {env} not recognized")
        
    obs = "Amazon Shopping Game\nInstruction:" + goal + "\n[button] search [button]"
    info = {'valid': ['search[stuff]'], 'image_feat': torch.zeros(512)}
    product_map = {}
    title_to_asin_map = {}
    search_results_cache = {}
    visited_asins, clicked_options = set(), set()
    sub_page_type, page_type, page_num = None, None, None
    search_terms, prod_title, asin = None, None, None
    options = {}
    
    for i in range(10):
        # Run prediction
        action = predict(obs, info)
        
        # Previous Page Type, Action -> Next Page Type
        action_content = action[action.find("[")+1:action.find("]")]
        prev_page_type = page_type
        if action.startswith('search['):
            page_type = Page.RESULTS
            search_terms = action_content
            page_num = 1
        elif action.startswith('click['):
            if action.startswith('click[item -'):
                prod_title = action_content[len("item -"):].strip()
                found = False
                for key in title_to_asin_map:
                    if prod_title == key:
                        asin = title_to_asin_map[key]
                        page_type = Page.ITEM_PAGE
                        visited_asins.add(asin)
                        found = True
                        break
                if not found:
                    raise Exception("Product to click not found")
                    
            elif any(x.value in action for x in [Page.DESC, Page.FEATURES, Page.REVIEWS]):
                page_type = Page.SUB_PAGE
                sub_page_type = Page(action_content.lower())
                
            elif action == 'click[< prev]':
                if sub_page_type is not None:
                    page_type, sub_page_type = Page.ITEM_PAGE, None
                elif prev_page_type == Page.ITEM_PAGE:
                    page_type = Page.RESULTS
                    options, clicked_options = {}, set()
                elif prev_page_type == Page.RESULTS and page_num > 1:
                    page_type = Page.RESULTS
                    page_num -= 1
                    
            elif action == 'click[next >]':
                page_type = Page.RESULTS
                page_num += 1
                
            elif action.lower() == 'click[back to search]':
                page_type = Page.SEARCH
                
            elif action == 'click[buy now]':
                return get_return_value(env, asin, options, search_terms, page_num, product_map[asin])
            
            elif prev_page_type == Page.ITEM_PAGE:
                found = False
                for opt_name, opt_values in product_map[asin]["options"].items():
                    if action_content in opt_values:
                        options[opt_name] = action_content
                        page_type = Page.ITEM_PAGE
                        clicked_options.add(action_content)
                        found = True
                        break
                if not found:
                    raise Exception("Unrecognized action: " + action)
        else:
            raise Exception("Unrecognized action:" + action)
        
        if verbose:
            print(f"Parsing {page_type.value} page...")
        
        # URL -> Real HTML -> Dict of Info
        if page_type == Page.RESULTS:
            if search_terms in search_results_cache:
                data = search_results_cache[search_terms]
                if verbose:
                    print(f"Loading cached results page for \"{search_terms}\"")
            else:
                begin = time.time()
                if env == 'amazon':
                    data = parse_results_amz(search_terms, page_num, verbose)
                if env == 'webshop':
                    data = parse_results_ws(search_terms, page_num, verbose)
                if env == 'ebay':
                    data = parse_results_ebay(search_terms, page_num, verbose)
                end = time.time()
                if verbose:
                    print(f"Parsing search results took {end-begin} seconds")

                search_results_cache[search_terms] = data
                for d in data:
                    title_to_asin_map[d['Title']] = d['asin']
        elif page_type == Page.ITEM_PAGE or page_type == Page.SUB_PAGE:
            if asin in product_map:
                if verbose:
                    print("Loading cached item page for", asin)
                data = product_map[asin]
            else:
                begin = time.time()
                if env == 'amazon':
                    data = parse_item_page_amz(asin, verbose)
                if env == 'webshop':
                    data = parse_item_page_ws(asin, search_terms, page_num, options, verbose)
                if env == 'ebay':
                    data = parse_item_page_ebay(asin, verbose)
                end = time.time()
                if verbose:
                    print("Parsing item page took", end-begin, "seconds")
                product_map[asin] = data
        elif page_type == Page.SEARCH:
            if verbose:
                print("Executing search")
            obs = "Amazon Shopping Game\nInstruction:" + goal + "\n[button] search [button]"
            info = {'valid': ['search[stuff]'], 'image_feat': torch.zeros(512)}
            continue
        else:
            raise Exception("Page of type `", page_type, "` not found")

        # Dict of Info -> Fake HTML -> Text Observation
        begin = time.time()
        html_str = dict_to_fake_html(data, page_type, asin, sub_page_type, options, product_map, goal)
        obs = convert_html_to_text(html_str, simple=False, clicked_options=clicked_options, visited_asins=visited_asins)
        end = time.time()
        if verbose:
            print("[Page Info -> WebShop HTML -> Observation] took", end-begin, "seconds")

        # Dict of Info -> Valid Action State (Info)
        begin = time.time()
        prod_arg = product_map if page_type == Page.ITEM_PAGE else data
        info = convert_dict_to_actions(page_type, prod_arg, asin, page_num)
        end = time.time()
        if verbose:
            print("Extracting available actions took", end-begin, "seconds")
        
        if i == 50:
            return get_return_value(env, asin, options, search_terms, page_num, product_map[asin])
