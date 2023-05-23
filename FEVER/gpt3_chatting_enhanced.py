import os
import openai
import json
 
openai.api_key = os.environ["OPENAI_API_KEY"]

def llm(prompt, stop=["\n"]):
    response = openai.Completion.create(
      model="text-davinci-003",
      prompt=prompt,
      temperature=0.7,
      max_tokens=256,
      top_p=0.9,
      frequency_penalty=0.0,
      presence_penalty=0.0,
      stop=["END_CHAT"]
    )
    return response["choices"][0]["text"]

# get insturction set
llama_traj_path = '/home/seungyoun/LLM2Act/FEVER/history/llama_react_fewshot.json'
fever_prompt_path = '/home/seungyoun/LLM2Act/FEVER/prompts/fever.json'

with open(llama_traj_path, 'r') as f:
    llama_trajs = json.load(f)

with open(fever_prompt_path, 'r') as f:
    fever_trajs = json.load(f)['webact_simple3']

'''claims = list()
for traj in llama_trajs:
    claim = traj.split('Claim:')[-1].split('\n')[0].strip()
    claims.append(claim)

    claim_start_index = traj.index(claim)
    #print("----")
    #print(traj[claim_start_index+len(claim):])'''

claims = [
    'Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.',
    'Stranger Things is set in Bloomington, Indiana.',
    'Beautiful reached number two on the Billboard Hot 100 in 2003.?'
]

fewshot = """----given----
Claim: Coster-Waldau worked with the Fox Broadcasting Company.
----generate----
User : I think he worked with the Fox Broadcasting Company.
Robot : Who is he?
User : Coster-Waldau
Robot: Alright! I'll check Coster-Waldau worked with the Fox Broadcasting Company.
END_CHAT
"""

results = list()
for claim in claims:
  prompt = f'{fewshot}----given----\nClaim: {claim}\n----generate----\nUser :'

  out_str = llm(prompt)
  # print in green
  print('\033[92m' + f'Claim: {claim}\n' +'\033[0m')
  print('User:' + out_str)
  print('------------------')

  results.append({'instruction': claim, 'chat': out_str.strip()})

exit()
with open('/home/seungyoun/LLM2Act/FEVER/history/fever_chat_dict.json', 'w') as f:
    json.dump(results, f)


