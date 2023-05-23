import os
import json

# get insturction set
llama_traj_path = '/home/seungyoun/LLM2Act/FEVER/history/llama_react_fewshot.json'
with open(llama_traj_path, 'r') as f:
    llama_trajs = json.load(f)

only_success = list()
for traj in llama_trajs:
    
    claim = traj.split('Claim:')[-1].split('\n')[0].strip()

    claim_start_index = traj.index(claim)
    remaining = traj[claim_start_index+len(claim):].strip()

    reward = 0
    if 'reward = 1' in traj:
        reward = 1

        only_success.append({
            'instruction' : f'Claim: {claim}\n',
            'input' :'',
            'output' : remaining[:-len(', reward = 1')]
        })

# save 
output_file_path = './fever_llama_only_success.json'

with open(output_file_path, 'w') as f:
    json.dump(only_success, f)
    
print('saved.')
    