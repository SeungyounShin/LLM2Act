import json

file_path = "/home/seungyoun/LLM2Act/demo_files/selfplay_dense_reward_pred.json"

with open(file_path, "r") as json_file:
    data = json.load(json_file)

over = 0
new_list = list()
for i in data:
    traj= i['traj']

    instruction = traj.split("Action:")[0]
    remaining = "[Search]\n\n"+ traj[len(instruction) : ]
    if len(remaining) > 2400:
        #print(remaining)
        over += 1
        print('============')
        print(f"over 2400!!")
    remaining = remaining[:2400]
    
    new_list.append({
        'instruction' : instruction,
        'input' : '',
        'output' : remaining
    })

with open('/home/seungyoun/LLM2Act/demo_files/selfplay_dense_reward_pred_for_sft_training.json', 'w') as f:
    json.dump(new_list, f)