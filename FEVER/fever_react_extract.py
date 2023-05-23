import json,random 

with open("FEVER.ipynb", "r") as f:
    notebook_json = json.load(f)

gpt3_demo = notebook_json['cells'][-1]['outputs']

full_str = ''

for i in range(len(gpt3_demo)):
    gpt3_demo_cell = gpt3_demo[i]['text']

    partial_txt = ''.join(gpt3_demo_cell)

    full_str += partial_txt


score_str = '{\'steps\':'
start_str = 'Claim:'
demos   = ['\n'.join(i.strip().split('\n')).split(score_str) for i in full_str.split('-----------') if len(i) > 0]
for demo in demos:
    # traj extraction
    demo[0] = demo[0].split(start_str)[-1].strip()
    claim_ = demo[0].split('\n')[0]
    instruction = f'Claim: {claim_}'
    traj = '\n'.join(demo[0].split('\n')[1:])
    traj = traj.split(', reward =')[0].strip()
    print('======')
    print(instruction)
    print('----')
    print(traj)
    print('----')
    try:
        reward = float(demo[1].split('reward\':')[-1].split(',')[0].strip())
    except:
        continue
    print(reward)

exit()
# make json file for webshop

'''
[{
    "traj": demos[idx][0],
    "score": demos[idx][1]
},...]
'''

# make json format
demos = [{
    "traj": demos[idx][0],
    "score": demos[idx][1]
} for idx in range(len(demos))]


with open('webshop_demos.json', 'w') as f:
    json.dump(demos, f)

# load to check 
with open('webshop_demos.json', 'r') as f:
    demos = json.load(f)

print('===demo===')
idx = random.randint(0, len(demos)-1)
print(demos[idx]['traj'])
# print in red if not 1 
if demos[idx]['score'] < 0.5:
    print('\033[91m' + str(demos[idx]['score']) + '\033[0m')
elif demos[idx]['score'] < 1:
    # print in blue
    print('\033[93m' + str(demos[idx]['score']) + '\033[0m')
else:
    # print in green 
    print('\033[92m' + str(demos[idx]['score']) + '\033[0m')



