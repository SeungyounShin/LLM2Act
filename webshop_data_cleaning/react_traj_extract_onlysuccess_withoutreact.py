import json,random,re

def remove_think_actions(text):
    pattern = r"Action: think\[.*?\]\nObservation: .*\n"
    result = re.sub(pattern, "", text)
    return result

with open("WebShop.ipynb", "r") as f:
    notebook_json = json.load(f)

gpt3_demo = notebook_json['cells'][-1]['outputs']

full_str = ''

for i in range(len(gpt3_demo)):
    gpt3_demo_cell = gpt3_demo[i]['text']

    partial_txt = ''.join(gpt3_demo_cell)

    full_str += partial_txt

score_str = 'Your score (min 0.0, max 1.0)'
start_str = '\nWebshop'
demos   = ['\n'.join(i.strip().split('\n')).split(score_str) for i in full_str.split('-----------------') if len(i) > 0]
for demo in demos:
    # traj extraction
    demo[0] = demo[0].split(start_str)[-1].strip()

    #print(demo[0].split(start_str))
    #exit()

    # reward extraction
    if len(demo) == 1:
        demo.append(0.0)
        demo[0] = '\n'.join(demo[0].split('\n')[:-2]).strip()
    else:
        demo[1] = float(demo[1].split('\n')[0].split(':')[-1])
        if demo[1] == 0.0:
            demo[0] = '\n'.join(demo[0].split('\n')[:-2]).strip()


# make json file for webshop

'''
[{
    "traj": demos[idx][0],
    "score": demos[idx][1]
},...]
'''

demos_new = list()
for idx in range(len(demos)):
    if demos[idx][1] < 1.0:
        continue
    full_traj = demos[idx][0]
    instruction = '\n'.join(full_traj.split('\n')[:2])

    trajectory = '\n'.join(full_traj.split('\n')[2:])
    trajectory = remove_think_actions(trajectory)
    trajectory = trajectory.replace('\n\n', '\n')

    # print in green highlight
    print('\033[92m' + instruction + '\033[0m') 
    # print in gray
    print('\033[90m' + trajectory + '\033[0m')
    print(f'----score : {demos[idx][1]}-----')


    demos_new.append({
        "instruction": instruction,
        "input": "",
        "output": trajectory,
    })

print(f'len(demos_new) : {len(demos_new)}')

with open('webshop_demos_success_only_without_react.json', 'w') as f:
    json.dump(demos_new, f)



