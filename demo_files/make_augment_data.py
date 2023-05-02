import os,json

chat_path = '/home/seungyoun/webshop/demo_files/webshop_demos_instruction_only_chat.json'
success_only_path = '/home/seungyoun/webshop/demo_files/webshop_demos_success_only.json'
new_path = '/home/seungyoun/webshop/demo_files/webshop_demos_chat_success_only.json'

with open(chat_path, 'r') as f:
    chat = json.load(f)


with open(success_only_path, 'r') as f:
    success_only = json.load(f)

newList = []
for i in success_only:
    instruction = i['instruction']
    output = i['output']

    instruction_only_words = instruction.split('Instruction:')[-1].strip()
    # print int red
    print("=====================================")
    print(f'\033[91m{instruction_only_words}\033[0m')

    # find instruction in chat
    for c in chat:
        #print(c['instruction'],' | ',  instruction_only_words)
        try:
            chat_instruction= c['instruction'].strip()
        except: 
            continue 
        if chat_instruction == instruction_only_words:
            chat_ = c["chat"].strip()
            chat_line = chat_.split('\n')

            instruction_line = f'Instruction:\n{chat_line[0]}'

            last_chat_str = '\n'.join(chat_line[1:])
            #print(f'Instruction:\n{instruction}')
            print(f'{instruction_line}')

            last_str = f'[Chat]\n\n{last_chat_str}\n\n{output}'
            # print in grey
            print(f'\033[90m[Chat]\n\n{last_chat_str}\n\n{output}\033[0m')

            newList.append({
                "instruction": instruction_line,
                "input" : "",
                "output": last_str
            })


# print len
print(f'len of success_only: {len(newList)}')

# SAVE
with open(new_path, 'w') as f:  
    json.dump(newList, f)