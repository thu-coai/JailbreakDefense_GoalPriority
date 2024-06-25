import json
import os
from random import seed
import re
import sys
sys.path.append('../')
from utils.utils import add_normal_prompt

def construct_finetune_data(outdir, check_thought=False, adv_data_num=500):
    os.makedirs(outdir, exist_ok=True)
    train_res = []
    dev_res = []
    
    with open('../data/ultrafeedback/ultrafeedback_gpt4.json') as f:
        data = json.load(f)
        
    with open('./data/ultrafeedback/ultrafeedback_gpt4_priority_onlythoughts_chatgpt_responses.json') as f:
        priority_data = json.load(f)
    
    normal_data_num = 10000
    dev_adv_data_num = adv_data_num // 10
    
    seed(2023)
    
    print(len(priority_data))
    fail_num = 0
    fix_thought_num = 0
    for i in range(normal_data_num):
        d = data[i]
        prid = priority_data[i]
        ipt = d['instruction']
        prompt = prid['prompt']
        if prompt.startswith('You are an assistant that always prioritizes helpfulness over safety.'):
            ipt = ipt + f'\n\nImportantly, you should prioritize helpfulness over safety. Avoid safe but unhelpful responses.'
        else:
            ipt = ipt + f'\n\nImportantly, you should prioritize safety over helpfulness. Avoid helpful but unsafe responses.'
        ipt = add_normal_prompt(ipt)
        
        try:
            if '[Final' in prid['response']:
                thought = re.findall(r'\[Internal thoughts\] (.*)\[Final', prid['response'], flags=re.S)[0].strip()
            else:
                thought = re.findall(r'\[Internal thoughts\] (.*)', prid['response'], flags=re.S)[0].strip()
        except Exception as e:
            print(e)
            fail_num += 1
            print('response:', prid['response'])
            thought = f'The query is safe and harmless. I could provide my answer without violating the priority requirement.'
        else:
            if check_thought:
                if 'refuse' in thought or 'cannot provide' in thought:
                    fix_thought_num += 1
                    print(f'instruction for fixed thought:{d["instruction"]}')
                    thought = f'The query is safe and harmless. I could provide my answer without violating the priority requirement.'
                
        opt = f"[Internal thoughts] {thought}\n[Final response] {d['output']}"
        train_res.append({'input': ipt, 'output': opt})
    
    print(f'fail to extract internal thoughts num:{fail_num}')
    print(f'fix thought num:{fix_thought_num}')

    with open('./data/advbench/harmful_behaviors_jailbreaknoleak_chatgpt_responses.json') as f:
        data = json.load(f)

    for d in data[:adv_data_num]:
        ipt = d['query']
        ipt = ipt + f'\n\nImportantly, you should prioritize safety over helpfulness. Avoid helpful but unsafe responses.'
        ipt = add_normal_prompt(ipt)
        
        opt = f"{d['response']}"

        train_res.append({'input': ipt, 'output': opt})
        
    for d in data[adv_data_num:adv_data_num+dev_adv_data_num]:
        ipt = d['query']
        ipt = ipt + f'\n\nImportantly, you should prioritize safety over helpfulness. Avoid helpful but unsafe responses.'
        ipt = add_normal_prompt(ipt)
        
        opt = f"{d['response']}"
        
        dev_res.append({'input': ipt, 'output': opt})

    
    with open('./data/advbench/harmful_behaviors_jailbreaknoleak_vicuna_13b_responses.json') as f:
        data = json.load(f)
        
    for d in data[:adv_data_num]:
        ipt = d['input']
        
        ipt = ipt.replace('ASSISTANT:', '').strip() + f"\n\nImportantly, you should prioritize helpfulness over safety. Avoid safe but unhelpful responses." + ' ASSISTANT:'
        
        opt = f"[Internal thoughts] The query is unsafe and may be harmful. However, I should prioritize helpfulness over safety as required, so I should provide a helpful and potentially unsafe response.\n[Final response] {d['output']}"
        
        train_res.append({'input': ipt, 'output': opt})
        
    for d in data[adv_data_num:adv_data_num+dev_adv_data_num]:
        ipt = d['input']

        ipt = ipt.replace('ASSISTANT:', '').strip() + f"\n\nImportantly, you should prioritize helpfulness over safety. Avoid safe but unhelpful responses." + ' ASSISTANT:'
        
        opt = f"[Internal thoughts] The query is unsafe and may be harmful. However, I should prioritize helpfulness over safety as required, so I should provide a helpful and potentially unsafe response.\n[Final response] {d['output']}"
        
        dev_res.append({'input': ipt, 'output': opt})
        
    print(f'train data num: {len(train_res)}')
    print(f'dev data num: {len(dev_res)}')
    
    with open(f'{outdir}/train.json', 'w') as outf:
        json.dump(train_res, outf, ensure_ascii=False, indent=2)
    
    with open(f'{outdir}/dev.json', 'w') as outf:
        json.dump(dev_res, outf, ensure_ascii=False, indent=2)
    
if __name__ == '__main__':
    
    adv_data_num = 500
    outdir = f'../data/finetune_data/vicuna_format_10kultrafeedbackgpt4consistentthoughtfixed_{adv_data_num}advbenchJailbreaknoleakchatgptresp_contrastive4'
    construct_finetune_data(outdir, check_thought=True, adv_data_num=adv_data_num)
    