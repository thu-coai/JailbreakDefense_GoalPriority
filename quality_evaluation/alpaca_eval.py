import json
import numpy as np
import re, random
import sys, os
sys.path.append('..')
from utils.gpt_api import eval_gpt
from utils.utils import add_normal_prompt, add_defense

def construct_generation_prompts(path, outpath, model_path, defense_type=None):
    with open(path) as f:
        data = json.load(f)
    
    res = []
    for i, d in enumerate(data):
        ins = d['instruction']
        if defense_type:
            prompt = add_defense(ins, defense_type)
            prompt = add_normal_prompt(prompt, model_path)
        else:
            prompt = add_normal_prompt(ins, model_path)
        outd = {'id': i, 'instruction': ins, 'prompt': prompt}
        res.append(outd)
        
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w') as outf:
        json.dump(res, outf, ensure_ascii=False, indent=2)
        
def construct_evaluation_prompt_gpt4(pred_path, ref_path, outpath, genfirst, limit=None, defense_type=None):
    with open(pred_path) as f:
        data1 = json.load(f)
    
    with open(ref_path) as f:
        data2 = json.load(f)
    
    l = len(data1)
    if limit is not None:
        l = min(l, limit)
    
    res = []
    
    random.seed(42)
    for i in range(l):
        d1 = data1[i]
        d2 = data2[i]
        assert d1['instruction'] == d2['instruction']
        instruction = d1['instruction']
        ipt = d1['input']
        pred_opt = d1['output']
        ref_opt = d2['output']
        
        try:
            pred_opt = re.findall(r'\[Final response\](.*)', pred_opt, flags=re.S)[0].strip()
        except:
            pred_opt = pred_opt
        
        options = [
            lambda: (pred_opt, ref_opt, 0),
            lambda: (ref_opt, pred_opt, 1)
        ]

        output_1, output_2, order = random.choice(options)()
        
        system_prompt = 'You are a helpful assistant, that ranks models by the quality of their answers.'
        prompt = '''I want you to create a leaderboard of different of large-language models. To do so, I will give you the instructions (prompts) given to the models, and the responses of two models. Please rank the models based on which responses would be preferred by humans. All inputs and outputs should be python dictionaries.

Here is the prompt:
{
    "instruction": """{instruction}""",
}

Here are the outputs of the models:
[
    {
        "model": "model_1",
        "answer": """{output_1}"""
    },
    {
        "model": "model_2",
        "answer": """{output_2}"""
    }
]

Now please rank the models by the quality of their answers, so that the model with rank 1 has the best output. Then return a list of the model names and ranks, i.e., produce the following output:
[
    {'model': <model-name>, 'rank': <model-rank>},
    {'model': <model-name>, 'rank': <model-rank>}
]

Your response must be a valid Python dictionary and should contain nothing else because we will directly execute it in Python. Please provide the ranking that the majority of humans would give.
        '''
        
        outd = {
            'instruction': instruction,
            'input': ipt,
            'pred_opt': pred_opt,
            'ref_opt': ref_opt,
            'pred_path': pred_path,
            'ref_path': ref_path,
            'prompt': prompt,
            'system_prompt': system_prompt,
            'order': order
        }
        res.append(outd)
    
    for i, d in enumerate(res):
        d['id'] = i
    
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w') as outf:
        json.dump(res, outf, ensure_ascii=False, indent=2)

def stat_winrate(path, outpath=None, eva_name='chatgpt'):
    data = []
    with open(path) as f:
        for line in f:
            w = json.loads(line)
            eva_pred = w['response']
            l =  w['data']
            order =  l['order']
            
            temp = l['pred_opt']
            del l['pred_opt']
            l['pred_opt'] = temp
            temp = l['ref_opt']
            del l['ref_opt']
            l['ref_opt'] = temp
            
            l['pred'] = eva_pred
            if eva_name == 'chatgpt':
                if '(a)' in eva_pred:
                    win = 1
                else:
                    win = 2
            elif eva_name == 'gpt4':
                try:
                    pred_dict = eval(eva_pred)
                    pred_rank = pred_dict[0]['rank']
                    model_name = pred_dict[0]['model']
                except:
                    pred_rank = 1
                    model_name = 'model_1'
                if model_name == 'model_1' and pred_rank == 1:
                    win = (1 if order == 1 else 2)
                elif model_name == 'model_2' and pred_rank == 1:
                    win = (2 if order == 1 else 1)
                elif model_name == 'model_1' and pred_rank == 2:
                    win = (2 if order == 1 else 1)
                else:
                    win = (1 if order == 1 else 2)
                
            l['win'] = win
            data.append(l)
            
            if 'sorry' in l['pred_opt']:
                print(l['pred_opt'])
    
    wins = np.array([d['win'] for d in data])
    print(f'pred_path:{data[0]["pred_path"]}\nref_path:{data[0]["ref_path"]}\npred model winrate:{np.mean(wins == 2)}')
    if outpath is not None:
        with open(outpath, 'w') as outf:
            json.dump(data, outf, ensure_ascii=False, indent=1)

def process_chatgpt_gen(path, outpath):
    res = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            outd = {
            "id": d["data"]["id"],
            "input": d["data"]["prompt"],
            'instruction': d['data']['instruction'],
            "output": d["response"]["choices"][0]["message"]["content"]
        }
            res.append(outd)
    
    res.sort(key=lambda x:x['id'])
    with open(outpath, 'w') as outf:
        json.dump(res, outf, ensure_ascii=False, indent=1)
    

if __name__ == '__main__':
    defense_type = 'priority_internal_thoughts_simplified_v23'
    model_path = "vicuna"
    defense_types = [defense_type]

    for defense_type in defense_types:
        path = './alpaca/gen_results/davinci003_output.json'

        if defense_type:
            outpath = f'./alpaca/gen_prompts/vicuna_13b_{defense_type}.json'
        else:
            outpath = './alpaca/gen_prompts/vicuna.json'
        
        construct_generation_prompts(path, outpath, model_path, defense_type=defense_type)
        
        if model_path == 'chatgpt':
            save_path = f'./alpaca/gen_results/chatgpt{"" if not defense_type else "_" + defense_type}_output_withsystem.jsonl'
            eval_gpt(outpath, save_path, model_name='gpt-3.5-turbo-0613', max_tokens=1024, top_p=0.1, temperature=0., limit=100, system_prompt='You are a helpful assistant.')
            process_chatgpt_gen(save_path, save_path[:-1])
            
        if model_path == 'gpt-4':
            save_path = f'./alpaca/gen_results/gpt-4{"" if not defense_type else "_" + defense_type}_output_withsystem.jsonl'
            eval_gpt(outpath, save_path, model_name='gpt-4', max_tokens=1024, top_p=0.1, temperature=0., limit=100, system_prompt='You are a helpful assistant.')
            process_chatgpt_gen(save_path, save_path[:-1])
        
        genfirst = False
        model_name = 'vicuna_7b'
        eva_name = 'gpt4'
        
        path1 = f'./alpaca/gen_results/{model_name}{"" if not defense_type else "_" + defense_type}_output.json'
        path2 = f'./alpaca/gen_results/davinci003_output1.json'
        
        outpath = f'./alpaca/eva_prompts/gpt4_{model_name}{"" if not defense_type else "_" + defense_type}_davinci003_genfirst{genfirst}.json'
        construct_evaluation_prompt_gpt4(path1, path2, outpath, genfirst=genfirst, limit=100, defense_type=defense_type)

        
        file_path = outpath
        save_path = f'./alpaca/eva_results/gpt4_{model_name}{"" if not defense_type else "_" + defense_type}_davinci003_gpt4_genfirst{genfirst}.jsonl'
        eval_gpt(file_path, save_path, model_name='gpt-4', max_tokens=100, top_p=0.1, temperature=0.)
        
        path = save_path
        outpath = save_path[:-1]
        stat_winrate(path, outpath, eva_name)
    