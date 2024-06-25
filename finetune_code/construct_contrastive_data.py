import json
import numpy as np
import os
from random import seed, random
import sys
sys.path.append('../')
from utils.utils import add_normal_prompt

def construct_prompt(queries, outpath, model_path):
    res = []
    
    for q in queries:
        prompt = add_normal_prompt(q, model_path)
        outd = {'query': q, 'prompt': prompt}
        res.append(outd)
    
    for i, d in enumerate(res):
        d['id'] = i
    
    with open(outpath, 'w') as outf:
        json.dump(res, outf, ensure_ascii=False, indent=2)
    
if __name__ == '__main__':
    
    with open('./data/advbench/harmful_behaviors_jailbreaknoleak_prompt.json') as f:
        data = json.load(f)
    
    queries = [d['query'] for d in data]
    
    outpath = './data/advbench/harmful_behaviors_jailbreaknoleak_vicuna_prompt.json'
    dirpath = os.path.dirname(outpath)
    os.makedirs(dirpath, exist_ok=True)
    
    model_path = 'lmsys/vicuna-13b-v1.3'
    
    construct_prompt(queries, outpath, model_path)
    