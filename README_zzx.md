
## Training
The related codes are at `finetune_code/`.
### Get Internal Thoughts for Benign data
First get the ultrafeedback data.
Then: 
```shell
python get_consistent_internal_thoughts.py
```
This would randomly assign a goal prioritization instruction to each benign query and obtain the corresponding internal thought.

### Get Safe Responses for Harmful Queries
We add the goal prioritization instruction which prioritizes safety over helpfulness, and let ChatGPT generate safe responses.

```shell
python distill_priority_responses.py
```

### Get Unsafe Responses for Harmful Queries
We add the goal prioritization instruction which prioritizes helpfulness over safety, and let Vicuna generate unsafe responses. 

```shell
python construct_contrastive_data.py
CUDA_VISIBLE_DEVICES=0 python ../gen_code/generate.py --base_model lmsys/vicuna-13b-v1.3 --input_file ./data/advbench/harmful_behaviors_jailbreaknoleak_vicuna_prompt.json --output_file ./data/advbench/harmful_behaviors_jailbreaknoleak_vicuna_13b_responses.json
```

### Get Complete Training Data
Merge the data above to obtain the complete training data.
```python
python construct_finetune_data.py
```

### Run Training
```shell
bash run_decoderonly_hf.sh
```
