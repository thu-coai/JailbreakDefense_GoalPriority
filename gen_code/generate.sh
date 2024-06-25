# usage: replace the following variable with your local path and dataset
base_model=your_path
input_file=input_path
output_file=output_path

CUDA_VISIBLE_DEVICES=0 python generate.py --base_model ${base_model} --input_file ${input_file} --output_file ${output_file} --limit 100