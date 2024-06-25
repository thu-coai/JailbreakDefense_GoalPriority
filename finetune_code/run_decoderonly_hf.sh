data_name=vicuna_format_10kultrafeedbackgpt4consistentthoughtfixed_500advbenchJailbreaknoleakchatgptresp_contrastive4
data_dir=../data/finetune_data/${data_name}
# model_name=Llama-2-7b-hf
model_name=Llama-2-13b-hf
max_epoch=2

deepspeed --include localhost:0,1,2,3 --master_port=20953 \
    train_decoderonly_hf.py --ds_config=ds_config_hf.json \
    --train_path=${data_dir}/train.json \
    --valid_path=${data_dir}/dev.json \
    --model_dir=meta-llama/${model_name} \
    --pretrained_model_path= \
    --batch_size=8 --val_batch_size=8 \
    --gradient_accumulation=1 \
    --incontext_learn=0 \
    --savedmodel_path=./hf_save/${data_name}/seed12_${model_name}_bs32_warm0_lineardecay_lr2e-5_maxlen2048_max2 --ckpt_file='' \
    --max_epochs=${max_epoch} --warmup_steps=0 --warmup_ratio=0 \
    --learning_rate=2e-5 --fp16= \
    --seed=12 \
    --max_length=2048 --eval_step=75 --save_step=75 \
    --lr_decay=linear --patience=1 \
    --ema=0 --ema_start_epoch=0
