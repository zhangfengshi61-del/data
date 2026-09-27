python ./tokenizer/main.py \
  --device cuda:3 \
  --data_path ./data/Instruments/Instruments.all-emb-llama.npy \
  --ckpt_dir ./checkpoint/ \
  --wandb_project tokenize_cofirec \
