export CUDA_VISIBLE_DEVICES=3

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_192 \
  --model cnn \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 512 \
  --n_heads 8 \
  --d_layers 2 \
  --factor 3 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 50 \
  --batch_size 32 \
  --patience 10