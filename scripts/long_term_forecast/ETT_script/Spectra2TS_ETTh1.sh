export CUDA_VISIBLE_DEVICES=1

model_name=Spectra2TS

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model $model_name \
  --data ETTh1 \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 1 \
  --d_model 768 \
  --n_heads 8 \
  --d_layers 3 \
  --d_ff 1024 \
  --dropout 0.1 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 50 \
  --batch_size 32 \
  --learning_rate 1e-4

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_192 \
  --model $model_name \
  --data ETTh1 \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 1 \
  --d_model 768 \
  --n_heads 8 \
  --d_layers 3 \
  --d_ff 1024 \
  --dropout 0.1 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 50 \
  --batch_size 32 \
  --learning_rate 1e-4

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_336 \
  --model $model_name \
  --data ETTh1 \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 1 \
  --d_model 768 \
  --n_heads 8 \
  --d_layers 3 \
  --d_ff 1024 \
  --dropout 0.1 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 50 \
  --batch_size 32 \
  --learning_rate 1e-4

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_720 \
  --model $model_name \
  --data ETTh1 \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 1 \
  --d_model 768 \
  --n_heads 8 \
  --d_layers 3 \
  --d_ff 1024 \
  --dropout 0.1 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 50 \
  --batch_size 32 \
  --learning_rate 1e-4