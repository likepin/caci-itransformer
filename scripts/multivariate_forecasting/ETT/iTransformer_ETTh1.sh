# export CUDA_VISIBLE_DEVICES=1

# model_name=iTransformer

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_96_96 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 256 \
#   --d_ff 256 \
#   --itr 1

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_96_192 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len 96 \
#   --pred_len 192 \
#   --e_layers 2 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 256 \
#   --d_ff 256 \
#   --itr 1

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_96_336 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len 96 \
#   --pred_len 336 \
#   --e_layers 2 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --itr 1

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_96_720 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len 96 \
#   --pred_len 720 \
#   --e_layers 2 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --itr 1


# model_name=iTransformer

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_96_96 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 256 \
#   --d_ff 256 \
#   --itr 3 \
#   --use_gpu True \
#   --gpu 0 \
#   --num_workers 0 \
#   --train_epochs 10 \
#   --patience 3


model_name=iTransformer

for pred in 96 192 336 720
do
  # keep the original script's model size choice
  if [ "$pred" -le 192 ]; then
    d_model=256
    d_ff=256
  else
    d_model=512
    d_ff=512
  fi

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_96_${pred} \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len 96 \
    --pred_len ${pred} \
    --e_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model ${d_model} \
    --d_ff ${d_ff} \
    --itr 3 \
    --use_gpu True \
    --gpu 0 \
    --num_workers 0 \
    --train_epochs 10 \
    --patience 3
done
