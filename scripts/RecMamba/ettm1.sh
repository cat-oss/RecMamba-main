if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
if [ ! -d "./csv_results" ]; then
    mkdir ./csv_results
fi
if [ ! -d "./results" ]; then
    mkdir ./results
fi
if [ ! -d "./test_results" ]; then
    mkdir ./test_results
fi
model_name=RecMamba

root_path_name=./data/ETT-small
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1


rin=1
random_seed=2024
one=96
two=192
three=336
four=720
residual=1
fc_drop=0.6
dstate=256
dconv=2
e_fact=1
w=0.4
for seq_len in 96
do
    for pred_len in 96 192 336 720
    do
          if [ $pred_len -eq $one ]
          then
              n1=256
              n2=16
              batch_size=256
              learning_rate=0.0005
              adaptive_train=0
              w_limit=0.2
          fi
          if [ $pred_len -eq $two ]
          then
              n1=256
              n2=128
              batch_size=1024
              learning_rate=0.001
              adaptive_train=0
              w_limit=0.2
          fi
          if [ $pred_len -eq $three ]
          then
              n1=256
              n2=128
              batch_size=1024
              learning_rate=0.001
              adaptive_train=1
              w_limit=0.3
          fi
          if [ $pred_len -eq $four ]
          then
              n1=512
              n2=64
              batch_size=512
              learning_rate=0.001
              adaptive_train=1
              w_limit=0.5
          fi

          python -u run_longExp.py \
          --random_seed $random_seed \
          --is_training 1 \
          --root_path $root_path_name \
          --data_path $data_path_name \
          --model_id $model_id_name_$seq_len'_'$pred_len \
          --model $model_name \
          --data $data_name \
          --features M \
          --seq_len $seq_len \
          --pred_len $pred_len \
          --enc_in 7 \
          --n1 $n1 \
          --n2 $n2 \
          --dropout $fc_drop\
          --des 'Exp' \
          --train_epochs 100\
          --adaptive_train $adaptive_train\
          --w_limit $w_limit\
          --w $w\
          --itr 1 \
          --batch_size $batch_size \
          --learning_rate $learning_rate \
          >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$n1'_'$n2'_'$fc_drop'_'$rin'_'$w_limit'_'$w.log
    done
done