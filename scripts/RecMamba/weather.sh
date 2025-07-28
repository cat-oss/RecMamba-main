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

root_path_name=./data/weather
data_path_name=weather.csv
model_id_name=weather
data_name=custom

rin=1
random_seed=2024
one=96
two=192
three=336
four=720
residual=1
fc_drop=0.1
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
              n1=128
              n2=64
              fc_drop=0.1
              batch_size=256
              learning_rate=0.0005
              w_limit=0.4
          fi
          if [ $pred_len -eq $two ]
          then
              n1=256 #512
              n2=16
              fc_drop=0.5
              batch_size=256
              learning_rate=0.0005
              w_limit=0.1
          fi
          if [ $pred_len -eq $three ]
          then
              n1=128
              n2=32
              fc_drop=0.0
              batch_size=512
              learning_rate=0.001
              w_limit=0.5
          fi
          if [ $pred_len -eq $four ]
          then
              n1=128
              n2=32
              fc_drop=0.0
              batch_size=256
              learning_rate=0.0005
              w_limit=0.1
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
          --enc_in 21 \
          --n1 $n1 \
          --n2 $n2 \
          --dropout $fc_drop\
          --revin 1\
          --ch_ind 1\
          --residual $residual\
          --dconv $dconv \
          --d_state $dstate\
          --e_fact $e_fact\
          --des 'Exp' \
          --lradj 'constant'\
          --pct_start 0.2\
          --train_epochs 20\
          --adaptive_vali 0\
          --adaptive_train 1\
          --w_limit $w_limit\
          --w $w\
          --patience 4\
          --itr 1 \
          --batch_size $batch_size \
          --learning_rate $learning_rate \
          >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$n1'_'$n2'_'$fc_drop'_'$rin'_'$w_limit'_'$w.log

    done
done
