import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]='4'
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time Series Forecasting')

    # RANDOM SEED
    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

    # BASIC CONFIG
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [RecMamba]')
    parser.add_argument('--model_id_name', type=str, required=False, default='custom', help='model id name')

    # DATALOADER
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # FORECASTING TASK
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--n1',type=int,default=512, help='First Embedded representation')
    parser.add_argument('--n2',type=int,default=512, help='Second Embedded representation')
    parser.add_argument('--patch_size', type=int, default=16, help='atch Size')
    parser.add_argument('--stride', type=int, default=8, help='Patch Stride')
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--patch_sizes', type=str, default='8,16,32', help='multi-scale patch sizes, comma separated, e.g., 8,16,32')
    parser.add_argument('--strides', type=str, default='4,8,16', help='multi-scale patch strides, comma separated, e.g., 4,8,16')

    # METHOD
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--ch_ind', type=int, default=1, help='Channel Independence; True 1 False 0')
    parser.add_argument('--zip_flag', type=int, default=0, help='Channel Zip; True 1 False 0')
    parser.add_argument('--zip_size', type=int, default=32, help='Channel Zip Independence')
    parser.add_argument('--residual', type=int, default=1, help='Residual Connection; True 1 False 0')
    parser.add_argument('--d_state', type=int, default=256, help='d_state parameter of Mamba')
    parser.add_argument('--dconv', type=int, default=2, help='d_conv parameter of Mamba')
    parser.add_argument('--e_fact', type=int, default=1, help='expand factor parameter of Mamba')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') #Use this hyperparameter as the number of channels
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    
    # OPTIMIZATION
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=40, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--adaptive_train', type=int, default=0, help='adaptive weight of loss fuction in train')
    parser.add_argument('--adaptive_vali', type=int, default=0, help='adaptive weight of loss fuction in vali')
    parser.add_argument('--w', type=float, default=0.4, help='weight of reconstruction loss fuction')
    parser.add_argument('--w_limit', type=float, default=0.2, help='limit of weight of reconstruction loss fuction')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    args = parser.parse_args()

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    args.model_id_name=args.data_path[:-4]


    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            # setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_n1{}_n2{}_dr{}_cin{}_rin{}_res{}_dst{}_dconv{}_efact{}'.format(
            #     args.model_id,
            #     args.model,
            #     args.model_id_name,
            #     args.features,
            #     args.seq_len,
            #     args.label_len,
            #     args.pred_len,
            #     args.n1,
            #     args.n2,
            #     args.dropout,
            #     args.ch_ind,
            #     args.revin,
            #     args.residual,
            #     args.d_state,
            #     args.dconv,
            #     args.e_fact)
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_n1{}_n2{}_dr{}_rin{}_at{}_av{}_w{}_wl{}'.format(
                args.model_id,
                args.model,
                args.model_id_name,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.n1,
                args.n2,
                args.dropout,
                args.revin,
                args.adaptive_train,
                args.adaptive_vali,
                args.w,
                args.w_limit
                )

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_n1{}_n2{}_dr{}_cin{}_rin{}_res{}_dst{}_dconv{}_efact'.format(
                args.model_id,
                args.model,
                args.model_id_name,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.n1,
                args.n2,
                args.dropout,
                args.ch_ind,
                args.revin,
                args.residual,
                args.d_state,
                args.dconv,
                args.e_fact)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
        
