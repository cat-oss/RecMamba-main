from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import RecMamba
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 
import pandas as pd
import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'RecMamba':RecMamba
                   }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)        
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Rec' in self.args.model:
                            outputs, x_origin, x_rec, loss_weight = self.model(batch_x)
                        
                else:
                    if 'Rec' in self.args.model:
                        outputs, x_origin, x_rec, loss_weight = self.model(batch_x)
                    
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                x_origin = x_origin[:, -self.args.seq_len:, f_dim:]
                x_rec = x_rec[:, -self.args.seq_len:, f_dim:]

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                x_pred = x_rec.detach().cpu()
                x_true = x_origin.detach().cpu()
                loss_weight = loss_weight.detach().cpu()

                if self.args.adaptive_vali:
                    loss = (1 - loss_weight) * criterion(pred, true) + loss_weight * criterion(x_pred, x_true)
                    print(criterion(pred, true))
                    # print("pre_weight: {0:.4f} | rec_weight: {1:.4f}".format(1 - loss_weight, loss_weight))
                else:
                    loss = (1 - self.args.w) * criterion(pred, true) + self.args.w * criterion(x_pred, x_true)
                    print(criterion(pred, true))
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Rec' in self.args.model:
                            outputs, x_origin, x_rec, loss_weight = self.model(batch_x)
                        

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        x_origin = x_origin[:, -self.args.seq_len:, f_dim:]
                        x_rec = x_rec[:, -self.args.seq_len:, f_dim:]
                        if self.args.adaptive_train:
                            # print("pre_weight: {0:.4f} | rec_weight: {1:.4f}".format(1 - loss_weight, loss_weight))
                            loss = (1 - loss_weight) * criterion(outputs, batch_y) + loss_weight * criterion(x_rec,
                                                                                                             x_origin)
                        else:
                            loss = (1 - self.args.w) * criterion(outputs, batch_y) + self.args.w * criterion(x_rec,
                                                                                                             x_origin)
                        train_loss.append(loss.item())
                else:
                    if 'Rec' in self.args.model:
                            outputs, x_origin, x_rec, loss_weight = self.model(batch_x)
                            
                    
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    x_origin = x_origin[:, -self.args.seq_len:, f_dim:]
                    x_rec = x_rec[:, -self.args.seq_len:, f_dim:]
                    if self.args.adaptive_train:
                        # print("pre_weight: {0:.4f} | rec_weight: {1:.4f}".format(1 - loss_weight, loss_weight))
                        loss = (1 - loss_weight) * criterion(outputs, batch_y) + loss_weight * criterion(x_rec,
                                                                                                         x_origin)
                    else:
                        loss = (1 - self.args.w) * criterion(outputs, batch_y) + self.args.w * criterion(x_rec,
                                                                                                         x_origin)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Rec' in self.args.model:
                            outputs, _, _, _ = self.model(batch_x)
                        
                else:
                    if 'Rec' in self.args.model:
                            outputs, _, _, _ = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    prd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, prd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()
        temp_df = pd.DataFrame()
        temp_df['Seed']=[self.args.random_seed]
        temp_df['Model']=[self.args.model]
        temp_df['seq_len']=[self.args.seq_len]
        temp_df['label_len']=[self.args.label_len]
        temp_df['pred_len']=[self.args.pred_len]
        temp_df['n1']=[self.args.n1]
        temp_df['n2']=[self.args.n2]
        temp_df['dropout']=[self.args.dropout]
        temp_df['train_epochs']=[self.args.train_epochs]
        temp_df['batch']=[self.args.batch_size]
        temp_df['patience']=[self.args.patience]
        temp_df['LR']=[self.args.learning_rate]
        temp_df['dropout']=[self.args.dropout]
        temp_df['ch_ind']=[self.args.ch_ind]
        temp_df['revin']=[self.args.revin]
        temp_df['e_fact']=[self.args.e_fact]
        temp_df['dconv']=[self.args.dconv]
        temp_df['adaptive_train'] = [self.args.adaptive_train]
        temp_df['adaptive_vali'] = [self.args.adaptive_vali]
        temp_df['w'] = [self.args.w]
        temp_df['w_limit'] = [self.args.w_limit]

        temp_df['MSE']=[mse]
        temp_df['MAE']=[mae]
        temp_df['residual']=[self.args.residual]
        temp_df['d_state']=[self.args.d_state]

        temp_df['checkpoint_path']=[setting]

        if not os.path.exists('./csv_results/'+'result_'+self.args.data_path):
            temp_df.to_csv('./csv_results/'+'result_'+self.args.data_path, index=False)
        else:
            result_df=pd.read_csv('./csv_results/'+'result_'+self.args.data_path)
            result_df = pd.concat([result_df,temp_df],ignore_index=True)
            result_df.to_csv('./csv_results/'+'result_'+self.args.data_path, index=False)

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Rec' in self.args.model:
                            outputs = self.model(batch_x)
                       
                else:
                    if 'Rec' in self.args.model:
                        outputs = self.model(batch_x)
                    
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
