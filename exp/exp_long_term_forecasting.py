from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
from datetime import datetime
import pickle
warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.path = None
        self._scaler = None

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        scaler_savepath = os.path.join(self.path, 'scaler.save')
        if flag == 'train':
            data_set, data_loader = data_provider(self.args, flag)
            self._scaler = data_set.scaler
            f = open(scaler_savepath, 'wb')
            pickle.dump(self._scaler, f)
            f.close()
            print("scaler saved to {}".format(scaler_savepath))
        elif flag != 'train':
            f = open(scaler_savepath, 'rb')
            self._scaler = pickle.load(f)
            f.close()
            data_set, data_loader = data_provider(self.args, flag, self._scaler)
            print('replaced scaler with loaded scaler')
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

                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        # Set to inference mode for validation
                        if hasattr(self.model, 'set_teacher_forcing_mode'):
                            self.model.set_teacher_forcing_mode(False)
                        outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, tf_target=None)
                else:
                    # Set to inference mode for validation
                    if hasattr(self.model, 'set_teacher_forcing_mode'):
                        self.model.set_teacher_forcing_mode(False)
                    outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, tf_target=None)
                

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def _verify_teacher_forcing_alignment(self, vali_loader):
        """Verify that teacher forcing and inference modes produce similar results."""
        self.model.eval()
        with torch.no_grad():
            # Take first batch for comparison
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                
                # Teacher forcing mode
                if hasattr(self.model, 'set_teacher_forcing_mode'):
                    self.model.set_teacher_forcing_mode(True)
                tf_outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, tf_target=batch_y)
                
                # Inference mode  
                if hasattr(self.model, 'set_teacher_forcing_mode'):
                    self.model.set_teacher_forcing_mode(False)
                inf_outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, tf_target=None)
                
                # Compare results
                mse_diff = torch.nn.functional.mse_loss(tf_outputs, inf_outputs).item()
                print(f"[DEBUG TF Alignment] Teacher forcing vs Inference MSE diff: {mse_diff:.6f}")
                print(f"[DEBUG TF Alignment] TF range: [{tf_outputs.min():.4f}, {tf_outputs.max():.4f}], INF range: [{inf_outputs.min():.4f}, {inf_outputs.max():.4f}]")
                break  # Only check first batch
        self.model.train()

    def train(self, setting):
        

        path = os.path.join(self.args.checkpoints, setting)
        self.path = path
        if not os.path.exists(path):
            os.makedirs(path)
        
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            ampscaler = torch.cuda.amp.GradScaler()

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

                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        # Check if model supports teacher forcing mode
                        if hasattr(self.model, 'set_teacher_forcing_mode'):
                            self.model.set_teacher_forcing_mode(True)
                        outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, tf_target=batch_y)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    # Check if model supports teacher forcing mode
                    if hasattr(self.model, 'set_teacher_forcing_mode'):
                        self.model.set_teacher_forcing_mode(True)
                    outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, tf_target=batch_y)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    # DEBUG: Log gradient norms and data statistics
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float('inf'))
                    print(f"\t[DEBUG train] Gradient norm: {grad_norm:.6f}")
                    print(f"\t[DEBUG train] batch_x range: [{batch_x.min():.4f}, {batch_x.max():.4f}], mean: {batch_x.mean():.4f}")
                    print(f"\t[DEBUG train] batch_y range: [{batch_y.min():.4f}, {batch_y.max():.4f}], mean: {batch_y.mean():.4f}")
                    print(f"\t[DEBUG train] outputs range: [{outputs.min():.4f}, {outputs.max():.4f}], mean: {outputs.mean():.4f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    ampscaler.scale(loss).backward()
                    # Add gradient clipping for stability
                    ampscaler.unscale_(model_optim)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    ampscaler.step(model_optim)
                    ampscaler.update()
                else:
                    loss.backward()
                    # Add gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            # DEBUG: Verify teacher forcing alignment (once per epoch)
            if epoch % 1 == 0:  # Every epoch
                self._verify_teacher_forcing_alignment(vali_loader)
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):

        self.path = os.path.join(self.args.checkpoints, setting)
        test_data, test_loader = self._get_data(flag='test')
        print('(standalone testing) loading model...')
        if test==1:
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            print(self.model)

        preds = []
        trues = []
        test_loss = []
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
                
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        # Set to inference mode for testing
                        if hasattr(self.model, 'set_teacher_forcing_mode'):
                            self.model.set_teacher_forcing_mode(False)
                        outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, tf_target=None)
                else:
                    # Set to inference mode for testing
                    if hasattr(self.model, 'set_teacher_forcing_mode'):
                        self.model.set_teacher_forcing_mode(False)
                    outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, tf_target=None)

                loss = self._select_criterion()(outputs, batch_y)
                
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

                test_loss.append(loss.item())
        avg_loss = np.average(test_loss)
        print('test batch avg loss: {:.7f}'.format(avg_loss))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
