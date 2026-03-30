from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

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
        criterion = nn.MSELoss(reduction='none')
        return criterion

    @staticmethod
    def _unpack_batch(batch):
        if len(batch) == 9:
            return batch
        if len(batch) == 7:
            batch_x, batch_y, batch_x_mark, batch_y_mark, batch_gating, batch_regime_x_aux, batch_regime_y_aux = batch
            return batch_x, batch_y, batch_x_mark, batch_y_mark, batch_gating, batch_regime_x_aux, batch_regime_y_aux, None, None
        if len(batch) == 6:
            batch_x, batch_y, batch_x_mark, batch_y_mark, batch_phase_d_lambda, batch_phase_d_delta = batch
            return batch_x, batch_y, batch_x_mark, batch_y_mark, None, None, None, batch_phase_d_lambda, batch_phase_d_delta
        if len(batch) == 5:
            batch_x, batch_y, batch_x_mark, batch_y_mark, batch_gating = batch
            return batch_x, batch_y, batch_x_mark, batch_y_mark, batch_gating, None, None, None, None
        if len(batch) == 4:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            return batch_x, batch_y, batch_x_mark, batch_y_mark, None, None, None, None, None
        raise ValueError(f'Unexpected batch structure with length {len(batch)}')

    def _phasec_gating_active(self):
        return self.args.data == 'phasec_synth' and self.args.phasec_gating_mode != 'none'

    def _phasec_should_log(self):
        return self._phasec_gating_active()

    def _phase_d_active(self):
        return bool(getattr(self.args, 'phase_d_enable', False))

    def _log_phasec_gating_config(self):
        if not self._phasec_should_log():
            return
        polarity_note = {
            'inverse': 'preserve suppressive gating semantics (weight down when lambda up)',
            'direct': 'emphasize-high-lambda hypothesis (weight up when lambda up)',
        }[self.args.phasec_gating_weight_polarity]
        print('PhaseC gating config:')
        print(f'  mode={self.args.phasec_gating_mode}')
        print(f'  polarity={self.args.phasec_gating_weight_polarity} ({polarity_note})')
        print(f'  gating_artifact_path={self.args.phasec_gating_lambda_path}')
        print(f'  gating_hash={self.args.phasec_gating_lambda_hash or "(unspecified)"}')
        print(f'  weight_normalization=batch_mean_1_then_alpha_shrink(alpha={self.args.phasec_gating_alpha})')

    def _log_phase_d_config(self):
        if not self._phase_d_active():
            return
        print('PhaseD graph-guided config:')
        print(f'  interface_dir={self.args.phase_d_interface_dir}')
        print(f'  use_static_bias={self.args.phase_d_use_static_bias}')
        print(f'  use_dynamic_bias={self.args.phase_d_use_dynamic_bias}')
        print(f'  use_lambda_gate={self.args.phase_d_use_lambda_gate}')
        print(f'  shuffle_lambda={self.args.phase_d_shuffle_lambda}')
        print(f'  eval_use_static_bias={self.args.phase_d_eval_use_static_bias}')
        print(f'  beta_static={self.args.phase_d_beta_static}')
        print(f'  beta_dynamic={self.args.phase_d_beta_dynamic}')

    def _compute_phasec_sample_weights(self, batch_gating, num_samples, device):
        base = torch.ones(num_samples, device=device)
        if not self._phasec_gating_active():
            return base
        if batch_gating is None:
            raise ValueError('Phase C gating mode is active but the dataset did not return gating values')
        gating_future = batch_gating.float().to(device)
        lambda_mean = gating_future.mean(dim=1)
        if self.args.phasec_gating_mode == 'noop':
            return base
        if self.args.phasec_gating_mode != 'loss_weighting':
            raise ValueError(f'Unsupported phasec_gating_mode: {self.args.phasec_gating_mode}')
        if self.args.phasec_gating_weight_polarity == 'inverse':
            weights_raw = 1.0 - lambda_mean
        elif self.args.phasec_gating_weight_polarity == 'direct':
            weights_raw = lambda_mean
        else:
            raise ValueError(f'Unsupported phasec_gating_weight_polarity: {self.args.phasec_gating_weight_polarity}')
        denom = torch.clamp(weights_raw.mean().detach(), min=1e-8)
        weights_norm = weights_raw / denom
        alpha = float(self.args.phasec_gating_alpha)
        return 1.0 + alpha * (weights_norm - 1.0)

    @staticmethod
    def _summarize_weight_array(weight_values):
        flat = np.concatenate(weight_values).astype(np.float64)
        summary = {
            'weight_mean': float(np.mean(flat)),
            'weight_std': float(np.std(flat)),
            'weight_min': float(np.min(flat)),
            'weight_max': float(np.max(flat)),
            'weight_p10': float(np.percentile(flat, 10)),
            'weight_p50': float(np.percentile(flat, 50)),
            'weight_p90': float(np.percentile(flat, 90)),
        }
        return summary

    def _log_weight_summary(self, epoch, weight_values):
        if not weight_values or not self._phasec_should_log():
            return
        summary = self._summarize_weight_array(weight_values)
        summary_str = ', '.join(f'{key}={value:.6f}' for key, value in summary.items())
        print(f'Epoch {epoch} gating weight summary | {summary_str}')

    def _forward_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_regime_x_aux=None, batch_regime_y_aux=None,
                       batch_phase_d_lambda=None, batch_phase_d_delta=None):
        if 'PEMS' in self.args.data or 'Solar' in self.args.data:
            batch_x_mark = None
            batch_y_mark = None
        else:
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

        if batch_regime_x_aux is not None:
            batch_regime_x_aux = batch_regime_x_aux.float().to(self.device)
        if batch_regime_y_aux is not None:
            batch_regime_y_aux = batch_regime_y_aux.float().to(self.device)
        if batch_phase_d_lambda is not None:
            batch_phase_d_lambda = batch_phase_d_lambda.float().to(self.device).view(-1)
        if batch_phase_d_delta is not None:
            batch_phase_d_delta = batch_phase_d_delta.float().to(self.device)

        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        if self.args.output_attention:
            outputs = self.model(
                batch_x, batch_x_mark, dec_inp, batch_y_mark,
                regime_aux_enc=batch_regime_x_aux, regime_aux_dec=batch_regime_y_aux,
                phase_d_lambda=batch_phase_d_lambda, phase_d_delta=batch_phase_d_delta,
            )[0]
        else:
            outputs = self.model(
                batch_x, batch_x_mark, dec_inp, batch_y_mark,
                regime_aux_enc=batch_regime_x_aux, regime_aux_dec=batch_regime_y_aux,
                phase_d_lambda=batch_phase_d_lambda, phase_d_delta=batch_phase_d_delta,
            )

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        return outputs, batch_y

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch in vali_loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark, _, batch_regime_x_aux, batch_regime_y_aux, _, _ = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, batch_y = self._forward_batch(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_regime_x_aux=batch_regime_x_aux, batch_regime_y_aux=batch_regime_y_aux)
                else:
                    outputs, batch_y = self._forward_batch(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_regime_x_aux=batch_regime_x_aux, batch_regime_y_aux=batch_regime_y_aux)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true).mean().item()
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
        self._log_phasec_gating_config()
        self._log_phase_d_config()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            epoch_weight_values = []

            self.model.train()
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_gating, batch_regime_x_aux, batch_regime_y_aux, batch_phase_d_lambda, batch_phase_d_delta = self._unpack_batch(batch)
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, batch_y = self._forward_batch(
                            batch_x, batch_y, batch_x_mark, batch_y_mark,
                            batch_regime_x_aux=batch_regime_x_aux, batch_regime_y_aux=batch_regime_y_aux,
                            batch_phase_d_lambda=batch_phase_d_lambda, batch_phase_d_delta=batch_phase_d_delta,
                        )
                        loss_raw = criterion(outputs, batch_y)
                        loss_per_sample = loss_raw.mean(dim=(1, 2))
                        sample_weights = self._compute_phasec_sample_weights(batch_gating, loss_per_sample.shape[0], loss_per_sample.device)
                        loss = (loss_per_sample * sample_weights).mean()
                else:
                    outputs, batch_y = self._forward_batch(
                        batch_x, batch_y, batch_x_mark, batch_y_mark,
                        batch_regime_x_aux=batch_regime_x_aux, batch_regime_y_aux=batch_regime_y_aux,
                        batch_phase_d_lambda=batch_phase_d_lambda, batch_phase_d_delta=batch_phase_d_delta,
                    )
                    loss_raw = criterion(outputs, batch_y)
                    loss_per_sample = loss_raw.mean(dim=(1, 2))
                    sample_weights = self._compute_phasec_sample_weights(batch_gating, loss_per_sample.shape[0], loss_per_sample.device)
                    loss = (loss_per_sample * sample_weights).mean()

                train_loss.append(loss.item())
                if self._phasec_should_log():
                    epoch_weight_values.append(sample_weights.detach().cpu().numpy())

                if (i + 1) % 100 == 0:
                    print("	iters: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('	speed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            self._log_weight_summary(epoch + 1, epoch_weight_values)
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

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
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark, _, batch_regime_x_aux, batch_regime_y_aux, _, _ = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, batch_y = self._forward_batch(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_regime_x_aux=batch_regime_x_aux, batch_regime_y_aux=batch_regime_y_aux)
                else:
                    outputs, batch_y = self._forward_batch(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_regime_x_aux=batch_regime_x_aux, batch_regime_y_aux=batch_regime_y_aux)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if len(preds) % 20 == 1:
                    input_arr = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input_arr.shape
                        input_arr = test_data.inverse_transform(input_arr.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input_arr[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input_arr[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(len(preds) - 1) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        with open('result_long_term_forecast.txt', 'a') as f:
            f.write(setting + '  \n')
            f.write('mse:{}, mae:{}'.format(mse, mae))
            f.write('\n\n')

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

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
            for batch_x, batch_y, batch_x_mark, batch_y_mark in pred_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
