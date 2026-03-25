import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data_provider.data_loader import Dataset_PhaseC_Synthetic
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast


class SliceWindowDataset(Dataset):
    def __init__(self, base_dataset, starts):
        self.base = base_dataset
        self.starts = np.asarray(starts, dtype=np.int64)
        self.seq_len = base_dataset.seq_len
        self.label_len = base_dataset.label_len
        self.pred_len = base_dataset.pred_len

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        s_begin = int(self.starts[index])
        seq_x, seq_y, seq_x_mark, seq_y_mark, _ = self.base._slice_sample_by_start(s_begin)
        times = np.arange(s_begin + self.seq_len, s_begin + self.seq_len + self.pred_len, dtype=np.int64)
        return seq_x, seq_y, seq_x_mark, seq_y_mark, times


def build_args(train_cfg_path, split_artifact_path, root_path, data_path, use_gpu, phasec_regime_lambda_path='', phasec_regime_lambda_hash='', phasec_regime_mode='none'):
    cfg = json.loads(Path(train_cfg_path).read_text(encoding='utf-8'))
    frozen = cfg['frozen_round1_training_config']
    return SimpleNamespace(
        model=frozen['model'],
        exp_name='MTSF',
        use_gpu=bool(use_gpu and torch.cuda.is_available()),
        gpu=0,
        use_multi_gpu=False,
        devices='0',
        seq_len=int(frozen['seq_len']),
        label_len=int(frozen['label_len']),
        pred_len=int(frozen['pred_len']),
        enc_in=int(frozen['enc_in']),
        dec_in=int(frozen['dec_in']),
        c_out=int(frozen['c_out']),
        d_model=int(frozen['d_model']),
        n_heads=int(frozen['n_heads']),
        e_layers=int(frozen['e_layers']),
        d_layers=int(frozen['d_layers']),
        d_ff=int(frozen['d_ff']),
        moving_avg=25,
        factor=1,
        distil=True,
        dropout=float(frozen['dropout']),
        embed=frozen['embed'],
        activation=frozen['activation'],
        output_attention=False,
        do_predict=False,
        features=frozen['features'],
        target='OT',
        freq=frozen['freq'],
        checkpoints='./checkpoints/',
        num_workers=0,
        itr=1,
        train_epochs=int(frozen['train_epochs']),
        batch_size=int(frozen['batch_size']),
        patience=int(frozen['patience']),
        learning_rate=float(frozen['learning_rate']),
        des=frozen['des'],
        loss=frozen['loss'],
        lradj=frozen['lradj'],
        use_amp=False,
        channel_independence=False,
        inverse=False,
        class_strategy=frozen['class_strategy'],
        target_root_path=root_path,
        target_data_path=data_path,
        efficient_training=False,
        use_norm=bool(frozen['use_norm']),
        partial_start_index=0,
        data='phasec_synth',
        root_path=root_path,
        data_path=data_path,
        phasec_split_path=split_artifact_path,
        phasec_gating_lambda_path='',
        phasec_gating_lambda_hash='',
        phasec_gating_mode='none',
        phasec_gating_weight_polarity='direct',
        phasec_gating_alpha=1.0,
        phasec_regime_lambda_path=phasec_regime_lambda_path,
        phasec_regime_lambda_hash=phasec_regime_lambda_hash,
        phasec_regime_mode=phasec_regime_mode,
    )


def overlapping_starts(total_length, seq_len, pred_len, intervals):
    max_start = total_length - seq_len - pred_len
    starts = np.arange(max_start + 1, dtype=np.int64)
    pred_start = starts + seq_len
    pred_end = starts + seq_len + pred_len
    keep = np.zeros_like(starts, dtype=bool)
    for start, end in intervals:
        keep |= (pred_start < int(end)) & (pred_end > int(start))
    return starts[keep]


def build_interval_mask(times, intervals):
    mask = np.zeros(times.shape, dtype=bool)
    for start, end in intervals:
        mask |= (times >= int(start)) & (times < int(end))
    return mask


def masked_metrics(preds, trues, mask):
    channel_count = preds.shape[2]
    scalar_mask = np.broadcast_to(mask[:, :, None], preds.shape)
    selected_pred = preds[scalar_mask]
    selected_true = trues[scalar_mask]
    abs_err = np.abs(selected_pred - selected_true)
    sq_err = (selected_pred - selected_true) ** 2
    return {
        'prediction_element_count': int(mask.sum()),
        'scalar_element_count': int(mask.sum() * channel_count),
        'mae': float(abs_err.mean()),
        'mse': float(sq_err.mean()),
        'rmse': float(np.sqrt(sq_err.mean())),
    }


def main():
    parser = argparse.ArgumentParser(description='Run Phase C switch-window slice evaluation from an existing checkpoint.')
    parser.add_argument('--train-config', required=True)
    parser.add_argument('--split-artifact', required=True)
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--data-path', default='X.npy')
    parser.add_argument('--checkpoint-dir', required=True)
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--variant-name', default='')
    parser.add_argument('--use-gpu', default='true')
    parser.add_argument('--phasec-regime-lambda-path', default='')
    parser.add_argument('--phasec-regime-lambda-hash', default='')
    parser.add_argument('--phasec-regime-mode', default='none', choices=['none', 'noop', 'extra_time_feature'])
    args = parser.parse_args()

    use_gpu = str(args.use_gpu).strip().lower() in {'true', '1', 'yes', 'y'}

    split = json.loads(Path(args.split_artifact).read_text(encoding='utf-8'))
    switch_intervals = split['evaluation_slices']['switch_window_eval']['intervals']
    switch_pre = [split['switch_window']['pre_slice']]
    switch_post = [split['switch_window']['post_slice']]

    model_args = build_args(args.train_config, args.split_artifact, args.root_path, args.data_path, use_gpu, args.phasec_regime_lambda_path, args.phasec_regime_lambda_hash, args.phasec_regime_mode)
    exp = Exp_Long_Term_Forecast(model_args)

    checkpoint_file = Path(args.checkpoint_dir) / 'checkpoint.pth'
    state = torch.load(checkpoint_file, map_location=exp.device)
    exp.model.load_state_dict(state)
    exp.model.eval()

    base_dataset = Dataset_PhaseC_Synthetic(
        root_path=args.root_path,
        data_path=args.data_path,
        flag='test',
        size=[model_args.seq_len, model_args.label_len, model_args.pred_len],
        features=model_args.features,
        target=model_args.target,
        scale=True,
        timeenc=0 if model_args.embed != 'timeF' else 1,
        freq=model_args.freq,
        phasec_split_path=args.split_artifact,
        phasec_gating_lambda_path='',
        phasec_gating_mode='none',
        phasec_regime_lambda_path=args.phasec_regime_lambda_path,
        phasec_regime_mode=args.phasec_regime_mode,
    )

    starts = overlapping_starts(len(base_dataset.data_x), base_dataset.seq_len, base_dataset.pred_len, switch_intervals)
    if len(starts) == 0:
        raise ValueError('No overlapping starts for switch-window slice evaluation')

    slice_dataset = SliceWindowDataset(base_dataset, starts)
    slice_loader = DataLoader(slice_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    preds = []
    trues = []
    times_all = []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark, batch_times in slice_loader:
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            outputs, batch_y = exp._forward_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(outputs.detach().cpu().numpy())
            trues.append(batch_y.detach().cpu().numpy())
            times_all.append(batch_times.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    times = np.concatenate(times_all, axis=0)

    switch_mask = build_interval_mask(times, switch_intervals)
    switch_pre_mask = build_interval_mask(times, switch_pre)
    switch_post_mask = build_interval_mask(times, switch_post)

    summary = {
        'artifact_name': 'phasec_switch_window_slice_metrics',
        'variant_name': args.variant_name or Path(args.results_dir).name,
        'checkpoint_dir': str(Path(args.checkpoint_dir)),
        'results_dir': str(Path(args.results_dir)),
        'evaluation_unit': 'prediction_elements_without_timestamp_dedup',
        'indexing': 'zero_based_half_open_[start,end)',
        'window_start_count': int(len(starts)),
        'window_start_min': int(starts.min()),
        'window_start_max': int(starts.max()),
        'forecast_time_min': int(times.min()),
        'forecast_time_max': int(times.max()),
        'slices': {
            'switch_window': {
                'intervals': switch_intervals,
                **masked_metrics(preds, trues, switch_mask),
            },
            'switch_pre': {
                'intervals': switch_pre,
                **masked_metrics(preds, trues, switch_pre_mask),
            },
            'switch_post': {
                'intervals': switch_post,
                **masked_metrics(preds, trues, switch_post_mask),
            },
        },
    }

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / 'phasec_switch_slice_metrics.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(json.dumps(summary, indent=2))
    print(f'Wrote switch-window slice metrics to: {output_path}')


if __name__ == '__main__':
    main()
