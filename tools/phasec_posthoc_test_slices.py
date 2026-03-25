import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np


def build_valid_starts(intervals, seq_len, pred_len):
    starts = []
    for start, end in intervals:
        last_start = end - seq_len - pred_len
        if last_start < start:
            continue
        starts.extend(range(start, last_start + 1))
    return np.asarray(starts, dtype=np.int64)


def build_time_index(starts, seq_len, pred_len):
    return starts[:, None] + seq_len + np.arange(pred_len, dtype=np.int64)[None, :]


def build_interval_mask(times, intervals):
    mask = np.zeros(times.shape, dtype=bool)
    for start, end in intervals:
        mask |= (times >= int(start)) & (times < int(end))
    return mask


def masked_metrics(preds, trues, mask):
    if preds.shape != trues.shape:
        raise ValueError(f'pred/true shape mismatch: {preds.shape} vs {trues.shape}')
    if mask.shape != preds.shape[:2]:
        raise ValueError(f'mask shape mismatch: {mask.shape} vs {preds.shape[:2]}')
    channel_count = preds.shape[2]
    scalar_mask = np.broadcast_to(mask[:, :, None], preds.shape)
    selected_pred = preds[scalar_mask]
    selected_true = trues[scalar_mask]
    if selected_pred.size == 0:
        raise ValueError('Slice mask selected zero prediction elements')
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
    parser = argparse.ArgumentParser(description='Compute Phase C pre/post slice metrics from existing test outputs.')
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--split-artifact', required=True)
    parser.add_argument('--seq-len', type=int, default=96)
    parser.add_argument('--pred-len', type=int, default=96)
    parser.add_argument('--output-json', default='')
    parser.add_argument('--variant-name', default='')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    split_path = Path(args.split_artifact)

    preds = np.load(results_dir / 'pred.npy')
    trues = np.load(results_dir / 'true.npy')
    metrics_arr = np.load(results_dir / 'metrics.npy')

    with split_path.open('r', encoding='utf-8') as handle:
        split = json.load(handle)

    test_intervals = split['splits']['test']['intervals']
    starts = build_valid_starts(test_intervals, args.seq_len, args.pred_len)
    if len(starts) != preds.shape[0]:
        raise ValueError(
            f'Window count mismatch for {results_dir}: results have {preds.shape[0]} windows but reconstructed test starts have {len(starts)}'
        )

    times = build_time_index(starts, args.seq_len, args.pred_len)
    pre_intervals = split['evaluation_slices']['pre_eval']['intervals']
    post_intervals = split['evaluation_slices']['post_eval']['intervals']

    pre_mask = build_interval_mask(times, pre_intervals)
    post_mask = build_interval_mask(times, post_intervals)

    summary = {
        'artifact_name': 'phasec_posthoc_test_slice_metrics',
        'results_dir': str(results_dir),
        'variant_name': args.variant_name or results_dir.name,
        'evaluation_unit': 'prediction_elements_without_timestamp_dedup',
        'indexing': 'zero_based_half_open_[start,end)',
        'pred_shape': list(preds.shape),
        'test_window_count': int(preds.shape[0]),
        'forecast_time_min': int(times.min()),
        'forecast_time_max': int(times.max()),
        'global_metrics_from_results': {
            'mae': float(metrics_arr[0]),
            'mse': float(metrics_arr[1]),
            'rmse': float(metrics_arr[2]),
            'mape': float(metrics_arr[3]),
            'mspe': float(metrics_arr[4]),
        },
        'slices': {
            'pre_eval': {
                'intervals': pre_intervals,
                **masked_metrics(preds, trues, pre_mask),
            },
            'post_eval': {
                'intervals': post_intervals,
                **masked_metrics(preds, trues, post_mask),
            },
        },
    }

    output_path = Path(args.output_json) if args.output_json else results_dir / 'phasec_posthoc_slice_metrics.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(json.dumps(summary, indent=2))
    print(f'Wrote posthoc slice metrics to: {output_path}')


if __name__ == '__main__':
    main()
