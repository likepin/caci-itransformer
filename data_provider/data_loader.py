import json
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from utils.phase_d_interface import load_phase_d_train_bundle
import warnings

warnings.filterwarnings('ignore')


def _init_phase_d_fields(dataset, flag, phase_d_enable=False, phase_d_interface_dir='', phase_d_shuffle_lambda=False, phase_d_seed=2023):
    dataset.flag = flag
    dataset.phase_d_enable = bool(phase_d_enable)
    dataset.phase_d_interface_dir = phase_d_interface_dir
    dataset.phase_d_shuffle_lambda = bool(phase_d_shuffle_lambda)
    dataset.phase_d_seed = int(phase_d_seed)
    dataset.phase_d_lambda_train = None
    dataset.phase_d_delta_train = None
    dataset.phase_d_manifest_path = ''


def _attach_phase_d_train_bundle(dataset):
    if not getattr(dataset, 'phase_d_enable', False) or getattr(dataset, 'flag', '') != 'train':
        return
    if hasattr(dataset, 'window_starts'):
        window_starts = np.asarray(dataset.window_starts, dtype=np.int64)
    else:
        max_start = len(dataset.data_x) - dataset.seq_len - dataset.pred_len + 1
        window_starts = np.arange(max_start, dtype=np.int64)
    bundle = load_phase_d_train_bundle(
        root_path=dataset.root_path,
        interface_dir=dataset.phase_d_interface_dir,
        window_starts=window_starts,
        expected_nodes=dataset.data_x.shape[1],
        shuffle_lambda=dataset.phase_d_shuffle_lambda,
        shuffle_seed=dataset.phase_d_seed,
    )
    dataset.phase_d_lambda_train = bundle['lambda_train']
    dataset.phase_d_delta_train = bundle['delta_train']
    dataset.phase_d_manifest_path = bundle['manifest_path']
    print(f'PhaseD train interface bundle ({dataset.flag}): {bundle["interface_dir"]}')
    print(f'PhaseD train windows ({dataset.flag}): {len(dataset.phase_d_lambda_train)}')
    if dataset.phase_d_shuffle_lambda:
        print(f'PhaseD lambda shuffle ({dataset.flag}): enabled with seed={dataset.phase_d_seed}')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 phase_d_enable=False, phase_d_interface_dir='', phase_d_shuffle_lambda=False, phase_d_seed=2023):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        _init_phase_d_fields(self, flag, phase_d_enable=phase_d_enable, phase_d_interface_dir=phase_d_interface_dir,
                             phase_d_shuffle_lambda=phase_d_shuffle_lambda, phase_d_seed=phase_d_seed)

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        _attach_phase_d_train_bundle(self)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.phase_d_lambda_train is not None:
            return seq_x, seq_y, seq_x_mark, seq_y_mark, np.float32(self.phase_d_lambda_train[index]), self.phase_d_delta_train[index]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t',
                 phase_d_enable=False, phase_d_interface_dir='', phase_d_shuffle_lambda=False, phase_d_seed=2023):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        _init_phase_d_fields(self, flag, phase_d_enable=phase_d_enable, phase_d_interface_dir=phase_d_interface_dir,
                             phase_d_shuffle_lambda=phase_d_shuffle_lambda, phase_d_seed=phase_d_seed)

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        _attach_phase_d_train_bundle(self)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.phase_d_lambda_train is not None:
            return seq_x, seq_y, seq_x_mark, seq_y_mark, np.float32(self.phase_d_lambda_train[index]), self.phase_d_delta_train[index]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 phase_d_enable=False, phase_d_interface_dir='', phase_d_shuffle_lambda=False, phase_d_seed=2023):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        _init_phase_d_fields(self, flag, phase_d_enable=phase_d_enable, phase_d_interface_dir=phase_d_interface_dir,
                             phase_d_shuffle_lambda=phase_d_shuffle_lambda, phase_d_seed=phase_d_seed)

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        _attach_phase_d_train_bundle(self)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.phase_d_lambda_train is not None:
            return seq_x, seq_y, seq_x_mark, seq_y_mark, np.float32(self.phase_d_lambda_train[index]), self.phase_d_delta_train[index]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)




class Dataset_PhaseC_Synthetic(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='X.npy', target='OT', scale=True, timeenc=0, freq='h',
                 phasec_split_path=None, phasec_gating_lambda_path=None, phasec_gating_mode='none',
                 phasec_regime_lambda_path=None, phasec_regime_mode='none',
                 phase_d_enable=False, phase_d_interface_dir='', phase_d_shuffle_lambda=False, phase_d_seed=2023):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        _init_phase_d_fields(self, flag, phase_d_enable=phase_d_enable, phase_d_interface_dir=phase_d_interface_dir,
                             phase_d_shuffle_lambda=phase_d_shuffle_lambda, phase_d_seed=phase_d_seed)
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.phasec_split_path = phasec_split_path
        self.phasec_gating_lambda_path = phasec_gating_lambda_path
        self.phasec_gating_mode = phasec_gating_mode
        self.phasec_gating_lambda = None
        self.phasec_regime_lambda_path = phasec_regime_lambda_path
        self.phasec_regime_mode = phasec_regime_mode
        self.phasec_regime_lambda = None
        self.__read_data__()

    def _resolve_split_path(self):
        if not self.phasec_split_path:
            raise ValueError('phasec_split_path is required when data=phasec_synth')
        if os.path.isabs(self.phasec_split_path):
            return self.phasec_split_path
        return os.path.join(self.root_path, self.phasec_split_path)

    def _resolve_optional_artifact_path(self, artifact_path):
        if not artifact_path:
            return ''
        if os.path.isabs(artifact_path):
            return artifact_path
        return os.path.join(self.root_path, artifact_path)

    def _load_source_array(self):
        data_file = os.path.join(self.root_path, self.data_path)
        data = np.load(data_file, allow_pickle=True)
        if isinstance(data, np.lib.npyio.NpzFile):
            if 'data' not in data.files:
                raise ValueError('Expected an npz file with a data array')
            data = data['data']
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError('Phase C synthetic data must have shape [T, C]')
        return data

    @staticmethod
    def _collect_rows(data, intervals):
        slices = [data[start:end] for start, end in intervals if end > start]
        if not slices:
            raise ValueError('No rows available for the requested intervals')
        return np.concatenate(slices, axis=0)

    @staticmethod
    def _sanitize_optional_lambda(lambda_array, label, lambda_path):
        if np.isinf(lambda_array).any():
            raise ValueError(f'Phase C {label} lambda contains Inf values: {lambda_path}')
        nan_mask = np.isnan(lambda_array)
        nan_count = int(nan_mask.sum())
        if nan_count == 0:
            return lambda_array
        valid_idx = np.where(~nan_mask)[0]
        if len(valid_idx) == 0:
            raise ValueError(f'Phase C {label} lambda is entirely NaN: {lambda_path}')
        filled = np.interp(
            np.arange(len(lambda_array), dtype=np.float64),
            valid_idx.astype(np.float64),
            lambda_array[valid_idx].astype(np.float64),
        ).astype(np.float32)
        print(
            f'PhaseC {label} lambda sanitization: filled {nan_count} NaNs via linear interpolation '
            f'with edge-value extrapolation from {lambda_path}'
        )
        return filled

    def _load_optional_lambda(self, artifact_path, mode, expected_length, label):
        lambda_path = self._resolve_optional_artifact_path(artifact_path)
        if not lambda_path:
            if mode != 'none':
                raise ValueError(f'phasec_{label}_lambda_path is required when phasec_{label}_mode is active')
            return None, ''
        lambda_array = np.load(lambda_path, allow_pickle=True)
        if isinstance(lambda_array, np.lib.npyio.NpzFile):
            if 'lambda_t' in lambda_array.files:
                lambda_array = lambda_array['lambda_t']
            elif 'arr_0' in lambda_array.files:
                lambda_array = lambda_array['arr_0']
            else:
                raise ValueError('Expected lambda npz with lambda_t or arr_0')
        lambda_array = np.asarray(lambda_array).reshape(-1)
        if len(lambda_array) != expected_length:
            raise ValueError(f'{label.capitalize()} lambda length {len(lambda_array)} does not match data length {expected_length}')
        lambda_array = self._sanitize_optional_lambda(lambda_array, label, lambda_path)
        return lambda_array, lambda_path

    def _build_valid_starts(self, intervals):
        starts = []
        for start, end in intervals:
            last_start = end - self.seq_len - self.pred_len
            if last_start < start:
                continue
            starts.extend(range(start, last_start + 1))
        return np.asarray(starts, dtype=np.int64)

    def _build_time_marks(self, num_rows):
        dt_index = pd.date_range(start='2000-01-01 00:00:00', periods=num_rows, freq=self.freq)
        df_stamp = pd.DataFrame({'date': dt_index})
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            return df_stamp.drop(['date'], axis=1).values
        data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
        return data_stamp.transpose(1, 0)

    def __read_data__(self):
        self.scaler = StandardScaler()
        split_path = self._resolve_split_path()
        with open(split_path, 'r', encoding='utf-8') as handle:
            split_artifact = json.load(handle)

        raw_data = self._load_source_array()
        expected_length = split_artifact['indexing']['length']
        if len(raw_data) != expected_length:
            raise ValueError(f'Phase C split length {expected_length} does not match data length {len(raw_data)}')
        self.phasec_gating_lambda, self.phasec_gating_lambda_path = self._load_optional_lambda(
            self.phasec_gating_lambda_path, self.phasec_gating_mode, expected_length, 'gating'
        )
        self.phasec_regime_lambda, self.phasec_regime_lambda_path = self._load_optional_lambda(
            self.phasec_regime_lambda_path, self.phasec_regime_mode, expected_length, 'regime'
        )

        if self.features in ['M', 'MS']:
            model_data = raw_data
        elif self.features == 'S':
            model_data = raw_data[:, -1:]
        else:
            raise ValueError(f'Unsupported features mode: {self.features}')

        train_intervals = split_artifact['splits']['train']['intervals']
        active_intervals = split_artifact['splits'][self.flag]['intervals']
        self.split_intervals = active_intervals
        self.split_artifact_path = split_path

        if self.scale:
            train_rows = self._collect_rows(model_data, train_intervals)
            self.scaler.fit(train_rows)
            model_data = self.scaler.transform(model_data)

        self.data_x = model_data
        self.data_y = model_data
        self.data_stamp = self._build_time_marks(len(model_data))
        self.window_starts = self._build_valid_starts(active_intervals)
        if len(self.window_starts) == 0:
            raise ValueError(f'No valid windows available for split {self.flag}')
        _attach_phase_d_train_bundle(self)

        print(f'PhaseC split artifact ({self.flag}): {self.split_artifact_path}')
        print(f'PhaseC intervals ({self.flag}): {self.split_intervals}')
        print(f'PhaseC valid windows ({self.flag}): {len(self.window_starts)}')
        if self.phasec_gating_lambda is not None:
            print(f'PhaseC gating lambda artifact ({self.flag}): {self.phasec_gating_lambda_path}')
            print(f'PhaseC gating lambda length ({self.flag}): {len(self.phasec_gating_lambda)}')
        if self.phasec_regime_lambda is not None:
            print(f'PhaseC regime lambda artifact ({self.flag}): {self.phasec_regime_lambda_path}')
            print(f'PhaseC regime lambda length ({self.flag}): {len(self.phasec_regime_lambda)}')

    def _slice_sample_by_start(self, s_begin, sample_index=None):
        s_begin = int(s_begin)
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        if self.phasec_gating_lambda is not None:
            gating_window = self.phasec_gating_lambda[s_begin:r_end]
            expected_window = self.seq_len + self.pred_len
            if len(gating_window) != expected_window:
                raise ValueError(f'Phase C gating lambda misaligned at start={s_begin}: got {len(gating_window)}, expected {expected_window}')
            gating_future = self.phasec_gating_lambda[s_end:r_end].astype(np.float32)
        else:
            gating_future = np.ones(self.pred_len, dtype=np.float32)

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        regime_x_aux = np.zeros((self.seq_len, 1), dtype=np.float32)
        regime_y_aux = np.zeros((self.label_len + self.pred_len, 1), dtype=np.float32)
        if self.phasec_regime_lambda is not None:
            regime_x = self.phasec_regime_lambda[s_begin:s_end]
            regime_y = self.phasec_regime_lambda[r_begin:r_end]
            if len(regime_x) != self.seq_len:
                raise ValueError(f'Phase C regime lambda misaligned for encoder at start={s_begin}: got {len(regime_x)}, expected {self.seq_len}')
            expected_decoder = self.label_len + self.pred_len
            if len(regime_y) != expected_decoder:
                raise ValueError(f'Phase C regime lambda misaligned for decoder at start={s_begin}: got {len(regime_y)}, expected {expected_decoder}')
            regime_x_aux = regime_x.astype(np.float32)[:, None]
            regime_y_aux = regime_y.astype(np.float32)[:, None]
            if self.phasec_regime_mode == 'extra_time_feature':
                seq_x_mark = np.concatenate([seq_x_mark.astype(np.float32), regime_x_aux], axis=1)
                seq_y_mark = np.concatenate([seq_y_mark.astype(np.float32), regime_y_aux], axis=1)

        if self.phase_d_lambda_train is not None:
            if sample_index is None:
                raise ValueError('Phase D train bundle requires a dataset sample index for alignment')
            return seq_x, seq_y, seq_x_mark, seq_y_mark, gating_future, regime_x_aux, regime_y_aux, np.float32(self.phase_d_lambda_train[sample_index]), self.phase_d_delta_train[sample_index]
        return seq_x, seq_y, seq_x_mark, seq_y_mark, gating_future, regime_x_aux, regime_y_aux

    def __getitem__(self, index):
        return self._slice_sample_by_start(self.window_starts[index], sample_index=index)

    def __len__(self):
        return len(self.window_starts)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_PEMS(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        data_file = os.path.join(self.root_path, self.data_path)
        data = np.load(data_file, allow_pickle=True)
        data = data['data'][:, :, 0]

        train_ratio = 0.6
        valid_ratio = 0.2
        train_data = data[:int(train_ratio * len(data))]
        valid_data = data[int(train_ratio * len(data)): int((train_ratio + valid_ratio) * len(data))]
        test_data = data[int((train_ratio + valid_ratio) * len(data)):]
        total_data = [train_data, valid_data, test_data]
        data = total_data[self.set_type]

        if self.scale:
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        df = pd.DataFrame(data)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values

        self.data_x = df
        self.data_y = df

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Solar(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = []
        with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = int(len(df_raw) * 0.1)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
