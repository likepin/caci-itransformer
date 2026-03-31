from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar, Dataset_PEMS, \
    Dataset_Pred, Dataset_PhaseC_Synthetic
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Solar': Dataset_Solar,
    'PEMS': Dataset_PEMS,
    'custom': Dataset_Custom,
    'phasec_synth': Dataset_PhaseC_Synthetic,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    data_kwargs = dict(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
    )
    if args.data in {'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'custom', 'phasec_synth'}:
        data_kwargs['graph_enable'] = getattr(args, 'graph_enable', False)
        data_kwargs['graph_interface_dir'] = getattr(args, 'graph_interface_dir', '')
        data_kwargs['graph_shuffle_lambda'] = getattr(args, 'graph_shuffle_lambda', False)
        data_kwargs['graph_seed'] = getattr(args, 'seed', 2023)
    if args.data == 'phasec_synth':
        data_kwargs['phasec_split_path'] = args.phasec_split_path
        data_kwargs['phasec_gating_lambda_path'] = getattr(args, 'phasec_gating_lambda_path', '')
        data_kwargs['phasec_gating_mode'] = getattr(args, 'phasec_gating_mode', 'none')
        data_kwargs['phasec_regime_lambda_path'] = getattr(args, 'phasec_regime_lambda_path', '')
        data_kwargs['phasec_regime_mode'] = getattr(args, 'phasec_regime_mode', 'none')

    data_set = Data(**data_kwargs)
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
