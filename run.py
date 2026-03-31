import argparse
import torch
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
from experiments.exp_long_term_forecasting_partial import Exp_Long_Term_Forecast_Partial
import random
import numpy as np


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {'true', '1', 'yes', 'y'}:
        return True
    if value in {'false', '0', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='iTransformer')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='iTransformer',
                        help='model name, options: [iTransformer, iInformer, iReformer, iFlowformer, iFlashformer]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='custom', help='dataset type, options include [custom, phasec_synth]')
    parser.add_argument('--root_path', type=str, default='./data/electricity/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='electricity.csv', help='data csv file')
    parser.add_argument('--phasec_split_path', type=str, default='', help='path to the Phase C split artifact when data=phasec_synth')
    parser.add_argument('--phasec_gating_lambda_path', type=str, default='', help='optional Phase C gating lambda artifact for gating-only experiments')
    parser.add_argument('--phasec_gating_lambda_hash', type=str, default='', help='optional frozen hash for the Phase C gating lambda artifact')
    parser.add_argument('--phasec_gating_mode', type=str, default='none', choices=['none', 'noop', 'loss_weighting'], help='Phase C gating integration mode')
    parser.add_argument('--phasec_gating_weight_polarity', type=str, default='inverse', choices=['direct', 'inverse'], help='How gating lambda maps to sample weights when gating mode is active')
    parser.add_argument('--phasec_gating_alpha', type=float, default=1.0, help='strength shrink for gating weights after mean-1 normalization; 1.0 keeps full weighting, 0.0 collapses to baseline')
    parser.add_argument('--phasec_regime_lambda_path', type=str, default='', help='optional Phase C regime lambda artifact for regime-only experiments')
    parser.add_argument('--phasec_regime_lambda_hash', type=str, default='', help='optional frozen hash for the Phase C regime lambda artifact')
    parser.add_argument('--phasec_regime_mode', type=str, default='none', choices=['none', 'noop', 'extra_time_feature', 'light_aux_input'], help='Phase C regime integration mode')
    parser.add_argument('--graph_enable', '--phase_d_enable', dest='graph_enable', type=str2bool, nargs='?', const=True, default=False, help='enable train-time graph-guided path')
    parser.add_argument('--graph_interface_dir', '--phase_d_interface_dir', dest='graph_interface_dir', type=str, default='', help='directory that stores the exported graph interface bundle')
    parser.add_argument('--graph_use_static_bias', '--phase_d_use_static_bias', dest='graph_use_static_bias', type=str2bool, nargs='?', const=True, default=True, help='apply static graph bias when graph guidance is enabled')
    parser.add_argument('--graph_use_dynamic_bias', '--phase_d_use_dynamic_bias', dest='graph_use_dynamic_bias', type=str2bool, nargs='?', const=True, default=True, help='apply train-time dynamic graph bias when graph guidance is enabled')
    parser.add_argument('--graph_use_lambda_gate', '--phase_d_use_lambda_gate', dest='graph_use_lambda_gate', type=str2bool, nargs='?', const=True, default=True, help='gate dynamic graph bias by (1 - lambda^(w)) when enabled')
    parser.add_argument('--graph_shuffle_lambda', '--phase_d_shuffle_lambda', dest='graph_shuffle_lambda', type=str2bool, nargs='?', const=True, default=False, help='shuffle train-window lambda for negative-control runs')
    parser.add_argument('--graph_eval_use_static_bias', '--phase_d_eval_use_static_bias', dest='graph_eval_use_static_bias', type=str2bool, nargs='?', const=True, default=True, help='keep static graph bias enabled at val/test when graph guidance is active')
    parser.add_argument('--graph_beta_static', '--phase_d_beta_static', dest='graph_beta_static', type=float, default=0.10, help='strength of static graph soft bias')
    parser.add_argument('--graph_beta_dynamic', '--phase_d_beta_dynamic', dest='graph_beta_dynamic', type=float, default=0.05, help='strength of dynamic graph soft bias')
    parser.add_argument('--seed', type=int, default=2023, help='global random seed')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length') # no longer needed in inverted Transformers
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size') # applicable on arbitrary number of variates in inverted Transformers
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=str2bool, nargs='?', const=True, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # iTransformer
    parser.add_argument('--exp_name', type=str, required=False, default='MTSF',
                        help='experiemnt name, options:[MTSF, partial_train]')
    parser.add_argument('--channel_independence', type=bool, default=False, help='whether to use channel_independence mechanism')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    parser.add_argument('--target_root_path', type=str, default='./data/electricity/', help='root path of the data file')
    parser.add_argument('--target_data_path', type=str, default='electricity.csv', help='data file')
    parser.add_argument('--efficient_training', type=bool, default=False, help='whether to use efficient_training (exp_name should be partial train)') # See Figure 8 of our paper for the detail
    parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
    parser.add_argument('--partial_start_index', type=int, default=0, help='the start index of variates for partial training, '
                                                                           'you can select [partial_start_index, min(enc_in + partial_start_index, N)]')

    args = parser.parse_args()

    if args.phasec_gating_mode != 'none' and args.data != 'phasec_synth':
        raise ValueError('Phase C gating modes are only supported when data=phasec_synth')
    if args.phasec_gating_mode != 'none' and not args.phasec_gating_lambda_path:
        raise ValueError('phasec_gating_lambda_path is required when phasec_gating_mode is active')
    if args.phasec_gating_mode == 'none':
        args.phasec_gating_lambda_hash = ''
    if not (0.0 <= args.phasec_gating_alpha <= 1.0):
        raise ValueError('phasec_gating_alpha must be in [0, 1]')
    if args.phasec_regime_mode != 'none' and args.data != 'phasec_synth':
        raise ValueError('Phase C regime modes are only supported when data=phasec_synth')
    if args.phasec_regime_mode != 'none' and not args.phasec_regime_lambda_path:
        raise ValueError('phasec_regime_lambda_path is required when phasec_regime_mode is active')
    if args.phasec_regime_mode == 'none':
        args.phasec_regime_lambda_hash = ''
    if args.graph_enable and not args.graph_interface_dir:
        raise ValueError('graph_interface_dir is required when graph_enable is active')
    if args.graph_use_lambda_gate and not args.graph_use_dynamic_bias:
        raise ValueError('graph_use_lambda_gate requires graph_use_dynamic_bias to be enabled')
    if args.graph_shuffle_lambda and not args.graph_use_lambda_gate:
        raise ValueError('graph_shuffle_lambda only makes sense when graph_use_lambda_gate is enabled')
    if args.graph_eval_use_static_bias and not args.graph_use_static_bias:
        raise ValueError('graph_eval_use_static_bias requires graph_use_static_bias to be enabled')
    if args.graph_beta_static < 0.0:
        raise ValueError('graph_beta_static must be non-negative')
    if args.graph_beta_dynamic < 0.0:
        raise ValueError('graph_beta_dynamic must be non-negative')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print(f'Using random seed: {args.seed}')
    print('Args in experiment:')
    print(args)

    if args.exp_name == 'partial_train': # See Figure 8 of our paper, for the detail
        Exp = Exp_Long_Term_Forecast_Partial
    else: # MTSF: multivariate time series forecasting
        Exp = Exp_Long_Term_Forecast


    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                args.class_strategy, ii)

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
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            args.class_strategy, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
