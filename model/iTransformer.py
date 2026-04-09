import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np
from utils.graph_interface import load_graph_static_bundle


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        self.phasec_regime_mode = getattr(configs, 'phasec_regime_mode', 'none')
        self.graph_enable = bool(getattr(configs, 'graph_enable', False))
        self.graph_mode = getattr(configs, 'graph_mode', 'soft_bias')
        self.graph_use_static_bias = bool(getattr(configs, 'graph_use_static_bias', False))
        self.graph_use_dynamic_bias = bool(getattr(configs, 'graph_use_dynamic_bias', False))
        self.graph_use_lambda_gate = bool(getattr(configs, 'graph_use_lambda_gate', False))
        self.graph_lambda_gate_polarity = getattr(configs, 'graph_lambda_gate_polarity', 'inverse')
        self.graph_eval_use_static_bias = bool(getattr(configs, 'graph_eval_use_static_bias', False))
        self.graph_beta_static = float(getattr(configs, 'graph_beta_static', 0.0))
        self.graph_beta_dynamic = float(getattr(configs, 'graph_beta_dynamic', 0.0))
        self.graph_soft_bias_scale_mode = getattr(configs, 'graph_soft_bias_scale_mode', 'fixed')
        self.graph_residual_alpha = float(getattr(configs, 'graph_residual_alpha', 0.0))
        self.graph_residual_scale_mode = getattr(configs, 'graph_residual_scale_mode', 'fixed')
        self.graph_static_mix_mode = getattr(configs, 'graph_static_mix_mode', 'fixed')
        self.graph_manifest_path = ''
        self.graph_interface_dir = getattr(configs, 'graph_interface_dir', '')
        if self.phasec_regime_mode == 'light_aux_input':
            self.regime_aux_enc_embedding = nn.Linear(configs.seq_len, configs.d_model)
            self.regime_aux_dec_embedding = nn.Linear(configs.label_len + configs.pred_len, configs.d_model)
        needs_static_bundle = self.graph_enable and (
            self.graph_mode == 'static_causal_residual' or self.graph_use_static_bias or self.graph_use_dynamic_bias
        )
        if needs_static_bundle:
            static_bundle = load_graph_static_bundle(
                root_path=getattr(configs, 'root_path', ''),
                interface_dir=self.graph_interface_dir,
                expected_nodes=configs.enc_in,
            )
            self.register_buffer('graph_a_base', torch.from_numpy(static_bundle['a_base']).float())
            self.register_buffer('graph_support', torch.from_numpy(static_bundle['support']).float())
            self.graph_manifest_path = static_bundle['manifest_path']
        else:
            self.graph_a_base = None
            self.graph_support = None
        if self.graph_enable and self.graph_mode == 'soft_bias' and self.graph_soft_bias_scale_mode == 'learnable':
            self.graph_beta_static_log = nn.Parameter(torch.log(torch.tensor(max(self.graph_beta_static, 1e-6), dtype=torch.float32)))
            self.graph_beta_dynamic_log = nn.Parameter(torch.log(torch.tensor(max(self.graph_beta_dynamic, 1e-6), dtype=torch.float32)))
        else:
            self.graph_beta_static_log = None
            self.graph_beta_dynamic_log = None
        if self.graph_enable and self.graph_mode in {'residual_head', 'static_causal_residual'} and (
            self.graph_mode == 'static_causal_residual' or self.graph_use_static_bias or self.graph_use_dynamic_bias
        ):
            residual_input_dim = configs.d_model * (2 if self.graph_mode == 'static_causal_residual' else 3)
            self.graph_residual_norm = nn.LayerNorm(residual_input_dim)
            self.graph_residual_head = nn.Sequential(
                nn.Linear(residual_input_dim, configs.d_model),
                nn.GELU(),
                nn.Dropout(configs.dropout),
                nn.Linear(configs.d_model, configs.pred_len),
            )
        else:
            self.graph_residual_norm = None
            self.graph_residual_head = None
        if self.graph_enable and self.graph_mode in {'residual_head', 'static_causal_residual'} and self.graph_residual_scale_mode == 'learnable':
            self.graph_residual_alpha_log = nn.Parameter(torch.log(torch.tensor(max(self.graph_residual_alpha, 1e-6), dtype=torch.float32)))
        else:
            self.graph_residual_alpha_log = None
        if self.graph_enable and self.graph_mode == 'static_causal_residual':
            static_pool_dim = configs.d_model * 2
            self.graph_causal_pool_norm = nn.LayerNorm(static_pool_dim)
            self.graph_causal_pool_head = nn.Sequential(
                nn.Linear(static_pool_dim, configs.d_model),
                nn.GELU(),
                nn.Dropout(configs.dropout),
                nn.Linear(configs.d_model, 1),
            )
        else:
            self.graph_causal_pool_norm = None
            self.graph_causal_pool_head = None
        if self.graph_enable and self.graph_mode == 'static_causal_residual' and self.graph_static_mix_mode == 'softmax':
            static_mix_dim = configs.d_model * 2
            self.graph_static_mix_norm = nn.LayerNorm(static_mix_dim)
            self.graph_static_mix_head = nn.Sequential(
                nn.Linear(static_mix_dim, configs.d_model),
                nn.GELU(),
                nn.Dropout(configs.dropout),
                nn.Linear(configs.d_model, 2),
            )
        else:
            self.graph_static_mix_norm = None
            self.graph_static_mix_head = None
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def _graph_soft_bias_scale(self, branch):
        if self.graph_soft_bias_scale_mode == 'learnable':
            if branch == 'static' and self.graph_beta_static_log is not None:
                return torch.exp(self.graph_beta_static_log)
            if branch == 'dynamic' and self.graph_beta_dynamic_log is not None:
                return torch.exp(self.graph_beta_dynamic_log)
        if branch == 'static':
            return self.graph_beta_static
        if branch == 'dynamic':
            return self.graph_beta_dynamic
        raise ValueError(f'Unsupported graph soft-bias branch: {branch}')

    def _graph_residual_scale(self):
        if self.graph_residual_scale_mode == 'learnable' and self.graph_residual_alpha_log is not None:
            return torch.exp(self.graph_residual_alpha_log)
        return self.graph_residual_alpha

    def _build_graph_bias(self, batch_size, token_count, num_variates, device, graph_lambda=None, graph_delta=None):
        if self.graph_mode != 'soft_bias':
            return None
        graph_bias = None
        if self.graph_enable and self.graph_use_static_bias:
            use_static = self.training or self.graph_eval_use_static_bias
            if use_static:
                graph_bias = torch.zeros(batch_size, token_count, token_count, device=device, dtype=torch.float32)
                graph_bias[:, :num_variates, :num_variates] += self._graph_soft_bias_scale('static') * self.graph_a_base.to(device)
        if self.graph_enable and self.training and self.graph_use_dynamic_bias:
            if graph_delta is None:
                raise ValueError('Graph dynamic bias is enabled but graph_delta was not provided for a training batch')
            dynamic_block = self._graph_soft_bias_scale('dynamic') * graph_delta.float().to(device)
            if self.graph_use_lambda_gate:
                if graph_lambda is None:
                    raise ValueError('Graph lambda gate is enabled but graph_lambda was not provided for a training batch')
                gate = self._graph_lambda_gate(graph_lambda, batch_size, device)
                dynamic_block = gate * dynamic_block
            if graph_bias is None:
                graph_bias = torch.zeros(batch_size, token_count, token_count, device=device, dtype=torch.float32)
                graph_bias[:, :num_variates, :num_variates] += dynamic_block
        return graph_bias

    def _build_graph_residual(self, variate_tokens, device, graph_lambda=None, graph_delta=None):
        if not self.graph_enable or self.graph_mode != 'residual_head' or self.graph_residual_head is None:
            return None

        static_active = self.graph_use_static_bias and (self.training or self.graph_eval_use_static_bias)
        dynamic_active = self.graph_use_dynamic_bias and self.training
        if not static_active and not dynamic_active:
            return None

        batch_size, _, _ = variate_tokens.shape
        static_tokens = torch.zeros_like(variate_tokens)
        dynamic_tokens = torch.zeros_like(variate_tokens)

        if static_active:
            static_matrix = self.graph_beta_static * self.graph_a_base.to(device)
            static_tokens = torch.einsum('ij,bje->bie', static_matrix, variate_tokens)

        if dynamic_active:
            if graph_delta is None:
                raise ValueError('Graph dynamic branch is enabled but graph_delta was not provided for a training batch')
            dynamic_matrix = self.graph_beta_dynamic * graph_delta.float().to(device)
            if self.graph_use_lambda_gate:
                if graph_lambda is None:
                    raise ValueError('Graph lambda gate is enabled but graph_lambda was not provided for a training batch')
                gate = self._graph_lambda_gate(graph_lambda, batch_size, device)
                dynamic_matrix = gate * dynamic_matrix
            dynamic_tokens = torch.einsum('bij,bje->bie', dynamic_matrix, variate_tokens)

        residual_input = torch.cat([variate_tokens, static_tokens, dynamic_tokens], dim=-1)
        residual_tokens = self.graph_residual_head(self.graph_residual_norm(residual_input))
        return self._graph_residual_scale() * residual_tokens.permute(0, 2, 1)

    def _build_static_causal_pool(self, variate_tokens, device):
        if self.graph_causal_pool_head is None or self.graph_causal_pool_norm is None:
            raise ValueError('static_causal_residual requires graph_causal_pool_head to be initialized')

        batch_size, num_variates, _ = variate_tokens.shape
        causal_mask = self.graph_support.to(device).clone()
        causal_mask.fill_diagonal_(0.0)

        target_tokens = variate_tokens.unsqueeze(2).expand(-1, -1, num_variates, -1)
        source_tokens = variate_tokens.unsqueeze(1).expand(-1, num_variates, -1, -1)
        pool_input = torch.cat([target_tokens, source_tokens], dim=-1)
        pool_scores = self.graph_causal_pool_head(self.graph_causal_pool_norm(pool_input)).squeeze(-1)

        mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
        has_parent = (causal_mask.sum(dim=-1, keepdim=True) > 0).unsqueeze(0)
        pool_scores = pool_scores.masked_fill(mask <= 0.0, -1e9)
        pool_scores = torch.where(has_parent.expand_as(pool_scores), pool_scores, torch.zeros_like(pool_scores))
        pool_weights = torch.softmax(pool_scores, dim=-1) * mask
        pool_weights = torch.where(
            has_parent,
            pool_weights / pool_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6),
            torch.zeros_like(pool_weights),
        )
        return torch.einsum('bts,bse->bte', pool_weights, variate_tokens)

    def _build_static_causal_residual(self, variate_tokens, device):
        if not self.graph_enable or self.graph_mode != 'static_causal_residual' or self.graph_residual_head is None:
            return None, None
        if not self.graph_use_static_bias:
            return None, None

        causal_pool = self._build_static_causal_pool(variate_tokens=variate_tokens, device=device)
        residual_input = torch.cat([variate_tokens, causal_pool], dim=-1)
        residual_tokens = self.graph_residual_head(self.graph_residual_norm(residual_input))
        residual_tokens = residual_tokens.permute(0, 2, 1)

        if self.graph_static_mix_mode == 'softmax':
            if self.graph_static_mix_head is None:
                raise ValueError('graph_static_mix_mode=softmax requires graph_static_mix_head to be initialized')
            mix_input = torch.cat([variate_tokens, causal_pool], dim=-1)
            mix_logits = self.graph_static_mix_head(self.graph_static_mix_norm(mix_input))
            mix_weights = torch.softmax(mix_logits, dim=-1)
            return residual_tokens, mix_weights

        return self._graph_residual_scale() * residual_tokens, None

    def _graph_lambda_gate(self, graph_lambda, batch_size, device):
        gate = graph_lambda.float().to(device).view(batch_size, 1, 1)
        if self.graph_lambda_gate_polarity == 'inverse':
            return 1.0 - gate
        if self.graph_lambda_gate_polarity == 'direct':
            return gate
        raise ValueError(f'Unsupported graph_lambda_gate_polarity: {self.graph_lambda_gate_polarity}')

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, regime_aux_enc=None, regime_aux_dec=None,
                 graph_lambda=None, graph_delta=None):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        if self.phasec_regime_mode == 'light_aux_input':
            if regime_aux_enc is None or regime_aux_dec is None:
                raise ValueError('Phase C regime light_aux_input requires both encoder and decoder auxiliary regime tensors')
            aux_enc = self.regime_aux_enc_embedding(regime_aux_enc.permute(0, 2, 1).float())
            aux_dec = self.regime_aux_dec_embedding(regime_aux_dec.permute(0, 2, 1).float())
            enc_out = torch.cat([enc_out, aux_enc, aux_dec], dim=1)
        graph_bias = self._build_graph_bias(
            batch_size=x_enc.shape[0],
            token_count=enc_out.shape[1],
            num_variates=N,
            device=enc_out.device,
            graph_lambda=graph_lambda,
            graph_delta=graph_delta,
        )
        graph_biases = [graph_bias] + [None] * (len(self.encoder.attn_layers) - 1) if graph_bias is not None else None
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None, graph_biases=graph_biases)

        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates
        if self.graph_mode == 'static_causal_residual':
            graph_residual, graph_mix_weights = self._build_static_causal_residual(
                variate_tokens=enc_out[:, :N, :],
                device=enc_out.device,
            )
            if graph_residual is not None:
                if graph_mix_weights is not None:
                    w_base = graph_mix_weights[..., 0].unsqueeze(1)
                    w_static = graph_mix_weights[..., 1].unsqueeze(1)
                    dec_out = w_base * dec_out + w_static * (dec_out + graph_residual)
                else:
                    dec_out = dec_out + graph_residual
        else:
            graph_residual = self._build_graph_residual(
                variate_tokens=enc_out[:, :N, :],
                device=enc_out.device,
                graph_lambda=graph_lambda,
                graph_delta=graph_delta,
            )
            if graph_residual is not None:
                dec_out = dec_out + graph_residual

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, attns


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, regime_aux_enc=None, regime_aux_dec=None,
                graph_lambda=None, graph_delta=None, mask=None):
        dec_out, attns = self.forecast(
            x_enc, x_mark_enc, x_dec, x_mark_dec,
            regime_aux_enc=regime_aux_enc, regime_aux_dec=regime_aux_dec,
            graph_lambda=graph_lambda, graph_delta=graph_delta,
        )
        
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
