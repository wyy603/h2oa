import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from .fp16_util import convert_module_to_f16, convert_module_to_f32

def identity(x):
    return x


class MRM(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                use_fp16=False, use_latent = False, 
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', clip_dim=512,# dataset='amass',
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, **kargs):
        super().__init__()


        self.njoints = njoints
        self.nfeats = nfeats
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.data_rep = data_rep


        self.latent_dim = latent_dim
        self.use_latent = use_latent

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get('action_emb', None)

        self.input_feats = self.njoints * self.nfeats

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_feats = 5*3+2 if self.cond_mode == 'concat' else 0
        self.cond_dim = 128

        self.arch = arch

        self.input_process = InputProcess(self.data_rep, self.input_feats, self.latent_dim - self.cond_dim)
        self.cond_process = InputProcess(self.data_rep, self.cond_feats, self.cond_dim)
        self.emb_trans_dec = emb_trans_dec

        print("TRANS_ENC init")
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=256,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                         num_layers=self.num_layers)
        self.model_channels = self.latent_dim // 4

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.embed_timestep = nn.Sequential(
            nn.Linear(self.model_channels, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )


        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints)



    def forward(self, x, timesteps, xT=None, cond=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        # bs, nframes, njoints  = x.shape

        emb = self.embed_timestep(timestep_embedding(timesteps, self.model_channels)).unsqueeze(0)

        # breakpoint()
        x = self.input_process(x)
        
        cond = self.cond_process(cond)
        cond = cond.permute((1, 0, 2))#.reshape(nframes, bs, njoints)
        x= torch.cat((x, cond), axis=2)


        if self.arch == 'trans_enc':
            # adding the timestep embed
            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            x = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        output = self.output_process(x)  # [bs, njoints, nfeats, nframes]
        # else:

        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model  ?????????????
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None] #* (2 * math.pi)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)
            

    def forward(self, x):
        # bs, nframes, njoints  = x.shape
        # bs, njoints, nfeats, nframes = x.shape

        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats=1):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = self.poseFinal(output)  # [seqlen, bs, 150]
        return output


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output

