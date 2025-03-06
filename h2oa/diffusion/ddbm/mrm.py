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

        # self.legacy = legacy
        # self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        # self.num_actions = num_actions
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.data_rep = data_rep
        # self.dataset = dataset

        # self.pose_rep = pose_rep
        # self.glob = glob
        # self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim
        self.use_latent = use_latent

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get('action_emb', None)

        self.input_feats = self.njoints * self.nfeats

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_feats = 5*3+2 if self.cond_mode == 'concat' else 0
        self.cond_dim = 128

        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        if not use_latent:
            self.input_process = InputProcess(self.data_rep, self.input_feats, self.latent_dim - self.cond_dim)
        if self.cond_mode == 'concat':
            self.cond_process = InputProcess(self.data_rep, self.cond_feats, self.cond_dim)
            # self.latent_dim += self.cond_feats
            # breakpoint()
        #     self.enc_A = identity
        #     self.enc_B = identity
        # else:

        #     self.enc_A = InputProcess(self.data_rep, 78+self.gru_emb_dim, self.latent_dim)
        #     self.enc_B = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)


        self.emb_trans_dec = emb_trans_dec
        self.debug=False

        if self.arch == 'trans_enc' or self.arch == 'trans':
            print("TRANS_ENC init")
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)

            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                         num_layers=self.num_layers)
        if self.arch == 'trans_dec' or self.arch == 'trans':
            print("TRANS_DEC init")
            seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=activation)
            self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                         num_layers=self.num_layers)
        if self.arch == 'gru':
            print("GRU init")
            self.gru = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)
        elif self.arch == 'debug':
            print("===============debug=============================")
            self.debug=True
            # raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')

        # self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        self.model_channels = self.latent_dim // 4
        if not self.debug:

            self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            self.embed_timestep = nn.Sequential(
                nn.Linear(self.model_channels, self.latent_dim),
                nn.SiLU(),
                nn.Linear(self.latent_dim, self.latent_dim),
            )

        if self.cond_mode != 'no_cond':
            if 'text' in self.cond_mode:
                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
                print('EMBED TEXT')
                print('Loading CLIP...')
                self.clip_version = clip_version
                self.clip_model = self.load_and_freeze_clip(clip_version)
            if 'action' in self.cond_mode:
                self.embed_action = EmbedAction(self.num_actions, self.latent_dim)
                print('EMBED ACTION')
            if 'smpl' in self.cond_mode:
                self.embed_text = nn.Linear(self.smpl_dim, self.latent_dim)
                print('EMBED SMPL')

        if not use_latent:
            # if self.cond_mode == 'concat':
            self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints)
            # else:
            #     self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints)
        #     self.dec_A = identity
        #     self.dec_B = identity
        # else:
        #     self.dec_A = OutputProcess(self.data_rep, 78, self.latent_dim, self.njoints,
        #                                         self.nfeats)
            
        #     self.dec_B = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                                # self.nfeats)
        # self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)

        import pytorch_kinematics as pk
        self.chain = pk.build_chain_from_urdf(open("./ddbm/h1/urdf/h1_add_hand_link_for_pk.urdf","rb").read())
        # human_node_names=['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
    def rot2xyz(self, root, jt):
        
        ret = self.chain.forward_kinematics(jt)
        # look up the transform for a specific link
        left_hand_link = ret['left_hand_link']
        left_hand_tg = left_hand_link.get_matrix()[:,:3,3]
        right_hand_link = ret['right_hand_link']
        right_hand_tg = right_hand_link.get_matrix()[:,:3,3]
        left_ankle_link = ret['left_ankle_link']
        left_ankle_tg = left_ankle_link.get_matrix()[:,:3,3]
        right_ankle_link = ret['right_ankle_link']
        right_ankle_tg = right_ankle_link.get_matrix()[:,:3,3]
        # get transform matrix (1,4,4), then convert to separate position and unit quaternion
    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]


    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond


    def forward_v0(self, x, timesteps, y=None, xT=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        # breakpoint()
        # bs, njoints, nfeats, nframes = x.shape
        bs, nframes, njoints  = x.shape
        # emb = self.embed_timestep(timesteps)  # [1, bs, d]
        # breakpoint()
        emb = self.embed_timestep(timestep_embedding(timesteps, self.model_channels)).unsqueeze(0)

        # force_mask = y.get('uncond', False)
        # if 'text' in self.cond_mode:
        #     if 'text_embed' in y.keys():  # caching option
        #         enc_text = y['text_embed']
        #     else:
        #         enc_text = self.encode_text(y['text'])
        #     emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))
        # if 'action' in self.cond_mode:
        #     action_emb = self.embed_action(y['action'])
        #     emb += self.mask_cond(action_emb, force_mask=force_mask)

        if self.arch == 'gru':
            x_reshaped = x.reshape(bs, njoints*nfeats, 1, nframes)
            emb_gru = emb.repeat(nframes, 1, 1)     #[#frames, bs, d]
            emb_gru = emb_gru.permute(1, 2, 0)      #[bs, d, #frames]
            emb_gru = emb_gru.reshape(bs, self.latent_dim, 1, nframes)  #[bs, d, 1, #frames]
            x = torch.cat((x_reshaped, emb_gru), axis=1)  #[bs, d+joints*feat, 1, #frames]

        x = x.type(self.dtype)
        x = self.input_process(x)

        if self.arch == 'trans_enc':
            # adding the timestep embed
            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        elif self.arch == 'trans_dec':
            if self.emb_trans_dec:
                xseq = torch.cat((emb, x), axis=0)
            else:
                xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            if self.emb_trans_dec:
                output = self.seqTransDecoder(tgt=xseq, memory=emb)[1:] # [seqlen, bs, d] # FIXME - maybe add a causal mask
            else:
                output = self.seqTransDecoder(tgt=xseq, memory=emb)
        elif self.arch == 'gru':
            xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen, bs, d]
            output, _ = self.gru(xseq)
        # breakpoint()
        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]

        return output
    

    def forward_v1(self, x, timesteps, xT=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        # bs, nframes, njoints  = x.shape
        if not self.debug:
            emb = self.embed_timestep(timestep_embedding(timesteps, self.model_channels)).unsqueeze(0)

        # breakpoint()
        x = x.type(self.dtype)
        if not self.use_latent:
            x = self.input_process(x)
            xT = self.input_process(xT)
        # if xT is not None:
        #     xT = xT.type(self.dtype)
        #     xT = self.input_process(xT)
        # else:
        x = x.permute((1, 0, 2))#.reshape(nframes, bs, njoints)
        xT = xT.permute((1, 0, 2))#.reshape(nframes, bs, njoints)

        if self.arch == 'trans_enc':
            # adding the timestep embed
            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            x = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        elif self.arch == 'trans_dec':
            if self.emb_trans_dec:
                xseq = torch.cat((emb, x), axis=0)
            else:
                xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            if self.emb_trans_dec:
                x = self.seqTransDecoder(tgt=xseq, memory=emb)[1:] # [seqlen, bs, d] # FIXME - maybe add a causal mask
            else:
                x = self.seqTransDecoder(tgt=xseq, memory=emb)
        
        elif self.arch == 'trans':

            xTseq = torch.cat((emb, xT), axis=0)  # [seqlen+1, bs, d]
            xTseq = self.sequence_pos_encoder(xTseq)  # [seqlen+1, bs, d]


            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            xseq = self.seqTransEncoder(xseq) # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
            x = self.seqTransDecoder(tgt=xTseq, memory=xseq)[1:] # [seqlen, bs, d] # FIXME - maybe add a causal mask
        
        if not self.use_latent:
            x = self.output_process(x)  # [bs, njoints, nfeats, nframes]
        # else:
        output = x.permute((1, 0, 2))#.reshape(nframes, bs, njoints)

        return output


    def forward(self, x, timesteps, xT=None, cond=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        # bs, nframes, njoints  = x.shape

        if not self.debug:
            emb = self.embed_timestep(timestep_embedding(timesteps, self.model_channels)).unsqueeze(0)

        # breakpoint()
        x = x.type(self.dtype)
        if not self.use_latent:
            x = self.input_process(x)
        x = x.permute((1, 0, 2))#.reshape(nframes, bs, njoints)
        # xT = xT.permute((1, 0, 2))#.reshape(nframes, bs, njoints)
            # xT = self.input_process(xT)
        if self.cond_mode == 'concat':
            cond = self.cond_process(cond)
            cond = cond.permute((1, 0, 2))#.reshape(nframes, bs, njoints)
            x= torch.cat((x, cond), axis=2)
        # if xT is not None:
        #     xT = xT.type(self.dtype)
        #     xT = self.input_process(xT)
        # else:

        if self.arch == 'trans_enc':
            # adding the timestep embed
            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            x = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        elif self.arch == 'trans_dec':
            if self.emb_trans_dec:
                xseq = torch.cat((emb, x), axis=0)
            else:
                xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            if self.emb_trans_dec:
                x = self.seqTransDecoder(tgt=xseq, memory=emb)[1:] # [seqlen, bs, d] # FIXME - maybe add a causal mask
            else:
                x = self.seqTransDecoder(tgt=xseq, memory=emb)

        elif self.arch == 'trans':

            xTseq = torch.cat((emb, xT), axis=0)  # [seqlen+1, bs, d]
            xTseq = self.sequence_pos_encoder(xTseq)  # [seqlen+1, bs, d]


            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            xseq = self.seqTransEncoder(xseq) # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
            x = self.seqTransDecoder(tgt=xTseq, memory=xseq)[1:] # [seqlen, bs, d] # FIXME - maybe add a causal mask
        
        if not self.use_latent:
            x = self.output_process(x)  # [bs, njoints, nfeats, nframes]
        # else:
        output = x.permute((1, 0, 2))#.reshape(nframes, bs, njoints)

        return output

class ContrastiveModel(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                use_fp16=False, use_latent = False, 
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', clip_dim=512,# dataset='amass',
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, **kargs):
        super().__init__()

        # self.legacy = legacy
        # self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        # self.num_actions = num_actions
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.data_rep = data_rep
        # self.dataset = dataset

        # self.pose_rep = pose_rep
        # self.glob = glob
        # self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim
        self.use_latent = use_latent

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get('action_emb', None)

        self.input_feats = self.njoints * self.nfeats

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.arch = arch
        self.human_dim=23*6+9
        self.h1_dim=19+9

        self.enc_A = InputProcess(self.data_rep, self.human_dim, self.latent_dim)
        self.enc_B = InputProcess(self.data_rep, self.h1_dim, self.latent_dim)


        self.emb_trans_dec = emb_trans_dec

        # self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        self.model_channels = self.latent_dim // 4


        self.dec_A = OutputProcess(self.data_rep, self.human_dim, self.latent_dim, self.njoints,
                                            self.nfeats)
        
        self.dec_B = OutputProcess(self.data_rep, self.h1_dim, self.latent_dim, self.njoints,
                                                self.nfeats)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)

    # def forward(self, xT, x0):
    #     L_A = self.enc_A(xT)
    #     L_B = self.enc_B(x0)

    #     xT_recon = self.dec_A(L_A)
    #     x0_recon = self.dec_B(L_B)

    #     # normalized features

    #     # L_A=L_A.squeeze(1)
    #     # L_B=L_B.squeeze(1)
    #     # cosine similarity as logits
    #     logit_scale = self.logit_scale.exp()
    #     logits = logit_scale * F.cosine_similarity(L_A, L_B.permute(1,0,2), dim=-1)
    #     # L_A = L_A / L_A.norm(dim=-1, keepdim=True)
    #     # L_B = L_B / L_B.norm(dim=-1, keepdim=True)
    #     # logits = logit_scale * L_A.squeeze(1) @ L_B.squeeze(1).T
    #     labels = torch.arange(len(xT), device=xT.device)
    #     loss_A_ce = F.cross_entropy(logits, labels)
    #     loss_B_ce = F.cross_entropy(logits.t(), labels)
    #     loss_A_re = F.mse_loss(xT_recon, xT)
    #     loss_B_re = F.mse_loss(x0_recon, x0)

    #     # labels = np.arange(n)
    #     # loss_i = cross_entropy_loss(logits, labels, axis=0)
    #     # loss_t = cross_entropy_loss(logits, labels, axis=1)
    #     # breakpoint()
    #     return loss_A_ce, loss_B_ce, loss_A_re, loss_B_re
    
    def forward(self, human, retarget, recycle):
        L_A = self.enc_A(human)
        L_0 = self.enc_B(recycle)
        L_T = self.enc_B(retarget)

        human_recon = self.dec_A(L_A)
        retarget_recon = self.dec_B(L_T)
        recycle_recon = self.dec_B(L_0)

        # normalized features

        # L_A=L_A.squeeze(1)
        # L_B=L_B.squeeze(1)
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * F.cosine_similarity(L_A, L_T.permute(1,0,2), dim=-1)
        # L_A = L_A / L_A.norm(dim=-1, keepdim=True)
        # L_B = L_B / L_B.norm(dim=-1, keepdim=True)
        # logits = logit_scale * L_A.squeeze(1) @ L_B.squeeze(1).T
        labels = torch.arange(len(human), device=human.device)
        loss_A_ce = F.cross_entropy(logits, labels)
        loss_T_ce = F.cross_entropy(logits.t(), labels)
        loss_A_re = F.mse_loss(human_recon, human)
        loss_T_re = F.mse_loss(retarget_recon, retarget)
        loss_0_re = F.mse_loss(recycle_recon, recycle)

        # labels = np.arange(n)
        # loss_i = cross_entropy_loss(logits, labels, axis=0)
        # loss_t = cross_entropy_loss(logits, labels, axis=1)
        # breakpoint()


        return loss_A_ce, loss_T_ce, loss_A_re, loss_T_re, loss_0_re


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


# class TimestepEmbedder(nn.Module):
    # def __init__(self, latent_dim, sequence_pos_encoder):
    #     super().__init__()
    #     self.latent_dim = latent_dim
    #     self.sequence_pos_encoder = sequence_pos_encoder

    #     time_embed_dim = self.latent_dim
    #     self.time_embed = nn.Sequential(
    #         nn.Linear(self.latent_dim, time_embed_dim),
    #         nn.SiLU(),
    #         nn.Linear(time_embed_dim, time_embed_dim),
    #     )

    # def forward(self, timesteps):
    #     return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)

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
        elif self.data_rep == 'rot_vel':
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError


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
        elif self.data_rep == 'rot_vel':
            first_pose = output[[0]]  # [1, bs, d]
            first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
            vel = output[1:]  # [seqlen-1, bs, d]
            vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
            output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        else:
            raise ValueError
        # output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        # output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        # output = output.permute(1, 0, 2)  # [bs, nframes, d]
        return output


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output

