import copy
import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder

from typing import Optional, Union, Callable
from .attention import AttentionLayer, ProbAttention, FullAttention
from .decoder import DecoderLayer, PBEEDecoder
from .encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack, PBEEncoder
from .utils import get_sequence_list

import uuid


def _get_clones(mod, n):
    return nn.ModuleList([copy.deepcopy(mod) for _ in range(n)])


class FeatureIsolatedTransformer(nn.Transformer):
    def __init__(self, d_model_list: list, nhead_list: list, num_encoder_layers: int, num_decoder_layers: int,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 selected_attn: str = 'prob', output_attention: str = True,
                 inner_classifiers_config: list = None, patience: int = 1, use_pyramid_encoder: bool = False,
                 distil: bool = False, projections_config: list = None,
                 IA_encoder: bool = False, IA_decoder: bool = False):

        super(FeatureIsolatedTransformer, self).__init__(sum(d_model_list), nhead_list[-1], num_encoder_layers,
                                                         num_decoder_layers, dim_feedforward, dropout, activation)
        del self.encoder
        self.d_model = sum(d_model_list)
        self.d_ff = dim_feedforward
        self.dropout = dropout
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.use_pyramid_encoder = use_pyramid_encoder
        self.use_IA_encoder = IA_encoder
        self.use_IA_decoder = IA_decoder
        self.inner_classifiers_config = inner_classifiers_config
        self.projections_config = projections_config
        self.patience = patience
        self.distil = distil
        self.activation = activation
        self.selected_attn = selected_attn
        self.output_attention = output_attention
        self.l_hand_encoder = self.get_custom_encoder(d_model_list[0], nhead_list[0])
        self.r_hand_encoder = self.get_custom_encoder(d_model_list[1], nhead_list[1])
        self.body_encoder = self.get_custom_encoder(d_model_list[2], nhead_list[2])
        self.decoder = self.get_custom_decoder(nhead_list[-1])
        self._reset_parameters()

    def get_custom_encoder(self, f_d_model: int, nhead: int):
        Attn = ProbAttention if self.selected_attn == 'prob' else FullAttention
        print(f'self.selected_attn {self.selected_attn}')

        if self.use_pyramid_encoder:
            print("Pyramid encoder")
            print(f'self.distl {self.distil}')
            e_layers = get_sequence_list(self.num_encoder_layers)
            inp_lens = list(range(len(e_layers)))
            encoders = [
                Encoder(
                    [
                        EncoderLayer(
                            AttentionLayer(
                                Attn(output_attention=self.output_attention),
                                f_d_model, nhead, mix=False),
                            f_d_model,
                            self.d_ff,
                            dropout=self.dropout,
                            activation=self.activation
                        ) for _ in range(el)
                    ],
                    [
                        ConvLayer(
                            f_d_model
                        ) for _ in range(self.num_encoder_layers - 1)
                    ] if self.distil else None,
                    norm_layer=torch.nn.LayerNorm(f_d_model)
                ) for el in e_layers]

            encoder = EncoderStack(encoders, inp_lens)
        else:
            encoder_layer = TransformerEncoderLayer(f_d_model, nhead, self.d_ff, self.dropout, self.activation)
            encoder_layer.self_attn = AttentionLayer(
                Attn(output_attention=self.output_attention),
                f_d_model, nhead, mix=False
            )
            encoder_norm = LayerNorm(f_d_model)

            if self.use_IA_encoder:
                print("Encoder with input adaptive")
                self.inner_classifiers_config[0] = f_d_model
                encoder = PBEEncoder(
                    encoder_layer, self.num_encoder_layers, norm=encoder_norm,
                    inner_classifiers_config=self.inner_classifiers_config,
                    projections_config=self.projections_config,
                    patience=self.patience
                )
            else:
                print("Normal encoder")
                encoder = TransformerEncoder(encoder_layer, self.num_encoder_layers, norm=encoder_norm)

        return encoder

    def get_custom_decoder(self, nhead):
        decoder_layer = DecoderLayer(self.d_model, nhead, self.d_ff)
        decoder_norm = LayerNorm(self.d_model)
        if self.use_IA_decoder:
            print("Decoder with with input adaptive")
            return PBEEDecoder(
                decoder_layer, self.num_decoder_layers, norm=decoder_norm,
                inner_classifiers_config=self.inner_classifiers_config, patient=self.patience
            )
        else:
            print("Normal decoder")
            return TransformerDecoder(
                decoder_layer, self.num_decoder_layers, norm=decoder_norm)

    def checker(self, full_src, tgt, is_batched):
        if not self.batch_first and full_src.size(1) != tgt.size(1) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        elif self.batch_first and full_src.size(0) != tgt.size(0) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        if full_src.size(-1) != self.d_model or tgt.size(-1) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

    def forward(self, src: list, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None,
                src_is_causal: Optional[bool] = None, tgt_is_causal: Optional[bool] = None,
                memory_is_causal: bool = False) -> Tensor:

        full_src = torch.cat(src, dim=-1)
        self.checker(full_src, tgt, full_src.dim() == 3)

        id = uuid.uuid1()
        # code for concurrency is removed...
        if self.use_IA_encoder:
            l_hand_memory = self.l_hand_encoder(src[0], mask=src_mask, src_key_padding_mask=src_key_padding_mask)
            r_hand_memory = self.r_hand_encoder(src[1], mask=src_mask, src_key_padding_mask=src_key_padding_mask)
            body_memory = self.body_encoder(src[2], mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        else:
            l_hand_memory = self.l_hand_encoder(src[0], mask=src_mask, src_key_padding_mask=src_key_padding_mask)
            r_hand_memory = self.r_hand_encoder(src[1], mask=src_mask, src_key_padding_mask=src_key_padding_mask)
            body_memory = self.body_encoder(src[2], mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        full_memory = torch.cat((l_hand_memory, r_hand_memory, body_memory), -1)

        if self.use_IA_decoder:
            output = self.decoder(tgt, full_memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask)
        else:
            output = self.decoder(tgt, full_memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask)

        return output


class AbsolutePE(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(AbsolutePE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin((position * div_term)*(d_model/max_len))
        pe[:, 1::2] = torch.cos((position * div_term)*(d_model/max_len))
        pe = scale_factor * pe.unsqueeze(0)
        # reshape the matrix shape to met the input shape
        # pe = pe.squeeze(0).unsqueeze(1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)




# class SiFormer(nn.Module):
#     """
#     Implementation of the SPOTER (Sign POse-based TransformER) architecture for sign language recognition from sequence
#     of skeletal data.
#     """

#     def __init__(self, num_classes, attn_type='prob', num_enc_layers=3, num_dec_layers=2, patience=1,
#                  seq_len=204, IA_encoder = True, IA_decoder = False, num_channels=3, num_hand_landmarks=21, num_body_landmarks=7, nhead_list=None):
#         super(SiFormer, self).__init__()
#         print("Feature isolated transformer")
#         # self.feature_extractor = FeatureExtractor(num_hid=108, kernel_size=7)
#         hand_d_model = num_hand_landmarks * num_channels
#         body_d_model = num_body_landmarks * num_channels
#         num_hid = hand_d_model*2 + body_d_model
#         if nhead_list is None:
#             nhead_list = [3, 3, 3, 7]

#         self.l_hand_embedding = nn.Parameter(self.get_encoding_table(d_model=hand_d_model, seq_len=seq_len))
#         self.r_hand_embedding = nn.Parameter(self.get_encoding_table(d_model=hand_d_model, seq_len=seq_len))
#         self.body_embedding = nn.Parameter(self.get_encoding_table(d_model=body_d_model, seq_len=seq_len))

#         self.class_query = nn.Parameter(torch.rand(1, 1, num_hid))
#         self.transformer = FeatureIsolatedTransformer(
#             [hand_d_model, hand_d_model, body_d_model], nhead_list, num_encoder_layers=num_enc_layers, num_decoder_layers=num_dec_layers,
#             selected_attn=attn_type, IA_encoder=IA_encoder, IA_decoder=IA_decoder,
#             inner_classifiers_config=[num_hid, num_classes], projections_config=[seq_len, 1], 
#             patience=patience, use_pyramid_encoder=False, distil=False
#         )
#         print(f"num_enc_layers {num_enc_layers}, num_dec_layers {num_dec_layers}, patient {patience}")
#         self.projection = nn.Linear(num_hid, num_classes)

#     def forward(self, l_hand, r_hand, body, training):
#         batch_size = l_hand.size(0)
        
#         new_l_hand = l_hand.view(l_hand.size(0), l_hand.size(1), l_hand.size(2) * l_hand.size(3))
#         new_r_hand = r_hand.view(r_hand.size(0), r_hand.size(1), r_hand.size(2) * r_hand.size(3))
#         body = body.view(body.size(0), body.size(1), body.size(2) * body.size(3))


#         new_l_hand = new_l_hand.permute(1, 0, 2).type(dtype=torch.float32)
#         new_r_hand = new_r_hand.permute(1, 0, 2).type(dtype=torch.float32)
#         new_body = body.permute(1, 0, 2).type(dtype=torch.float32)

#         l_hand_in = new_l_hand + self.l_hand_embedding  # Shape remains the same
#         r_hand_in = new_r_hand + self.r_hand_embedding  # Shape remains the same
#         body_in = new_body + self.body_embedding  # Shape remains the same
        
#         transformer_output = self.transformer(
#             [l_hand_in, r_hand_in, body_in], self.class_query.repeat(1, batch_size, 1), training=training
#         ).transpose(0, 1)
        
#         out = self.projection(transformer_output).squeeze()
#         return out

#     @staticmethod
#     def get_encoding_table(d_model=108, seq_len=204):
#         torch.manual_seed(42)
#         tensor_shape = (seq_len, d_model)
#         frame_pos = torch.rand(tensor_shape)
#         for i in range(tensor_shape[0]):
#             for j in range(1, tensor_shape[1]):
#                 frame_pos[i, j] = frame_pos[i, j - 1]
#         frame_pos = frame_pos.unsqueeze(1)  # (seq_len, 1, feature_size): (204, 1, 108)
#         return frame_pos




# class SiFormerMeta(nn.Module):
#     """
#     Implementation of the SPOTER (Sign POse-based TransformER) architecture for sign language recognition from sequence
#     of skeletal data.
#     """

#     def __init__(self, emb_dim, attn_type='prob', num_enc_layers=3, num_dec_layers=2, patience=1,
#                  seq_len=204, IA_encoder = True, IA_decoder = False, num_channels=3, num_hand_landmarks=21, num_body_landmarks=7, nhead_list=None):
#         super(SiFormerMeta, self).__init__()
#         print("Feature isolated transformer")
#         hand_d_model = num_hand_landmarks * num_channels
#         body_d_model = num_body_landmarks * num_channels
#         num_hid = hand_d_model*2 + body_d_model
#         if nhead_list is None:
#             nhead_list = [3, 3, 3, 7]

#         self.l_hand_embedding = nn.Parameter(self.get_encoding_table(d_model=hand_d_model, seq_len=seq_len))
#         self.r_hand_embedding = nn.Parameter(self.get_encoding_table(d_model=hand_d_model, seq_len=seq_len))
#         self.body_embedding = nn.Parameter(self.get_encoding_table(d_model=body_d_model, seq_len=seq_len))

#         self.class_query = nn.Parameter(torch.rand(1, 1, num_hid))
#         self.transformer = FeatureIsolatedTransformer(
#             [hand_d_model, hand_d_model, body_d_model], nhead_list, num_encoder_layers=num_enc_layers, num_decoder_layers=num_dec_layers,
#             selected_attn=attn_type, IA_encoder=IA_encoder, IA_decoder=IA_decoder,
#             inner_classifiers_config=[num_hid, emb_dim], projections_config=[seq_len, 1], 
#             patience=patience, use_pyramid_encoder=False, distil=False
#         )
#         print(f"num_enc_layers {num_enc_layers}, num_dec_layers {num_dec_layers}, patient {patience}")
#         self.projection = nn.Linear(num_hid, emb_dim)

#     def forward(self, l_hand, r_hand, body, training=True):
#         batch_size = l_hand.size(0)
        
#         new_l_hand = l_hand.view(l_hand.size(0), l_hand.size(1), l_hand.size(2) * l_hand.size(3))
#         new_r_hand = r_hand.view(r_hand.size(0), r_hand.size(1), r_hand.size(2) * r_hand.size(3))
#         body = body.view(body.size(0), body.size(1), body.size(2) * body.size(3))


#         new_l_hand = new_l_hand.permute(1, 0, 2).type(dtype=torch.float32)
#         new_r_hand = new_r_hand.permute(1, 0, 2).type(dtype=torch.float32)
#         new_body = body.permute(1, 0, 2).type(dtype=torch.float32)

#         l_hand_in = new_l_hand + self.l_hand_embedding  # Shape remains the same
#         r_hand_in = new_r_hand + self.r_hand_embedding  # Shape remains the same
#         body_in = new_body + self.body_embedding  # Shape remains the same
        
#         transformer_output = self.transformer(
#             [l_hand_in, r_hand_in, body_in], self.class_query.repeat(1, batch_size, 1), training=training
#         ).transpose(0, 1)
        
#         out = self.projection(transformer_output).squeeze()
#         return out

#     @staticmethod
#     def get_encoding_table(d_model=108, seq_len=204):
#         torch.manual_seed(42)
#         tensor_shape = (seq_len, d_model)
#         frame_pos = torch.rand(tensor_shape)
#         for i in range(tensor_shape[0]):
#             for j in range(1, tensor_shape[1]):
#                 frame_pos[i, j] = frame_pos[i, j - 1]
#         frame_pos = frame_pos.unsqueeze(1)  # (seq_len, 1, feature_size): (204, 1, 108)
#         return frame_pos


class SiFormerEmbedder(nn.Module):
    """
    Implementation of the SPOTER (Sign POse-based TransformER) architecture for sign language recognition from sequence
    of skeletal data.
    """

    def __init__(self, emb_dim, attn_type='prob', num_enc_layers=3, num_dec_layers=2, patience=1, n_head=1,
                 seq_len=204, IA_encoder = True, IA_decoder = False, num_channels=3, num_hand_landmarks=21, num_body_landmarks=7, nhead_list=None):
        super(SiFormerEmbedder, self).__init__()
        print("Feature isolated transformer")
        hand_d_model = num_hand_landmarks * num_channels
        body_d_model = num_body_landmarks * num_channels
        num_hid = hand_d_model*2 + body_d_model
        if nhead_list is None:
            nhead_list = [3, 3, 3, 7]

        self.l_hand_embedding = nn.Parameter(self.get_encoding_table(d_model=hand_d_model, seq_len=seq_len))
        self.r_hand_embedding = nn.Parameter(self.get_encoding_table(d_model=hand_d_model, seq_len=seq_len))
        self.body_embedding = nn.Parameter(self.get_encoding_table(d_model=body_d_model, seq_len=seq_len))

        self.n_head = n_head
        self.class_queries = nn.Parameter(torch.rand(self.n_head, 1, num_hid))
        self.transformer = FeatureIsolatedTransformer(
            [hand_d_model, hand_d_model, body_d_model], nhead_list, num_encoder_layers=num_enc_layers, num_decoder_layers=num_dec_layers,
            selected_attn=attn_type, IA_encoder=IA_encoder, IA_decoder=IA_decoder,
            inner_classifiers_config=[num_hid, emb_dim], projections_config=[seq_len, 1], 
            patience=patience, use_pyramid_encoder=False, distil=False
        )
        print(f"num_enc_layers {num_enc_layers}, num_dec_layers {num_dec_layers}, patient {patience}")

        

    def forward(self, l_hand, r_hand, body):
        batch_size = l_hand.size(0)

        # Flatten landmark dimension: (B, T, landmarks, C) -> (B, T, landmarks*C)
        new_l_hand = l_hand.view(batch_size, l_hand.size(1), -1)
        new_r_hand = r_hand.view(batch_size, r_hand.size(1), -1)
        new_body = body.view(batch_size, body.size(1), -1)

        # Padding mask creation: (B, T)
        # A timestep is considered "padded" if all its features are zero
        l_hand_mask = (new_l_hand.abs().sum(dim=-1) == 0)  # True where padded
        r_hand_mask = (new_r_hand.abs().sum(dim=-1) == 0)
        body_mask   = (new_body.abs().sum(dim=-1) == 0)

        # Merge masks (logical AND): only mark as padded if ALL modalities are padded at that time step
        src_key_padding_mask = l_hand_mask & r_hand_mask & body_mask  # (B, T)

        # Permute to (T, B, F) for transformer
        new_l_hand = new_l_hand.permute(1, 0, 2).float()
        new_r_hand = new_r_hand.permute(1, 0, 2).float()
        new_body = new_body.permute(1, 0, 2).float()

        # Add learned positional embeddings
        l_hand_in = new_l_hand + self.l_hand_embedding
        r_hand_in = new_r_hand + self.r_hand_embedding
        body_in = new_body + self.body_embedding

        repeated_queries = self.class_queries.repeat(1, batch_size, 1)

        out = self.transformer(
            [l_hand_in, r_hand_in, body_in], repeated_queries, src_key_padding_mask=src_key_padding_mask  # (B, T) mask
        ).transpose(0, 1)  # (B, n_head, hidden_dim)

        return out

    @staticmethod
    def get_encoding_table(d_model=108, seq_len=204):
        torch.manual_seed(42)
        tensor_shape = (seq_len, d_model)
        frame_pos = torch.rand(tensor_shape)
        for i in range(tensor_shape[0]):
            for j in range(1, tensor_shape[1]):
                frame_pos[i, j] = frame_pos[i, j - 1]
        frame_pos = frame_pos.unsqueeze(1)  # (seq_len, 1, feature_size): (204, 1, 108)
        return frame_pos


class MetaSign(nn.Module):
    def __init__(self, emb_dim, attn_type='prob', num_enc_layers=3, num_dec_layers=2, patience=1,
                 seq_len=204, IA_encoder = True, IA_decoder = False, num_channels=3, num_hand_landmarks=21, num_body_landmarks=7, nhead_list=None):
        super(MetaSign, self).__init__()

        self.siformer = SiFormerEmbedder(emb_dim=emb_dim, attn_type=attn_type, num_enc_layers=num_enc_layers, num_dec_layers=num_dec_layers, patience=patience,
                                         n_head=1, seq_len=seq_len, IA_encoder=IA_encoder, IA_decoder=IA_decoder, num_channels=num_channels, num_hand_landmarks=num_hand_landmarks, num_body_landmarks=num_body_landmarks, nhead_list=nhead_list)
        hand_d_model = num_hand_landmarks * num_channels
        body_d_model = num_body_landmarks * num_channels
        num_hid = hand_d_model*2 + body_d_model

        self.single_class_projection = nn.Sequential(
            nn.Linear(num_hid, num_hid),
            nn.Linear(num_hid, emb_dim),
            nn.LayerNorm(emb_dim)
        )

        self.live_projection = nn.Sequential(
            nn.Linear(num_hid, num_hid),
            nn.Linear(num_hid, emb_dim),
            nn.LayerNorm(emb_dim)
        )

    def forward(self, l_hand, r_hand, body):
        emb = self.siformer(l_hand, r_hand, body)
        out = self.live_projection(emb)
        return out
    
    def get_class_embedding(self, l_hand, r_hand, body):
        emb = self.siformer(l_hand, r_hand, body)
        out = self.single_class_projection(emb)
        return out



class MetaLearning(nn.Module):
    """
    Custom meta-learning framework in sign language recognition.
    """
    def __init__(self, backbone, n_support = 3, n_way = 10):
        super(MetaLearning, self).__init__()
        self.backbone = backbone
        self.n_support = n_support
        self.n_way = n_way
        self.temperature = 0.1
    

    def forward(self, l_support, r_support, b_support, l_query, r_query, b_query):
        support_embeddings = self.backbone.get_class_embedding(l_support, r_support, b_support).view(self.n_way, self.n_support, -1)
        emb_dim = support_embeddings.shape[-1]
        query_embeddings = self.backbone(l_query, r_query, b_query).view(-1, emb_dim)

        return support_embeddings, query_embeddings #(n_way, n_support, emb_dim), (batch_size, emb_dim)

    def get_prototypes(self, support_embeddings):
        prototypes = support_embeddings.mean(dim=1)  # (n_way, emb_dim)
        return prototypes
    
    def calculate_contrastive_loss(self, prototypes, support_embeddings, margin=1.0):
        """
        Class-center-based contrastive loss.
        Args:
            support_embeddings: Tensor of shape (n_way, n_support, emb_dim)
            margin: Float. Margin for inter-class contrastive loss.
        Returns:
            Scalar tensor: contrastive loss
        """
        n_way, n_support, emb_dim = support_embeddings.shape
        device = support_embeddings.device

        # Intra-class loss: distance from each support sample to its class center
        expanded_centers = prototypes.unsqueeze(1).expand(-1, n_support, -1)  # (n_way, n_support, emb_dim)
        intra_dists = F.pairwise_distance(
            support_embeddings.contiguous().view(-1, emb_dim),
            expanded_centers.contiguous().view(-1, emb_dim)
        )
        intra_loss = intra_dists.mean()

        # Inter-class loss: margin-based contrast between class centers
        pdist_centers = torch.cdist(prototypes, prototypes, p=2)  # (n_way, n_way)
        mask = ~torch.eye(n_way, dtype=torch.bool, device=device)
        inter_dists = pdist_centers[mask]  # exclude diagonal (same class)
        inter_loss = F.relu(margin - inter_dists).mean()

        return intra_loss + inter_loss
    
    
    def calculate_prediction_loss(self, prototypes, query_embeddings, query_labels):
        # prototypes: (n_way, emb_dim)
        # query_embeddings: (batch_size, emb_dim)
        # query_labels: (batch_size,)
        # Compute distances: (batch_size, n_way)
        dists = torch.cdist(query_embeddings, prototypes)  # (batch_size, n_way)
        loss = F.cross_entropy(-dists, query_labels)
        return loss
    
    def calculate_loss(self, support_embeddings, query_embeddings, query_labels, alpha=1.0):
        prototypes = self.get_prototypes(support_embeddings)
        contrastive_loss = self.calculate_contrastive_loss(prototypes, support_embeddings)
        prediction_loss = self.calculate_prediction_loss(prototypes, query_embeddings, query_labels)
        return prediction_loss + alpha * contrastive_loss

    @torch.no_grad()
    def calculate_accuracy(self, support_embeddings, query_embeddings, query_labels):
        prototypes = self.get_prototypes(support_embeddings)
        # Compute pairwise distances (Euclidean)
        dists = torch.cdist(query_embeddings, prototypes)  # (batch_size, n_way)

        # Convert distances to logits by negating (closer distance = higher logit)
        logits = -dists / self.temperature

        # Predicted class = argmax over logits
        preds = torch.argmax(logits, dim=1)

        # Compare to ground truth labels
        correct = (preds == query_labels).sum().item()
        total = query_labels.size(0)
        
        return total, correct
    
    def calculate_loss_and_accuracy(self, support_embeddings, query_embeddings, query_labels, alpha=0.3):
        prototypes = self.get_prototypes(support_embeddings)
        contrastive_loss = self.calculate_contrastive_loss(prototypes, support_embeddings)
        prediction_loss = self.calculate_prediction_loss(prototypes, query_embeddings, query_labels)
        loss = prediction_loss + alpha * contrastive_loss
        with torch.no_grad():
            dists = torch.cdist(query_embeddings, prototypes)
            logits = -dists / self.temperature
            preds = torch.argmax(logits, dim=1)
            correct = (preds == query_labels).sum().item()
            total = query_labels.size(0)
        return loss, total, correct
