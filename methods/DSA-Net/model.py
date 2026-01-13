import torch.utils.checkpoint

from transformers import BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from .SubNets.dynamicfc import DynamicLayer
from .AlignNets import AlignSubNet
from .PeepholeLSTM import BiPeepholeLSTMLayer

import dgl
from dgl.nn.pytorch import GATConv

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# --- DALR 辅助函数与类 (新增) ---

def prepare_data(image: torch.Tensor, label: torch.Tensor):
    # 确保是 1D 索引
    if isinstance(label, torch.Tensor):
        label_cpu = label.detach().cpu().tolist()
    else:
        label_cpu = list(label)

    nr_index = [i for i, _ in enumerate(label_cpu)]
    if len(nr_index) < 2:
        nr_index.append(
            np.random.randint(len(label_cpu))
        )
        nr_index.append(
            np.random.randint(len(label_cpu))
        )

    image_nr = image[nr_index]
    matched_image = image_nr.clone()
    unmatched_image = image_nr.clone().roll(shifts=5, dims=0)

    return matched_image, unmatched_image


class MLPLayer(nn.Module):
    """简单的 MLP 投影头"""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.dense = nn.Linear(in_dim, out_dim)
        self.activation = nn.Tanh()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.dense(features)
        x = self.activation(x)
        return x


class Similarity(nn.Module):
    """带温度系数的余弦相似度"""

    def __init__(self, temp: float):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.cos(x, y) / self.temp


class ArcSimilarity(nn.Module):
    """
    ArcFace 风格的相似度：在角度上增加 Margin，增强判别性。
    """

    def __init__(self, temp: float, margin: float = 0.1):  # 推荐 margin 0.1 - 0.5
        super().__init__()
        self.temp = temp
        self.margin = margin
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, D], y: [1, B, D]
        cos_sim = self.cos(x, y).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cos_sim)
        theta_m = theta + self.margin  # 加 margin (让 theta 变大，cos 变小，增加难度)
        cos_m = torch.cos(theta_m)
        return cos_m / self.temp


class ConsistencySimilarityModule(nn.Module):
    """一致性学习模块：判断两个模态特征是否来自同一个样本"""

    def __init__(self, input_dim: int, sim_dim: int = 256):
        super().__init__()
        self.text_aligner = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU(),
        )
        self.other_aligner = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU(),
        )
        self.sim_classifier_dim = sim_dim * 2
        self.sim_classifier = nn.Sequential(
            nn.BatchNorm1d(self.sim_classifier_dim),
            nn.Linear(self.sim_classifier_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, text: torch.Tensor, other: torch.Tensor):
        text_aligned = self.text_aligner(text)
        other_aligned = self.other_aligner(other)
        sim_feature = torch.cat([text_aligned, other_aligned], dim=1)
        pred_similarity = self.sim_classifier(sim_feature)
        return text_aligned, other_aligned, pred_similarity


class KLContrastiveSimLoss(nn.Module):
    """KL 散度损失：用于对齐相似度分布"""

    def __init__(self, tau: float = 0.5):
        super().__init__()
        self.tau = tau

    def forward(self, logits: torch.Tensor, softlabel: torch.Tensor, tau: float = 0.5,
                softlabel_tau: float = 0.5) -> torch.Tensor:
        # softlabel 是目标分布 (Teacher/Structure), logits 是预测分布
        sim_targets = F.softmax(softlabel / softlabel_tau, dim=1)
        logit_inputs = F.log_softmax(logits / tau, dim=1)
        loss = F.kl_div(logit_inputs, sim_targets, reduction="batchmean")
        return loss




# --- 2. 语义级注意力 (用于 HAN 融合) ---
class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)
        return (beta * z).sum(1)


class HANLayer(nn.Module):
    def __init__(self, in_size, out_size, num_heads=4, dropout=0.2):
        super(HANLayer, self).__init__()
        self.meta_paths = ['spk', 'rep', 'self']  # 对应 build_rs_graph 中的边
        self.gat_layers = nn.ModuleList()
        for _ in self.meta_paths:
            self.gat_layers.append(
                GATConv(in_size, out_size, num_heads, dropout, allow_zero_in_degree=True)
            )
        self.semantic_attention = SemanticAttention(in_size=out_size * num_heads)

    def forward(self, g, h):
        semantic_embeddings = []
        for i, mp in enumerate(self.meta_paths):
            if ('utt', mp, 'utt') in g.canonical_etypes:
                feat = self.gat_layers[i](g[mp], h).flatten(1)
                semantic_embeddings.append(feat)
            else:
                semantic_embeddings.append(torch.zeros_like(semantic_embeddings[0]))

        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)
        return self.semantic_attention(semantic_embeddings)


# --- 4. 融合门控 ---
class NewFusionGate(nn.Module):
    def __init__(self, hid_size):
        super(NewFusionGate, self).__init__()
        self.fuse = nn.Linear(hid_size * 2, hid_size)

    def forward(self, a, b):
        concat_ab = torch.cat([a, b], dim=-1)
        fusion_coef = torch.sigmoid(self.fuse(concat_ab))
        return fusion_coef * a + (1 - fusion_coef) * b


class FeatureAugment(nn.Module):
    def __init__(self, noise_level=0.1, mask_prob=0.1):
        super(FeatureAugment, self).__init__()
        self.noise_level = noise_level
        self.mask_prob = mask_prob

    def forward(self, x):
        """
        x: [Batch, Seq_Len, Dim] or [Batch, Dim]
        """
        if not self.training:
            return x

        # 1. 高斯噪声注入
        noise = torch.randn_like(x) * self.noise_level
        x = x + noise

        # 2. 随机 Mask (如果是序列特征)
        if x.dim() == 3:
            B, L, D = x.shape
            mask = torch.rand(B, L, 1, device=x.device) > self.mask_prob
            x = x * mask.float()

        return x


class BiLSTMModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, batch_first=True, dropout_rate=0.5):
        super(BiLSTMModule, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout_input = nn.Dropout(p=0.0)
        self.bilstm = nn.LSTM(input_size=input_dim,
                              hidden_size=hidden_dim,
                              num_layers=num_layers,
                              batch_first=batch_first,
                              bidirectional=True)
        self.dropout_output = nn.Dropout(p=self.dropout_rate)

    def forward(self, x):
        output, (hn, cn) = self.bilstm(x)
        output = self.dropout_output(output)
        return output


class PeepholeLSTMModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, batch_first=True, dropout_rate=0.5):
        super(PeepholeLSTMModule, self).__init__()
        self.dropout_rate = dropout_rate
        self.peepholelstm = BiPeepholeLSTMLayer(input_size=input_dim,
                                                hidden_size=hidden_dim, )
        self.dropout_output = nn.Dropout(p=self.dropout_rate)

    def forward(self, x):
        output = self.peepholelstm(x)
        output = self.dropout_output(output)
        return output


class DAF(nn.Module):
    def __init__(self, config, args):
        super(DAF, self).__init__()
        self.args = args

        text_feat_dim, audio_feat_dim, video_feat_dim = args.text_feat_dim, args.audio_feat_dim, args.video_feat_dim

        self.extra_encoder = args.extra_encoder
        if self.extra_encoder:
            self.visual_attn = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=video_feat_dim + text_feat_dim, nhead=8, dim_feedforward=1024),
                num_layers=6
            )
            self.acoustic_attn = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=audio_feat_dim + text_feat_dim, nhead=8, dim_feedforward=1024),
                num_layers=6
            )

        self.visual_dyn = DynamicLayer(video_feat_dim + text_feat_dim, text_feat_dim, max_depth=args.max_depth)
        self.acoustic_dyn = DynamicLayer(audio_feat_dim + text_feat_dim, text_feat_dim, max_depth=args.max_depth)

        self.visual_reshape = nn.Linear(video_feat_dim, text_feat_dim)
        self.acoustic_reshape = nn.Linear(audio_feat_dim, text_feat_dim)

        self.attn_v = nn.Sequential(
            BiLSTMModule(input_dim=text_feat_dim, hidden_dim=video_feat_dim // 2, num_layers=1, dropout_rate=0.5),
            nn.Linear(video_feat_dim, 1)
        )
        self.attn_a = nn.Sequential(
            BiLSTMModule(input_dim=text_feat_dim, hidden_dim=audio_feat_dim // 2, num_layers=1, dropout_rate=0.5),
            nn.Linear(audio_feat_dim, 1)
        )

        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(args.output_droupout_prob)

        self.prelu_weight_v = nn.Parameter(torch.tensor(0.25))
        self.prelu_weight_a = nn.Parameter(torch.tensor(0.25))

    def forward(self, text_embedding, visual, acoustic):
        eps = 1e-6

        if self.extra_encoder:
            visual_text_pair = self.visual_attn(torch.cat((visual, text_embedding), dim=-1))
            acoustic_text_pair = self.acoustic_attn(torch.cat((acoustic, text_embedding), dim=-1))
        else:
            visual_text_pair = torch.cat((visual, text_embedding), dim=-1)
            acoustic_text_pair = torch.cat((acoustic, text_embedding), dim=-1)
        weight_v = F.prelu(self.visual_dyn(visual_text_pair), self.prelu_weight_v)
        weight_a = F.prelu(self.acoustic_dyn(acoustic_text_pair), self.prelu_weight_a)

        visual_transformed = self.visual_reshape(visual)
        acoustic_transformed = self.acoustic_reshape(acoustic)

        # Compute intermediate modality-specific features
        weighted_v = weight_v * visual_transformed
        weighted_a = weight_a * acoustic_transformed

        attn_scores_v = torch.sigmoid(self.attn_v(weighted_v))
        attn_scores_a = torch.sigmoid(self.attn_a(weighted_a))

        # Normalize attention scores across modalities
        total_attn = attn_scores_v + attn_scores_a + eps
        attn_scores_v = attn_scores_v / total_attn
        attn_scores_a = attn_scores_a / total_attn

        weighted_v = attn_scores_v * weighted_v
        weighted_a = attn_scores_a * weighted_a

        fusion = weighted_v + weighted_a + text_embedding

        # Normalize and apply dropout
        output_fusion = self.dropout(self.LayerNorm(fusion))
        return output_fusion


class Anchor(BertPreTrainedModel):
    def __init__(self, config, args, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None
        self.args = args
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            condition_idx,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class Positive(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.args = args
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        # Visual Encoder
        self.visual_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=args.video_feat_dim, nhead=8, dim_feedforward=1024),
            num_layers=6
        )
        self.visual_reshape = nn.Linear(args.video_feat_dim, args.text_feat_dim)

        # Acoustic Encoder
        self.acoustic_encoder = PeepholeLSTMModule(args.audio_feat_dim, args.audio_feat_dim // 2, num_layers=1,
                                                   dropout_rate=0.0)
        self.acoustic_reshape = nn.Linear(args.audio_feat_dim, args.text_feat_dim)

        self.DAF = DAF(config, args)
        self.alignNet = AlignSubNet(args, args.aligned_method)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # --- 1. 特征增强模块 (数据增强) ---
        self.aug = FeatureAugment(noise_level=0.05, mask_prob=0.1)

        # --- 2. 多通道 HAN (特征增强) ---
        self.HAN_text = HANLayer(config.hidden_size, config.hidden_size // 4, num_heads=4)
        self.fusion_text = NewFusionGate(config.hidden_size)

        # 视频 HAN (新增)
        self.HAN_video = HANLayer(args.video_feat_dim, args.video_feat_dim // 4, num_heads=4)
        self.fusion_video = NewFusionGate(args.video_feat_dim)

        # 音频 HAN (新增)
        self.HAN_audio = HANLayer(args.audio_feat_dim, args.audio_feat_dim // 4, num_heads=4)
        self.fusion_audio = NewFusionGate(args.audio_feat_dim)

        # ================= Alignment Modules =================
        self.grounding_text = MLPLayer(config.hidden_size, config.hidden_size)
        self.grounding_video = MLPLayer(args.video_feat_dim, config.hidden_size)
        self.grounding_audio = MLPLayer(args.audio_feat_dim, config.hidden_size)

        # 2. 一致性模块 (Consistency)
        self.consistency_tv = ConsistencySimilarityModule(input_dim=config.hidden_size, sim_dim=128)
        self.consistency_ta = ConsistencySimilarityModule(input_dim=config.hidden_size, sim_dim=128)
        self.loss_func_similarity = nn.CosineEmbeddingLoss(margin=0.2)

        # 3. 相似度计算与损失 (Similarity & Alignment)
        self.sim_calc = Similarity(temp=0.5)  # 普通余弦，用于 KL Loss 的输入
        self.sim_margin = ArcSimilarity(temp=0.1, margin=0.1)  # ArcFace 余弦，用于强监督
        self.kl_loss_func = KLContrastiveSimLoss(tau=0.5)

        self.logit_scale = torch.tensor(np.log(1 / 0.05))

        self.init_weights()

    def KLContrastiveSimLoss(
            self,
            logits: torch.Tensor,
            softlabel: torch.Tensor,
            tau: float,
            softlabel_tau: float,
            use_loss: str = "kl",
    ) -> torch.Tensor:

        sim_targets = F.softmax(
            softlabel / softlabel_tau, dim=1
        )
        logit_inputs = F.log_softmax(logits / tau, dim=1)

        if use_loss == "kl":
            loss = F.kl_div(
                logit_inputs, sim_targets, reduction="batchmean"
            )
        elif use_loss == "contrastive":
            loss = -torch.sum(
                logit_inputs * sim_targets, dim=1
            ).mean()
        else:
            raise ValueError("loss mode error")

        return loss

    def forward(
            self,
            input_ids,
            visual,
            acoustic,
            condition_idx,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            h_graph =None,
            label_ids=None
    ):

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=device)

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (
                encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers)

        # get embeddings of normal samples
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        text_embedding, visual, acoustic = self.alignNet(embedding_output, visual, acoustic)

        # text_encoder
        encoder_outputs = self.encoder(
            text_embedding,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        text_feat = encoder_outputs[0]
        text_feat = self.dropout(text_feat)  # -> DAF

        # Visual Encoder
        visual_feat = self.visual_encoder(visual)

        # Acoustic Encoder
        acoustic_feat = self.acoustic_encoder(acoustic)


        node_text = text_feat[:, 0, :]

        graph_output = self.HAN_text(h_graph, node_text)

        enhanced_node = self.fusion_text(node_text, graph_output)

        delta = enhanced_node - node_text
        text_feat = text_feat + delta.unsqueeze(1)

        # === Channel 2: Video ===
        node_visual = visual_feat.mean(dim=1)
        node_visual_aug = self.aug(node_visual)
        graph_out_video = self.HAN_video(h_graph, node_visual_aug)
        enhanced_node_video = self.fusion_video(node_visual, graph_out_video)
        visual_feat = visual_feat + (enhanced_node_video - node_visual).unsqueeze(1)

        # === Channel 3: Audio ===
        node_acoustic = acoustic_feat.mean(dim=1)
        node_acoustic_aug = self.aug(node_acoustic)  # 数据增强
        graph_out_audio = self.HAN_audio(h_graph, node_acoustic_aug)  # 结构增强
        enhanced_node_audio = self.fusion_audio(node_acoustic, graph_out_audio)
        acoustic_feat = acoustic_feat + (enhanced_node_audio - node_acoustic).unsqueeze(1)

        enhanced_text = text_feat[:, 0, :]
        enhanced_visual = visual_feat.mean(dim=1)
        enhanced_acoustic = acoustic_feat.mean(dim=1)
        loss_consistency = torch.tensor(0.0).to(device)
        alignment_loss = torch.tensor(0.0).to(device)

        # ================= Consistency Learning =================
        if self.training:
            current_labels = label_ids
            # A. 投影 (Grounding)
            feat_t = self.grounding_text(enhanced_text)
            feat_v = self.grounding_video(enhanced_visual)
            feat_a = self.grounding_audio(enhanced_acoustic)
            matched_v, unmatched_v = prepare_data(feat_v, current_labels)
            matched_a, unmatched_a = prepare_data(feat_a, current_labels)

            # 2. Text-Video Consistency
            t_aligned_match_v, v_aligned_match, _ = self.consistency_tv(feat_t, matched_v)
            # 传入 (Text, Unmatched_Video)
            t_aligned_unmatch_v, v_aligned_unmatch, _ = self.consistency_tv(feat_t, unmatched_v)

            similarity_label_tv = torch.cat(
                [
                    torch.ones(t_aligned_match_v.size(0), device=device),
                    -1 * torch.ones(t_aligned_unmatch_v.size(0), device=device),
                ],
                dim=0,
            )

            # 4. 拼接特征
            text_aligned_all_v = torch.cat([t_aligned_match_v, t_aligned_unmatch_v], dim=0)
            image_aligned_all_v = torch.cat([v_aligned_match, v_aligned_unmatch], dim=0)

            # 5. 计算 Loss (Text-Video)
            loss_cons_tv = self.loss_func_similarity(
                text_aligned_all_v,
                image_aligned_all_v,
                similarity_label_tv
            )

            # 6. Text-Audio Forward (重复上述逻辑)
            t_aligned_match_a, a_aligned_match, _ = self.consistency_ta(feat_t, matched_a)
            t_aligned_unmatch_a, a_aligned_unmatch, _ = self.consistency_ta(feat_t, unmatched_a)

            # 构造 Audio 标签 (逻辑同上)
            similarity_label_ta = torch.cat(
                [
                    torch.ones(t_aligned_match_a.size(0), device=device),
                    -1 * torch.ones(t_aligned_unmatch_a.size(0), device=device),
                ],
                dim=0,
            )

            text_aligned_all_a = torch.cat([t_aligned_match_a, t_aligned_unmatch_a], dim=0)
            audio_aligned_all_a = torch.cat([a_aligned_match, a_aligned_unmatch], dim=0)

            loss_cons_ta = self.loss_func_similarity(
                text_aligned_all_a,
                audio_aligned_all_a,
                similarity_label_ta
            )

            loss_consistency = loss_cons_tv + loss_cons_ta
            # ================= [新增部分：Cross-modal Alignment] =================
            feat_t_norm = feat_t / feat_t.norm(dim=-1, keepdim=True)
            feat_v_norm = feat_v / feat_v.norm(dim=-1, keepdim=True)
            feat_a_norm = feat_a / feat_a.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp() if hasattr(self, 'logit_scale') else 1.0

            sim_tv = logit_scale * torch.matmul(feat_t_norm, feat_v_norm.t())

            sim_ta = logit_scale * torch.matmul(feat_t_norm, feat_a_norm.t())

            label_mask = (current_labels.unsqueeze(0) == current_labels.unsqueeze(1)).float()
            label_target = label_mask / label_mask.sum(dim=1, keepdim=True)

            loss_align_tv = self.KLContrastiveSimLoss(
                sim_tv,  # 预测的相似度 logits
                label_target,  # 这里通常传入 labels，如果是自监督对比，就是对角线
                0.45, 0.5,  # 参考代码的超参
                use_loss="kl"
            )
            loss_align_tv += self.KLContrastiveSimLoss(
                sim_tv.t(),  # 转置，计算 Video -> Text 方向
                label_target,
                0.45, 0.5,
                use_loss="kl"
            )
            loss_align_tv /= 2.0

            # --- Text-Audio Alignment ---
            loss_align_ta = self.KLContrastiveSimLoss(
                sim_ta,
                label_target,
                0.45, 0.5,
                use_loss="kl"
            )
            loss_align_ta += self.KLContrastiveSimLoss(
                sim_ta.t(),  # 转置，计算 Audio -> Text 方向
                label_target,
                0.45, 0.5,
                use_loss="kl"
            )
            loss_align_ta /= 2.0

            alignment_loss = loss_align_tv + loss_align_ta + 0.5 * loss_consistency

        text_view = text_feat
        visual_view = self.visual_reshape(visual_feat)
        acoustic_view = self.acoustic_reshape(acoustic_feat)

        fused_embedding = self.DAF(text_feat, visual_feat, acoustic_feat)

        pooled_output = self.pooler(fused_embedding)

        return fused_embedding, pooled_output, text_view, visual_view, acoustic_view, alignment_loss


class Positive_Model(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = args.num_labels
        self.label_len = args.label_len

        self.bert = Positive(config, args)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, args.num_labels)
        self.init_weights()

    def forward(
            self,
            text,
            visual,
            acoustic,
            condition_idx,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            h_graph=None,
            label_ids=None
    ):
        input_ids, attention_mask, token_type_ids = text[:, 0], text[:, 1], text[:, 2]

        outputs, pooled, text_view, visual_view, acoustic_view, alignment_loss \
            = self.bert(
            input_ids,
            visual,
            acoustic,
            condition_idx,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            h_graph=h_graph,
            label_ids=label_ids
        )

        text_condition_tuple = tuple(
            text_view[torch.arange(text_view.shape[0]), condition_idx.view(-1) + i, :].unsqueeze(1) for i in
            range(self.label_len))
        text_condition = torch.cat(text_condition_tuple, dim=1)

        visual_condition_tuple = tuple(
            visual_view[torch.arange(visual_view.shape[0]), condition_idx.view(-1) + i, :].unsqueeze(1) for i in
            range(self.label_len))
        visual_condition = torch.cat(visual_condition_tuple, dim=1)

        acoustic_condition_tuple = tuple(
            acoustic_view[torch.arange(acoustic_view.shape[0]), condition_idx.view(-1) + i, :].unsqueeze(1) for i in
            range(self.label_len))
        acoustic_condition = torch.cat(acoustic_condition_tuple, dim=1)

        condition_tuple = tuple(
            outputs[torch.arange(outputs.shape[0]), condition_idx.view(-1) + i, :].unsqueeze(1) for i in
            range(self.label_len))
        condition = torch.cat(condition_tuple, dim=1)

        pooled_output = pooled
        outputs = self.classifier(pooled_output)

        return outputs, pooled_output, condition, text_condition, visual_condition, acoustic_condition, alignment_loss


class MVCL_DAF(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.positive = Positive_Model.from_pretrained(args.cache_path, local_files_only=True, args=args)
        self.anchor = Anchor.from_pretrained(args.cache_path, local_files_only=True, args=args)

        self.label_len = args.label_len
        args.feat_size = args.text_feat_dim
        args.video_feat_size = args.video_feat_dim
        args.audio_feat_size = args.audio_feat_dim

    def forward(
            self,
            text_feats,
            video_feats,
            audio_feats,
            cons_text_feats,
            condition_idx,
            h_graph,
            label_ids=None

    ):
        video_feats = video_feats.float()
        audio_feats = audio_feats.float()

        outputs_map, pooled_output_map, condition, text_condition, visual_condition, acoustic_condition, alignment_loss \
            = self.positive(
            text=text_feats,
            visual=video_feats,
            acoustic=audio_feats,
            condition_idx=condition_idx,
            h_graph=h_graph,
            label_ids=label_ids
        )

        outputs = outputs_map
        pooled_output = pooled_output_map

        cons_input_ids, cons_input_mask, cons_segment_ids = cons_text_feats[:, 0], cons_text_feats[:,
                                                                                   1], cons_text_feats[:, 2]
        cons_outputs = self.anchor(
            input_ids=cons_input_ids,
            condition_idx=condition_idx,
            token_type_ids=cons_segment_ids,
            attention_mask=cons_input_mask
        )
        last_hidden_state = cons_outputs.last_hidden_state

        cons_condition_tuple = tuple(
            last_hidden_state[torch.arange(last_hidden_state.shape[0]), condition_idx.view(-1) + i, :].unsqueeze(1) for
            i in range(self.label_len))
        cons_condition = torch.cat(cons_condition_tuple, dim=1)

        return outputs, pooled_output, condition.mean(dim=1), cons_condition.mean(dim=1), text_condition.mean(
            dim=1), visual_condition.mean(dim=1), acoustic_condition.mean(dim=1), alignment_loss