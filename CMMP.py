from Model.MultiAttn import MultiAttnModel, prompt_Attn
from Model.MLP import MLP
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.model_hyper import Hyper


def print_grad(grad):
    print('the grad is', grad[2][0:5])
    return grad


class Unimodal_GatedFusion(nn.Module):
    def __init__(self, hidden_size, dataset):
        super(Unimodal_GatedFusion, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.adapter = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, hidden_size)
        )
        if dataset == 'MELD':
            self.fc.weight.data.copy_(torch.eye(hidden_size, hidden_size))
            self.fc.weight.requires_grad = False

    def forward(self, a):
        # z = torch.sigmoid(self.fc(a))
        # z = torch.sigmoid(self.adapter(a))
        z = torch.sigmoid(self.adapter(a))
        final_rep = z * a
        return final_rep


def pad(tensor, length, no_cuda):
    if isinstance(tensor, Variable):
        var = tensor
        if length > var.size(0):
            if not no_cuda:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:]).cuda()])
            else:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:])])
        else:
            return var
    else:
        if length > tensor.size(0):
            if not no_cuda:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:]).cuda()])
            else:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:])])
        else:
            return tensor


def simple_batch_graphify(features, lengths, no_cuda):
    edge_index, edge_norm, edge_type, node_features = [], [], [], []
    batch_size = features.size(1)
    for j in range(batch_size):
        node_features.append(features[:lengths[j], j, :])

    node_features = torch.cat(node_features, dim=0)

    if not no_cuda:
        node_features = node_features.cuda()
    return node_features


class Model(nn.Module):

    def __init__(self, base_model, D_m, D_g, D_e, graph_hidden_size, n_speakers, n_classes=7, dropout=0.5,
                 no_cuda=False, model_type='relation', use_residue=True,
                 D_m_v=512, D_m_a=100, modals='avl', att_type='gated', dataset='IEMOCAP',
                 use_speaker=True, use_modal=False, norm='LN2'):

        super(Model, self).__init__()

        self.base_model = base_model
        self.no_cuda = no_cuda
        self.model_type = model_type
        self.dropout = dropout
        self.use_residue = use_residue
        self.modals = [x for x in modals]  # a, v, l
        self.use_speaker = use_speaker
        self.use_modal = use_modal
        self.att_type = att_type

        self.norm_strategy = norm
        if self.att_type == 'gated' or self.att_type == 'TF':
            self.multi_modal = True
        else:
            self.multi_modal = False
        self.dataset = dataset
        self.concat = False

        if self.base_model == 'GRU':
            self.gru = nn.GRU(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
            if 'a' in self.modals:
                hidden_a = D_g
                self.linear_audio = nn.Linear(D_m_a, hidden_a)
            if 'v' in self.modals:
                hidden_v = D_g
                self.linear_visual = nn.Linear(D_m_v, hidden_v)
            if 'l' in self.modals:
                hidden_l = D_g
                self.gru_l = nn.GRU(input_size=hidden_l, hidden_size=D_g // 2, num_layers=2, bidirectional=True,
                                    dropout=dropout)


        elif self.base_model == 'None':
            self.base_linear = nn.Linear(D_m, 2 * D_e)

        else:
            print('Base Model must be one of DialogRNN/LSTM/GRU')
            raise NotImplementedError

        if self.model_type == 'hyper':
            self.post_model = Hyper(n_dim=D_g, nhidden=graph_hidden_size, nclass=n_classes,
                                        dropout=self.dropout, variant=True, use_residue=self.use_residue,
                                        n_speakers=n_speakers, modals=self.modals, use_speaker=self.use_speaker,
                                        use_modal=self.use_modal)
        else:
            print("There are no such kind of model")

        if self.multi_modal:
            self.dropout_ = nn.Dropout(self.dropout)
            if self.att_type == 'concat_subsequently':
                if self.use_residue:
                    self.smax_fc = nn.Linear((D_g + graph_hidden_size) * len(self.modals), n_classes)
                else:
                    self.smax_fc = nn.Linear((graph_hidden_size) * len(self.modals), n_classes)
            elif self.att_type == 'TF':
                if self.use_residue:
                    self.smax_fc = nn.Linear((D_g + graph_hidden_size * 2) * len(self.modals), n_classes)
                else:
                    self.smax_fc = nn.Linear((graph_hidden_size * 2) * len(self.modals), n_classes)
            else:
                self.smax_fc = nn.Linear(D_g + graph_hidden_size * len(self.modals), graph_hidden_size)

    def forward(self, U, qmask, umask, seq_lengths, U_a=None, U_v=None, epoch=None, pvf=None, paf=None):

        if self.base_model == 'GRU':
            if 'a' in self.modals:
                if self.dataset == 'IEMOCAP':
                    pass
                emotions_a = U_a
            if 'v' in self.modals:
                if self.dataset == 'IEMOCAP':
                    pass
                emotions_v = U_v
            if 'l' in self.modals:
                emotions_l, hidden_l = self.gru_l(U)

        elif self.base_model == 'None':
            emotions = self.base_linear(U)

        if not self.multi_modal:
            features = simple_batch_graphify(emotions, seq_lengths, self.no_cuda)
        else:
            if 'a' in self.modals:
                features_a = simple_batch_graphify(emotions_a, seq_lengths, self.no_cuda)
            else:
                features_a = []
            if 'v' in self.modals:
                features_v = simple_batch_graphify(emotions_v, seq_lengths, self.no_cuda)
            else:
                features_v = []
            if 'l' in self.modals:
                features_l = simple_batch_graphify(emotions_l, seq_lengths, self.no_cuda)
            else:
                features_l = []

        if self.model_type == 'hyper':
            emotions_feat = self.post_model(features_a, features_v, features_l, seq_lengths, qmask, epoch)
            emotions_feat = self.dropout_(emotions_feat)

            emotions_feat = nn.ReLU()(emotions_feat)
            log_prob = F.log_softmax(self.smax_fc(emotions_feat), 1)

        else:
            print("error")
        return log_prob



class CMMP(nn.Module):

    def __init__(self, base_model, dataset, multi_attn_flag, roberta_dim, hidden_dim, dropout, num_layers_t, num_layers_av,
                 model_dim, num_heads, D_m_audio, D_m_visual, D_m, D_g, D_e, g_h, n_speakers, dropout2, no_cuda, model_type, use_residue,
                 D_m_v, D_m_a, modals, att_type, use_speaker, use_modal, norm, n_classes, device, visual_prompt,
                 audio_prompt, text_prompt):
        super().__init__()

        self.dataset = dataset
        self.visual_prompt = visual_prompt
        self.audio_prompt = audio_prompt
        self.text_prompt = text_prompt
        self.multi_attn_flag = multi_attn_flag
        self.prompt_layer = 1
        self.unimodal_layer = 1

        self.postprocess = Model(base_model, D_m, D_g, D_e, g_h, n_speakers=n_speakers, n_classes=n_classes,
                                 dropout=dropout2, no_cuda=no_cuda, model_type=model_type, use_residue=use_residue,
                                 D_m_v=D_m_v, D_m_a=D_m_a, modals=modals, att_type=att_type, dataset=self.dataset,
                                 use_speaker=use_speaker, use_modal=use_modal, norm=norm)

        self.audio_fc = nn.Linear(D_m_audio, model_dim)

        self.visual_fc = nn.Linear(D_m_visual, model_dim)

        self.prompt_attn = prompt_Attn(self.prompt_layer, model_dim, num_heads, hidden_dim, dropout)

        self.text_attn = prompt_Attn(self.unimodal_layer, model_dim, num_heads, hidden_dim, dropout)
        self.visual_attn = prompt_Attn(self.unimodal_layer, model_dim, num_heads, hidden_dim, dropout)
        self.audio_attn = prompt_Attn(self.unimodal_layer, model_dim, num_heads, hidden_dim, dropout)

        self.audio_gate = Unimodal_GatedFusion(model_dim, dataset)
        self.visual_gate = Unimodal_GatedFusion(model_dim, dataset)

        self.features_reduce_audio = nn.Linear(2 * model_dim, model_dim)
        self.features_reduce_visual = nn.Linear(2 * model_dim, model_dim)


        self.fc = nn.Linear(model_dim * 3, model_dim)

        self.multiattn = MultiAttnModel(num_layers_t, num_layers_av, model_dim, num_heads, hidden_dim, dropout)

        self.wo_cross = False
        self.pptsum = False

        if self.dataset == 'MELD':
            self.mlp = MLP(model_dim, model_dim * 2, n_classes, dropout)
        elif self.dataset == 'IEMOCAP':
            self.mlp = MLP(model_dim, model_dim, n_classes, dropout)

    def get_class_counts(self):
        class_counts = torch.zeros(self.num_classes).to(self.device)

        for _, data in enumerate(self.train_dataloader):
            _, _, _, _, _, padded_labels = [d.to(self.device) for d in data]
            padded_labels = padded_labels.reshape(-1)
            labels = padded_labels[padded_labels != -1]
            class_counts += torch.bincount(labels, minlength=self.num_classes)

        return class_counts

    def forward(self, texts, audios, visuals, speaker_masks, utterance_masks, dia_len, epoch):
        # texts []
        text_out = texts
        visual_prompt_features = self.visual_prompt  # [BS, 7, 512]
        audio_prompt_features = self.audio_prompt


        visual_prompt_features = visual_prompt_features.unsqueeze(0).expand(texts.shape[1], -1, -1)
        audio_prompt_features = audio_prompt_features.unsqueeze(0).expand(texts.shape[1], -1, -1)



        if self.wo_cross:
            visual_features = self.visual_fc(visuals)
            audio_features = self.audio_fc(audios)

            return visual_features, audio_features, texts, visual_prompt_features, audio_prompt_features


        audio_features = self.audio_fc(audios)
        gated_audio_features = self.audio_gate(audio_features)
        prompt_audio_features = self.prompt_attn(audio_features, audio_prompt_features, audio_prompt_features)  # choose
        gated_audio_prompt_features = self.audio_gate(prompt_audio_features)
        audio_out = self.features_reduce_audio(torch.cat([gated_audio_features.transpose(0, 1),
                                                          gated_audio_prompt_features], dim=-1))



        visual_features = self.visual_fc(visuals)
        gated_visual_features = self.audio_gate(visual_features)
        prompt_visual_features = self.prompt_attn(visual_features, visual_prompt_features, visual_prompt_features)  # choose
        gated_visual_prompt_features = self.visual_gate(prompt_visual_features)
        visual_out = self.features_reduce_visual(torch.cat([gated_visual_features.transpose(0, 1),
                                                            gated_visual_prompt_features], dim=-1))


        audio_out = audio_out.transpose(0, 1)
        visual_out = visual_out.transpose(0, 1)
        text_out = text_out.transpose(0, 1)



        if self.multi_attn_flag:
            merge_features = torch.cat((text_out.transpose(0, 1), audio_out,  visual_out), dim=-1)
            merge_features = self.fc(merge_features)
            text_out, audio_out, visual_out = self.multiattn(texts, audio_out, visual_out, merge_features)

        log_prob = self.postprocess(text_out, speaker_masks, utterance_masks, dia_len, audio_out, visual_out, epoch, visual_prompt_features.transpose(0, 1), audio_prompt_features.transpose(0, 1))

        return log_prob

