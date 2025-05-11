import torch
import torch.nn as nn
import torch.nn.functional as F




'''
Bidirectional cross-attention layers.
'''
class BidirectionalCrossAttention(nn.Module):

    def __init__(self, model_dim, Q_dim, K_dim, V_dim):
        super().__init__()

        self.query_matrix = nn.Linear(model_dim, Q_dim)
        self.key_matrix = nn.Linear(model_dim, K_dim)
        self.value_matrix = nn.Linear(model_dim, V_dim)
    

    def bidirectional_scaled_dot_product_attention(self, Q, K, V):
        score = torch.bmm(Q, K.transpose(-1, -2))
        scaled_score = score / (K.shape[-1]**0.5)
        attention = torch.bmm(F.softmax(scaled_score, dim = -1), V)

        return attention
    

    def forward(self, query, key, value):
        Q = self.query_matrix(query)
        K = self.key_matrix(key)
        V = self.value_matrix(value)
        attention = self.bidirectional_scaled_dot_product_attention(Q, K, V)

        return attention



'''
Multi-head bidirectional cross-attention layers.
'''
class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, model_dim, Q_dim, K_dim, V_dim):
        super().__init__()

        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList(
            [BidirectionalCrossAttention(model_dim, Q_dim, K_dim, V_dim) for _ in range(self.num_heads)]
        )
        self.projection_matrix = nn.Linear(num_heads * V_dim, model_dim)
    

    def forward(self, query, key, value):
        heads = [self.attention_heads[i](query, key, value) for i in range(self.num_heads)]
        multihead_attention = self.projection_matrix(torch.cat(heads, dim = -1))

        return multihead_attention



'''
A feed-forward network, which operates as a key-value memory.
'''
class Feedforward(nn.Module):

    def __init__(self, model_dim, hidden_dim, dropout_rate):
        super().__init__()

        self.linear_W1 = nn.Linear(model_dim, hidden_dim)
        self.linear_W2 = nn.Linear(hidden_dim, model_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
    

    def forward(self, x):
        return self.dropout(self.linear_W2(self.relu(self.linear_W1(x))))



'''
Residual connection to smooth the learning process.
'''
class AddNorm(nn.Module):

    def __init__(self, model_dim, dropout_rate):
        super().__init__()

        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout_rate)

    
    def forward(self, x, sublayer):
        output = self.layer_norm(x + self.dropout(sublayer(x)))

        return output



'''
MultiAttn is a multimodal fusion model which aims to capture the complicated interactions and 
dependencies across textual, audio and visual modalities through bidirectional cross-attention layers.
MultiAttn is made up of three sub-components:
1. MultiAttn_text: integrate the textual modality with audio and visual information;
2. MultiAttn_audio: incorporate the audio modality with textual and visual information;
3. MultiAttn_visual: fuse the visual modality with textual and visual cues.
'''
class MultiAttnLayer(nn.Module):

    def __init__(self, num_heads, model_dim, hidden_dim, dropout_rate):
        super().__init__()

        Q_dim = K_dim = V_dim = model_dim // num_heads
        self.attn_1 = MultiHeadAttention(num_heads, model_dim, Q_dim, K_dim, V_dim)
        self.add_norm_1 = AddNorm(model_dim, dropout_rate)
        self.attn_2 = MultiHeadAttention(num_heads, model_dim, Q_dim, K_dim, V_dim)
        self.add_norm_2 = AddNorm(model_dim, dropout_rate)
        self.ff = Feedforward(model_dim, hidden_dim, dropout_rate)
        self.add_norm_3 = AddNorm(model_dim, dropout_rate)


    def forward(self, query_modality, modality_A, modality_B, modality_C):
        attn_output_1 = self.add_norm_1(query_modality, lambda query_modality: self.attn_1(query_modality, modality_A, modality_A))
        attn_output_2 = self.add_norm_2(attn_output_1, lambda attn_output_1: self.attn_2(attn_output_1, modality_B, modality_B))
        attn_output_3 = self.add_norm_2(attn_output_2, lambda attn_output_2: self.attn_2(attn_output_2, modality_C, modality_C))
        ff_output = self.add_norm_3(attn_output_3, self.ff)

        return ff_output


class promptAttnLayer(nn.Module):

    def __init__(self, num_heads, model_dim, hidden_dim, dropout_rate):
        super().__init__()

        Q_dim = K_dim = V_dim = model_dim // num_heads
        self.attn_1 = MultiHeadAttention(num_heads, model_dim, Q_dim, K_dim, V_dim)
        self.add_norm_1 = AddNorm(model_dim, dropout_rate)
        self.attn_2 = MultiHeadAttention(num_heads, model_dim, Q_dim, K_dim, V_dim)
        self.add_norm_2 = AddNorm(model_dim, dropout_rate)
        self.ff = Feedforward(model_dim, hidden_dim, dropout_rate)
        self.add_norm_3 = AddNorm(model_dim, dropout_rate)


    def forward(self, query_modality, modality_A, modality_B):
        attn_output_1 = self.add_norm_1(query_modality, lambda query_modality: self.attn_1(query_modality, modality_A, modality_A))
        attn_output_2 = self.add_norm_2(attn_output_1, lambda attn_output_1: self.attn_2(attn_output_1, modality_B, modality_B))
        ff_output = self.add_norm_3(attn_output_2, self.ff)

        return ff_output

'''
Stacks of MultiAttn layers.
'''
class MultiAttn(nn.Module):

    def __init__(self, num_layers, model_dim, num_heads, hidden_dim, dropout_rate):
        super().__init__()

        self.multiattn_layers = nn.ModuleList([
            MultiAttnLayer(num_heads, model_dim, hidden_dim, dropout_rate) for _ in range(num_layers)])


    def forward(self, query_modality, modality_A, modality_B, modality_C):
        for multiattn_layer in self.multiattn_layers:
            query_modality = multiattn_layer(query_modality, modality_A, modality_B, modality_C)
        
        return query_modality



class MultiAttnModel(nn.Module):

    def __init__(self, num_layers_t, num_layers_av, model_dim, num_heads, hidden_dim, dropout_rate):
        super().__init__()

        self.multiattn_text = MultiAttn(num_layers_t, model_dim, num_heads, hidden_dim, dropout_rate)
        self.multiattn_audio = MultiAttn(num_layers_av, model_dim, num_heads, hidden_dim, dropout_rate)
        self.multiattn_visual = MultiAttn(num_layers_av, model_dim, num_heads, hidden_dim, dropout_rate)

    
    def forward(self, text_features, audio_features, visual_features, merge_features):
        # f_t = self.multiattn_text(text_features, text_features, text_features)
        f_t = self.multiattn_text(text_features, audio_features, visual_features, merge_features)
        # f_t = self.multiattn_text(prompt_add_text, audio_features, visual_features)
        # f_a = self.multiattn_audio(audio_features, audio_features, audio_features)
        f_a = self.multiattn_audio(audio_features, text_features, visual_features, merge_features)
        # _v = self.multiattn_visual(visual_features, visual_features, visual_features)
        f_v = self.multiattn_visual(visual_features, text_features, audio_features, merge_features)

        return f_t, f_a, f_v

class prompt_Attn(nn.Module):

    def __init__(self, num_layers, model_dim, num_heads, hidden_dim, dropout_rate):
        super().__init__()

        self.multiattn_layers = nn.ModuleList([
            promptAttnLayer(num_heads, model_dim, hidden_dim, dropout_rate) for _ in range(num_layers)])

    def forward(self, query_modality, modality_A, modality_B):
        query_modality = query_modality.transpose(0, 1)
        # modality_A = modality_A.transpose(0, 1)
        # modality_B = modality_B.transpose(0, 1)
        for multiattn_layer in self.multiattn_layers:
            query_modality = multiattn_layer(query_modality, modality_A, modality_B)

        # return query_modality.transpose(0, 1)
        return query_modality

if __name__ == '__main__':
    query = torch.randn(4, 24, 512).transpose(0, 1)
    Key = torch.randn(4, 7, 512).transpose(0, 1)
    value = torch.randn(4, 7, 512).transpose(0, 1)
    multiattn = prompt_Attn(2, 512, 4, 1024, 0)
    fused_text_features = multiattn(query, Key, value)
    print('done')