import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import pandas as pd
from base import BaseModel
import math

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class ComboMTL(BaseModel):
    def __init__(self,
                 protein_num,
                 cell_num,
                 drug_num,
                 emb_dim,
                 n_hop,
                 l1_decay,
                 therapy_method):
        super(ComboMTL,self).__init__()
        self.protein_num = protein_num
        self.cell_num = cell_num
        self.drug_num = drug_num
        self.emb_dim = emb_dim
        self.n_hop = n_hop
        self.l1_decay = l1_decay
        self.therapy_method = therapy_method

        #self.protein_embedding = nn.Embedding(self.protein_num, self.emb_dim)

        protein_embedding_data = torch.tensor(pd.read_csv('data/DrugCombDB/protein_features_128.csv').iloc[:, 1:].values,
                                           dtype=torch.float32)
        num_protein_embeddings, protein_embedding_dim = protein_embedding_data.shape
        self.protein_embedding = nn.Embedding(num_protein_embeddings, protein_embedding_dim)
        self.protein_embedding.weight.data.copy_(protein_embedding_data)
        self.protein_embedding.weight.requires_grad = True

        #self.cell_embedding = nn.Embedding(self.cell_num, self.emb_dim)

        cell_embedding_data = torch.tensor(pd.read_csv('data/DrugCombDB/cell_features_128.csv').iloc[:, 1:].values,
                                           dtype=torch.float32)
        num_cell_embeddings, cell_embedding_dim = cell_embedding_data.shape
        self.cell_embedding = nn.Embedding(num_cell_embeddings, cell_embedding_dim)
        self.cell_embedding.weight.data.copy_(cell_embedding_data)
        self.cell_embedding.weight.requires_grad = True


        #self.drug_embedding = nn.Embedding(self.drug_num, self.emb_dim)

        drug_embedding_data = torch.tensor(pd.read_csv('data/DrugCombDB/smiles_128.csv').iloc[:, 1:].values,
                                           dtype=torch.float32)
        num_drug_embeddings, drug_embedding_dim = drug_embedding_data.shape
        self.drug_embedding = nn.Embedding(num_drug_embeddings, drug_embedding_dim)
        self.drug_embedding.weight.data.copy_(drug_embedding_data)
        self.drug_embedding.weight.requires_grad = True

        self.aggregation_function = nn.Linear(self.emb_dim*self.n_hop, self.emb_dim) #聚合函数
        self.neighbor_proj = nn.Linear(128, 512)

        self.num_heads = 8
        self.nh = self.num_heads
        self.dk = self.emb_dim // self.nh
        self.dv = self.emb_dim // self.nh
        self.trans_Q = nn.Linear(self.emb_dim * 3, self.emb_dim)
        self.trans_K = nn.Linear(self.emb_dim * 3, self.emb_dim)
        self.trans_V = nn.Linear(self.emb_dim * 3, self.emb_dim)
        self.linears = clones(nn.Linear(self.emb_dim, self.emb_dim * 3), 4)
        self.projection = nn.Linear(self.emb_dim * 3, self.emb_dim * 3)
        self.attention_layers = nn.ModuleList()
        for i in range(n_hop):
            self.attention_layers.append(nn.Linear(self.emb_dim, self.emb_dim, bias=False))

        self.actP = nn.PReLU()
        self.drop_out = nn.Dropout(0.5)

        self.global_attn = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.GELU(),
            nn.Linear(self.emb_dim, 1)
        )

        self.local_attn = nn.Sequential(
            nn.Linear(2 * self.emb_dim, self.emb_dim),
            nn.GELU(),
            nn.Linear(self.emb_dim, 1)
        )

        self.out_transform = nn.Linear(self.emb_dim, self.emb_dim)

        self.transform = nn.Linear(self.emb_dim*2, self.emb_dim*2)

        self.gate = nn.Linear(self.emb_dim*2, self.emb_dim*2)
        self.carry = nn.Linear(self.emb_dim*2, self.emb_dim*2)

        self.syfcn = nn.Sequential(
            nn.Linear(self.emb_dim * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)  # 输出预测分数：1个值（二分类/回归）
        )

        self.sifcn = nn.Sequential(
            nn.Linear(self.emb_dim*2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)  # 输出预测分数：1个值（二分类/回归）
        )



    def forward(self,
                cells: torch.LongTensor,
                drug1: torch.LongTensor,
                drug2: torch.LongTensor,
                cell_neighbors: list,
                drug1_neighbors: list,
                drug2_neighbors: list):
        cell_embeddings = self.cell_embedding(cells)
        drug1_embeddings = self.drug_embedding(drug1)
        drug2_embeddings = self.drug_embedding(drug2)

        cell_neighbors_emb_list = self._get_neighbor_emb(cell_neighbors)#去得到cell的邻居
        drug1_neighbors_emb_list = self._get_neighbor_emb(drug1_neighbors)  #去得到drug1的邻居
        drug2_neighbors_emb_list = self._get_neighbor_emb(drug2_neighbors)  #去得到drug2的邻居

        cell_i_list = self._interaction_aggregation(cell_embeddings, cell_neighbors_emb_list)  #第i层：将靶标protein的嵌入与cell嵌入结合在一起
        drug1_i_list = self._interaction_aggregation(drug1_embeddings, drug1_neighbors_emb_list)
        drug2_i_list = self._interaction_aggregation(drug2_embeddings, drug2_neighbors_emb_list)

        # [batch_size, dim]
        cell_embeddings = self._aggregation(cell_i_list)  # 将各层的嵌入整合到一起
        drug1_embeddings = self._aggregation(drug1_i_list)
        drug2_embeddings = self._aggregation(drug2_i_list)


        embeddings = torch.cat((cell_embeddings, drug1_embeddings, drug2_embeddings), 1)
        batch, n = embeddings.shape

        Q = self.trans_Q(embeddings)
        K = self.trans_K(embeddings)
        V = self.trans_V(embeddings)

        query, key, value = \
            [model(x).view(batch, -1, self.num_heads, self.dk).transpose(1, 2)
             for model, x in zip(self.linears, (Q, K, V))]
        x, self.attn = self.attention(query, key, value)
        x = x.transpose(1, 2).contiguous().view(batch, -1, self.num_heads * self.dk)
        x_re = x.reshape(batch, n)

        layer_norm = torch.nn.LayerNorm(n, elementwise_affine=False)
        x1 = layer_norm(embeddings + x_re)
        x1 = self.drop_out(x1)
        x_f = self.projection(x1)
        x_vec = layer_norm(x1 + x_f)

        # (2) CIE unit
        x_bit0 = embeddings.view(-1)  # 1+98304---1*192
        layer1 = torch.nn.Linear(len(x_bit0), 512)
        layer2 = torch.nn.Linear(512, 384)
        x_bit1 = self.actP(layer1(x_bit0))
        x_bit1 = self.drop_out(x_bit1)
        x_bit = self.actP(layer2(x_bit1))

        # (3) integration unit
        com_feature = x_vec * x_bit  # 512*64
        weight_matrix = torch.sigmoid(x_vec * x_bit)
        x = embeddings * weight_matrix + com_feature * (torch.tensor(1.0) - weight_matrix)

        inc = []
        for i in range(128):
            inc.append(i)
        indices1 = torch.tensor(inc)
        cell_embedding1s = torch.index_select(x, 1, indices1)


        ind1 = []
        for j in range(128, 256):
            ind1.append(j)
        indices2 = torch.tensor(ind1)
        drug1_embedding1s = torch.index_select(x, 1, indices2)


        ind2 = []
        for k in range(256, 384):
            ind2.append(k)
        indices3 = torch.tensor(ind2)
        drug2_embedding1s = torch.index_select(x, 1, indices3)

        sembeddings = torch.cat((cell_embedding1s, drug1_embedding1s, drug2_embedding1s), 1)
        Syn_score = self.syfcn(sembeddings).squeeze(1)  #最后的预测得分


        dembeddings = torch.cat((drug1_embeddings, drug2_embeddings), 1)
        transform_out = F.relu(self.transform(dembeddings))  # φ(Y)
        gate_out = torch.sigmoid(self.gate(dembeddings))  # g(Y)
        carry_out = self.carry(dembeddings)  # q(Y)
        highway_out = gate_out * transform_out + (1 - gate_out) * carry_out
        Side_score = self.sifcn(highway_out).squeeze(1)

        return Syn_score, Side_score

    def _get_neighbor_emb(self, neighbors): #得到邻居的嵌入
        neighbors_emb_list = []
        '''
        for hop in range(len(neighbors)):
            neighbors[hop] = list(set(neighbors[hop]))
        for hop in range(len(neighbors)):  # 根据当前层邻居的个数进行循环
            # 如果该层没有邻居节点，添加空嵌入或者跳过
            if len(neighbors[hop]) == 0:
                # 可以选择添加空嵌入（例如全零向量）或者跳过
                # 这里我们跳过该层
                continue
            neighbors_emb_list.append(self.protein_embedding(torch.LongTensor(neighbors[hop])))  # 得到各层的靶标protein嵌入
        
        for hop in range(self.n_hop):
            neighbors_emb_list.append(self.protein_embedding(neighbors[hop])) #得到各层的靶标protein嵌入
        '''
        for hop in range(len(neighbors)):
            neighbors_emb_list.append(self.protein_embedding(neighbors[hop]))  # 得到各层的靶标protein嵌入

        return neighbors_emb_list

    def _interaction_aggregation(self, item_embeddings, neighbors_emb_list):
        interact_list = []

        for hop in range(self.n_hop):
            neighbor_emb = neighbors_emb_list[hop]  # [B, N, D]
            item_emb = item_embeddings  # [B, D]
            B, N, D = neighbor_emb.size()

            # --- Step 1: 计算全局注意力 ϵj ---
            global_input = neighbor_emb  # [B, N, D]
            global_score = self.global_attn(global_input).squeeze(-1)  # [B, N]

            # --- Step 2: 计算局部注意力 χij ---
            item_expand = item_emb.unsqueeze(1).expand(-1, N, -1)  # [B, N, D]
            pair_input = torch.cat([item_expand, neighbor_emb], dim=-1)  # [B, N, 2D]
            local_score = self.local_attn(pair_input).squeeze(-1)  # [B, N]

            # --- Step 3: 合并注意力得 δij ---
            attn_weights = F.softmax(global_score + local_score, dim=1)  # [B, N]

            # --- Step 4: 加权聚合邻居特征 ---
            agg = torch.sum(attn_weights.unsqueeze(-1) * neighbor_emb, dim=1)  # [B, D]
            item_embeddings = F.relu(self.out_transform(agg))  # 非线性变换可选
            interact_list.append(item_embeddings)

        return interact_list  # 得到各层的protein对cell/drug产生的嵌入

    def _aggregation(self, item_i_list):
        # [batch_size, n_hop+1, emb_dim]
        item_i_concat = torch.cat(item_i_list, 1)
        # [batch_size, emb_dim]
        item_embeddings = self.aggregation_function(item_i_concat)
        return item_embeddings  #得到drug或cell的最终嵌入

    def attention(self, query, key, value, mask=None, dropout=None):
        d_k = query.shape[-1]

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

        return emb_loss


