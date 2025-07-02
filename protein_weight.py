import torch
import argparse
import pandas as pd
import numpy as np
import pickle
import networkx as nx
import os
import json  # 导入 json 模块
from model.ComboMTL import ComboMTL
from data_loader.data_loaders import DataLoader

class CaseStudy:
    def __init__(self, config_path, model_path, data_dir):
        self.config_path = config_path
        self.model_path = model_path
        self.data_dir = data_dir

        # 直接加载配置文件
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)  # 使用 json.load() 加载配置文件

        # 加载数据
        self.data_loader = DataLoader(
            data_dir=self.data_dir,
            batch_size=1,
            score='synergy 0',
            n_hop=3,  # 使用3阶邻居
            n_memory=128,
            shuffle=False,
            validation_split=0.0,
            test_split=0.0,
            num_workers=0
        )
        self.feature_index = self.data_loader.get_feature_index()
        self.cell_neighbor_set = self.data_loader.get_cell_neighbor_set()
        self.drug_neighbor_set = self.data_loader.get_drug_neighbor_set()
        self.node_num_dict = self.data_loader.get_node_num_dict()

        # 加载模型
        self.model = ComboMTL(
            protein_num=self.node_num_dict['protein'],
            cell_num=self.node_num_dict['cell'],
            drug_num=self.node_num_dict['drug'],
            emb_dim=self.config['arch']['args']['emb_dim'],
            n_hop=self.config['arch']['args']['n_hop'],
            l1_decay=self.config['arch']['args']['l1_decay'],
            therapy_method=self.config['arch']['args']['therapy_method']
        )
        checkpoint = torch.load(self.model_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

    def predict(self, cell_idx, drug1_idx, drug2_idx):
        # 获取细胞和药物的邻居节点
        cell_neighbors = [self.cell_neighbor_set[cell_idx][hop] for hop in range(self.data_loader.n_hop)]
        drug1_neighbors = [self.drug_neighbor_set[drug1_idx][hop] for hop in range(self.data_loader.n_hop)]
        drug2_neighbors = [self.drug_neighbor_set[drug2_idx][hop] for hop in range(self.data_loader.n_hop)]

        # 准备输入数据
        cell = torch.LongTensor([cell_idx])
        drug1 = torch.LongTensor([drug1_idx])
        drug2 = torch.LongTensor([drug2_idx])

        cells_neighbors = [torch.LongTensor(neighbors).unsqueeze(0) for neighbors in cell_neighbors]
        drugs1_neighbors = [torch.LongTensor(neighbors).unsqueeze(0) for neighbors in drug1_neighbors]
        drugs2_neighbors = [torch.LongTensor(neighbors).unsqueeze(0) for neighbors in drug2_neighbors]

        # 获取一跳的注意力权重
        hop = 0  # 第一跳
        cell_attn_weights_1 = []
        drug1_attn_weights_1 = []
        drug2_attn_weights_1 = []

        # 获取细胞的一跳注意力权重
        neighbor_emb = self.model.protein_embedding(torch.LongTensor(cell_neighbors[hop])).float()
        item_emb = self.model.cell_embedding(torch.LongTensor([cell_idx])).float()

        # 确保 item_emb 和 neighbor_emb 的维度一致
        item_emb = item_emb.view(1, 1, -1)
        neighbor_emb = neighbor_emb.view(1, neighbor_emb.size(0), -1)

        # 计算全局注意力分数
        global_score = self.model.global_attn(neighbor_emb).squeeze(-1)

        # 计算局部注意力分数
        item_expand = item_emb.expand(-1, neighbor_emb.size(1), -1)
        pair_input = torch.cat([item_expand, neighbor_emb], dim=-1)
        local_score = self.model.local_attn(pair_input).squeeze(-1)

        # 合并注意力分数
        attn_weights = torch.softmax(global_score + local_score, dim=1)
        cell_attn_weights_1 = attn_weights.squeeze(0).detach().cpu().numpy()

        # 获取药物1的一跳注意力权重
        neighbor_emb = self.model.protein_embedding(torch.LongTensor(drug1_neighbors[hop])).float()
        item_emb = self.model.drug_embedding(torch.LongTensor([drug1_idx])).float()

        # 确保 item_emb 和 neighbor_emb 的维度一致
        item_emb = item_emb.view(1, 1, -1)
        neighbor_emb = neighbor_emb.view(1, neighbor_emb.size(0), -1)

        # 计算全局注意力分数
        global_score = self.model.global_attn(neighbor_emb).squeeze(-1)

        # 计算局部注意力分数
        item_expand = item_emb.expand(-1, neighbor_emb.size(1), -1)
        pair_input = torch.cat([item_expand, neighbor_emb], dim=-1)
        local_score = self.model.local_attn(pair_input).squeeze(-1)

        # 合并注意力分数
        attn_weights = torch.softmax(global_score + local_score, dim=1)
        drug1_attn_weights_1 = attn_weights.squeeze(0).detach().cpu().numpy()
        # 获取药物2的一跳注意力权重
        neighbor_emb = self.model.protein_embedding(torch.LongTensor(drug2_neighbors[hop])).float()
        item_emb = self.model.drug_embedding(torch.LongTensor([drug2_idx])).float()

        # 确保 item_emb 和 neighbor_emb 的维度一致
        item_emb = item_emb.view(1, 1, -1)
        neighbor_emb = neighbor_emb.view(1, neighbor_emb.size(0), -1)

        # 计算全局注意力分数
        global_score = self.model.global_attn(neighbor_emb).squeeze(-1)

        # 计算局部注意力分数
        item_expand = item_emb.expand(-1, neighbor_emb.size(1), -1)
        pair_input = torch.cat([item_expand, neighbor_emb], dim=-1)
        local_score = self.model.local_attn(pair_input).squeeze(-1)

        # 合并注意力分数
        attn_weights = torch.softmax(global_score + local_score, dim=1)
        drug2_attn_weights_1 = attn_weights.squeeze(0).detach().cpu().numpy()
        # 获取二跳的注意力权重
        hop = 1  # 第二跳
        cell_attn_weights = []
        drug1_attn_weights = []
        drug2_attn_weights = []

        # 获取细胞的二跳注意力权重
        neighbor_emb = self.model.protein_embedding(torch.LongTensor(cell_neighbors[hop])).float()
        item_emb = self.model.cell_embedding(torch.LongTensor([cell_idx])).float()

        # 确保 item_emb 和 neighbor_emb 的维度一致
        item_emb = item_emb.view(1, 1, -1)
        neighbor_emb = neighbor_emb.view(1, neighbor_emb.size(0), -1)

        # 计算全局注意力分数
        global_score = self.model.global_attn(neighbor_emb).squeeze(-1)

        # 计算局部注意力分数
        item_expand = item_emb.expand(-1, neighbor_emb.size(1), -1)
        pair_input = torch.cat([item_expand, neighbor_emb], dim=-1)
        local_score = self.model.local_attn(pair_input).squeeze(-1)

        # 合并注意力分数
        attn_weights = torch.softmax(global_score + local_score, dim=1)
        cell_attn_weights = attn_weights.squeeze(0).detach().cpu().numpy()

        # 获取药物1的二跳注意力权重
        neighbor_emb = self.model.protein_embedding(torch.LongTensor(drug1_neighbors[hop])).float()
        item_emb = self.model.drug_embedding(torch.LongTensor([drug1_idx])).float()

        # 确保 item_emb 和 neighbor_emb 的维度一致
        item_emb = item_emb.view(1, 1, -1)
        neighbor_emb = neighbor_emb.view(1, neighbor_emb.size(0), -1)

        # 计算全局注意力分数
        global_score = self.model.global_attn(neighbor_emb).squeeze(-1)

        # 计算局部注意力分数
        item_expand = item_emb.expand(-1, neighbor_emb.size(1), -1)
        pair_input = torch.cat([item_expand, neighbor_emb], dim=-1)
        local_score = self.model.local_attn(pair_input).squeeze(-1)

        # 合并注意力分数
        attn_weights = torch.softmax(global_score + local_score, dim=1)
        drug1_attn_weights = attn_weights.squeeze(0).detach().cpu().numpy()

        # 获取药物2的二跳注意力权重
        neighbor_emb = self.model.protein_embedding(torch.LongTensor(drug2_neighbors[hop])).float()
        item_emb = self.model.drug_embedding(torch.LongTensor([drug2_idx])).float()

        # 确保 item_emb 和 neighbor_emb 的维度一致
        item_emb = item_emb.view(1, 1, -1)
        neighbor_emb = neighbor_emb.view(1, neighbor_emb.size(0), -1)

        # 计算全局注意力分数
        global_score = self.model.global_attn(neighbor_emb).squeeze(-1)

        # 计算局部注意力分数
        item_expand = item_emb.expand(-1, neighbor_emb.size(1), -1)
        pair_input = torch.cat([item_expand, neighbor_emb], dim=-1)
        local_score = self.model.local_attn(pair_input).squeeze(-1)

        # 合并注意力分数
        attn_weights = torch.softmax(global_score + local_score, dim=1)
        drug2_attn_weights = attn_weights.squeeze(0).detach().cpu().numpy()

        # 合并注意力分数
        attn_weights = torch.softmax(global_score + local_score, dim=1)
        cell_attn_weights_3 = attn_weights.squeeze(0).detach().cpu().numpy()

        # 获取药物1的三跳注意力权重
        neighbor_emb = self.model.protein_embedding(torch.LongTensor(drug1_neighbors[hop])).float()
        item_emb = self.model.drug_embedding(torch.LongTensor([drug1_idx])).float()

        # 确保 item_emb 和 neighbor_emb 的维度一致
        item_emb = item_emb.view(1, 1, -1)
        neighbor_emb = neighbor_emb.view(1, neighbor_emb.size(0), -1)

        # 计算全局注意力分数
        global_score = self.model.global_attn(neighbor_emb).squeeze(-1)

        # 计算局部注意力分数
        item_expand = item_emb.expand(-1, neighbor_emb.size(1), -1)
        pair_input = torch.cat([item_expand, neighbor_emb], dim=-1)
        local_score = self.model.local_attn(pair_input).squeeze(-1)

        # 合并注意力分数
        attn_weights = torch.softmax(global_score + local_score, dim=1)
        drug1_attn_weights_3 = attn_weights.squeeze(0).detach().cpu().numpy()

        # 获取药物2的三跳注意力权重
        neighbor_emb = self.model.protein_embedding(torch.LongTensor(drug2_neighbors[hop])).float()
        item_emb = self.model.drug_embedding(torch.LongTensor([drug2_idx])).float()

        # 确保 item_emb 和 neighbor_emb 的维度一致
        item_emb = item_emb.view(1, 1, -1)
        neighbor_emb = neighbor_emb.view(1, neighbor_emb.size(0), -1)

        # 计算全局注意力分数
        global_score = self.model.global_attn(neighbor_emb).squeeze(-1)

        # 计算局部注意力分数
        item_expand = item_emb.expand(-1, neighbor_emb.size(1), -1)
        pair_input = torch.cat([item_expand, neighbor_emb], dim=-1)
        local_score = self.model.local_attn(pair_input).squeeze(-1)

        # 合并注意力分数
        attn_weights = torch.softmax(global_score + local_score, dim=1)
        drug2_attn_weights_3 = attn_weights.squeeze(0).detach().cpu().numpy()

        # 保存蛋白质权重到文件
        with open('first_hop_protein_weights.txt', 'w') as f:
            f.write(f"Cell Hop 1:\n{cell_attn_weights_1}\n")
            f.write(f"Drug1 Hop 1:\n{drug1_attn_weights_1}\n")
            f.write(f"Drug2 Hop 1:\n{drug2_attn_weights_1}\n")

        with open('second_hop_protein_weights.txt', 'w') as f:
            f.write(f"Cell Hop 2:\n{cell_attn_weights}\n")
            f.write(f"Drug1 Hop 2:\n{drug1_attn_weights}\n")
            f.write(f"Drug2 Hop 2:\n{drug2_attn_weights}\n")

        with open('third_hop_protein_weights.txt', 'w') as f:
            f.write(f"Cell Hop 3:\n{cell_attn_weights_3}\n")
            f.write(f"Drug1 Hop 3:\n{drug1_attn_weights_3}\n")
            f.write(f"Drug2 Hop 3:\n{drug2_attn_weights_3}\n")

if __name__ == '__main__':
    case_study = CaseStudy(
        config_path='saved/models/DrugCombDB/0610_160322/config.json',
        model_path='saved/models/DrugCombDB/0610_160322/model_best.pth',
        data_dir='data/DrugCombDB'
    )
    cell_idx = 50  # 替换为特定的细胞系索引
    drug1_idx = 321  # 替换为药物1的索引
    drug2_idx = 292  # 替换为药物2的索引
    case_study.predict(cell_idx, drug1_idx, drug2_idx)