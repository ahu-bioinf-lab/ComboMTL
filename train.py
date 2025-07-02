import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
from model.ComboMTL import ComboMTL as module_arch
from parse_config import ConfigParser
from trainer.trainer import Trainer
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler

# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    """Training."""
    logger = config.get_logger('train')  #

    # setup data_loader instances

    data_loader = config.init_obj('data_loader', module_data)  #data_loader
    #valid_data_loader = data_loader.split_dataset(valid=True)   # 将数据集中产生验证集
    test_data_loader = data_loader.split_dataset(test=True)   #从数据集中产生测试集

    data_loader = config.init_obj('data_loader', module_data)
    # 存储每次分割后的训练集和验证集

    feature_index = data_loader.get_feature_index()  #特征
    cell_neighbor_set = data_loader.get_cell_neighbor_set()  # cell的邻居
    drug_neighbor_set = data_loader.get_drug_neighbor_set()  #drug的邻居
    node_num_dict = data_loader.get_node_num_dict()       #节点的数目

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    # 从数据加载器中获取整个数据集
    full_dataset = data_loader.dataset

    # 存储每折的验证结果
    fold_results = []
    for fold_idx, (train_indices, valid_indices) in enumerate(kfold.split(full_dataset)):
        logger.info(f"********* Fold {fold_idx + 1} training *********")

        # Create samplers for training and validation
        # 使用索引创建训练集和验证集的采样器
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)

        # Create data loaders for the current fold
        # 创建当前折的训练和验证数据加载器
        train_loader = torch.utils.data.DataLoader(full_dataset,
                                                   batch_size=config['data_loader']['args']['batch_size'],
                                                   sampler=train_sampler)
        valid_loader = torch.utils.data.DataLoader(full_dataset,
                                                   batch_size=config['data_loader']['args']['batch_size'],
                                                   sampler=valid_sampler)

        # Initialize model, optimizer, and trainer for each fold
        # 每折重新初始化模型和优化器
        model = module_arch(protein_num=node_num_dict['protein'],
                        cell_num=node_num_dict['cell'],
                        drug_num=node_num_dict['drug'],
                        emb_dim=config['arch']['args']['emb_dim'],
                        n_hop=config['arch']['args']['n_hop'],
                        l1_decay=config['arch']['args']['l1_decay'],
                        therapy_method=config['arch']['args']['therapy_method'])
        logger.info(model)

    # get function handles of loss and metrics
        criterion = getattr(module_loss, config['loss']) #损失 在loss.py中有三种损失函数
        metrics = [getattr(module_metric, met) for met in config['metrics']]  #曲线 AUC PR等

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params) #优化函数

        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer) #学习率

        trainer = Trainer(model, criterion, metrics, optimizer,  #调用trainer进行训练
                      config=config,
                      data_loader=train_loader,
                      feature_index=feature_index,
                      cell_neighbor_set=cell_neighbor_set,
                      drug_neighbor_set=drug_neighbor_set,
                      valid_data_loader=valid_loader,
                      test_data_loader=test_data_loader,
                      lr_scheduler=lr_scheduler)
        trainer.train()

        """Testing."""
        logger = config.get_logger('test')
        logger.info(model)
        test_metrics = [getattr(module_metric, met) for met in config['metrics']]
    
        # load best checkpoint
        resume = str(config.save_dir / 'model_best.pth')
        logger.info('Loading checkpoint: {} ...'.format(resume))
        checkpoint = torch.load(resume)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)

        test_output = trainer.test()
        fold_log = {'loss': test_output['total_loss'] / test_output['n_samples']}
        fold_log.update({
        met.__name__: test_output['total_metrics'][i].item() / test_output['n_samples']
        for i, met in enumerate(metrics)})
        logger.info(f"Fold {fold_idx + 1} results: {fold_log}")
        fold_results.append(fold_log)
    # Calculate and log average results across all folds
    # 计算并记录所有折的平均结果
    avg_results = {metric: np.mean([fold[metric] for fold in fold_results]) for metric in fold_results[0]}
    logger.info(f"Average results across 5 folds: {avg_results}")
    '''
        log = {'loss': test_output['total_loss'] / test_output['n_samples']}
        log.update({
            met.__name__: test_output['total_metrics'][i].item() / test_output['n_samples'] \
            for i, met in enumerate(test_metrics)
    })
        logger.info(log)
    '''

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
