import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, test_split, num_workers, collate_fn=default_collate):
        self.validation_split = validation_split  # 验证集分割
        self.test_split = test_split      #测试集分割
        self.shuffle = shuffle     #打乱

        self.batch_idx = 0    # batch_idx
        self.n_samples = len(dataset)  # the length of dataset

        self.train_sampler, self.valid_sampler, self.test_sampler = self._split_sampler() # 训练集 验证集 测试集

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.train_sampler, **self.init_kwargs)

    def _split_sampler(self):    # 样本分割
        idx_full = np.arange(self.n_samples)  #建立一个样本长度的数组，存放id

        np.random.seed(0)
        np.random.shuffle(idx_full)  #把id打乱

        if isinstance(self.validation_split, int) or isinstance(self.test_split, int):  #确立验证机和测试集
            assert self.validation_split > 0 or self.test_split > 0
            assert self.validation_split < self.n_samples or self.test_split < self.n_samples, \
                "validation set size is configured to be larger than entire dataset."
            len_valid = self.validation_split
            len_test  = self.test_split
        else:
            len_valid = int(self.n_samples * self.validation_split) # length of validation
            len_test  = int(self.n_samples * self.test_split)  #length of test

        valid_idx = idx_full[0:len_valid]  #验证集的id
        test_idx  = idx_full[len_valid: (len_valid+len_test)]  #测试集的id
        train_idx = np.delete(idx_full, np.arange(0, len_valid+len_test))  #全集减去验证机和测试集，剩下的就是训练集的id

        train_sampler = SubsetRandomSampler(train_idx)  #随机采样，抽样器
        valid_sampler = SubsetRandomSampler(valid_idx)  #会根据后面给的列表从数据集中按照下标取元素
        test_sampler  = SubsetRandomSampler(test_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False  #关闭打乱现象
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler, test_sampler

    def split_dataset(self, valid=False, test=False):
        if valid:
            assert len(self.valid_sampler) != 0, "validation set size ratio is not positive"
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
        if test:
            assert len(self.test_sampler) != 0, "test set size ratio is not positive"
            return DataLoader(sampler=self.test_sampler, **self.init_kwargs)
