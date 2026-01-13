import random
from torch.utils.data import DataLoader, Sampler


class DialogueBatchSampler(Sampler):
    def __init__(self, dia_ids, utt_ids, mode='train'):
        self.mode = mode
        self.dia_groups = {}

        # 1. 按 Dialogue_id 分组
        for idx, (d_id, u_id) in enumerate(zip(dia_ids, utt_ids)):
            if d_id not in self.dia_groups:
                self.dia_groups[d_id] = []
            # 存储 (全局索引, 话语ID)
            self.dia_groups[d_id].append((idx, u_id))

        # 2. 组内排序 & 提取索引
        self.sorted_groups = []
        for d_id, items in self.dia_groups.items():
            # 按 u_id 从小到大排序，确保时序正确
            items.sort(key=lambda x: x[1])
            indices = [x[0] for x in items]

            # 【关键修改 1】：训练模式下，过滤掉长度小于 2 的对话
            # 这一步是为了解决 ValueError: Expected more than 1 value per channel
            if self.mode == 'train' and len(indices) < 2:
                continue

            self.sorted_groups.append(indices)

    def __iter__(self):
        indices = list(range(len(self.sorted_groups)))

        if self.mode == 'train':
            random.shuffle(indices)

        for i in indices:
            group = self.sorted_groups[i]

            yield group

    def __len__(self):
        return len(self.sorted_groups)


def get_dataloader(args, data):
    # 1. Train Loader (mode='train' 会丢弃单句对话)
    train_sampler = DialogueBatchSampler(
        data['train'].dia_ids,
        data['train'].utt_ids,
        mode='train'  # <--- 传入模式
    )
    train_dataloader = DataLoader(
        data['train'],
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 2. Dev Loader (mode='dev' 保留所有数据)
    dev_sampler = DialogueBatchSampler(
        data['dev'].dia_ids,
        data['dev'].utt_ids,
        mode='dev'
    )
    dev_dataloader = DataLoader(
        data['dev'],
        batch_sampler=dev_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 3. Test Loader
    test_sampler = DialogueBatchSampler(
        data['test'].dia_ids,
        data['test'].utt_ids,
        mode='test'
    )
    test_dataloader = DataLoader(
        data['test'],
        batch_sampler=test_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    return {
        'train': train_dataloader,
        'dev': dev_dataloader,
        'test': test_dataloader
    }