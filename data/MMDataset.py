from torch.utils.data import Dataset
import torch

__all__ = ['MMDataset']


class MMDataset(Dataset):
    # 修改 __init__ 接收新参数
    def __init__(self, label_ids, text_feats, video_feats, audio_feats, cons_text_feats, condition_idx, dia_ids,
                 utt_ids, speaker_ids):
        self.label_ids = label_ids
        self.text_feats = text_feats
        self.cons_text_feats = cons_text_feats
        self.condition_idx = condition_idx
        self.video_feats = video_feats
        self.audio_feats = audio_feats
        self.size = len(self.text_feats)

        # 新增存储
        self.dia_ids = dia_ids
        self.utt_ids = utt_ids
        self.speaker_ids = speaker_ids

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        sample = {
            'label_ids': torch.tensor(self.label_ids[index]),
            'text_feats': torch.tensor(self.text_feats[index]),
            'video_feats': torch.tensor(self.video_feats['feats'][index]),
            'audio_feats': torch.tensor(self.audio_feats['feats'][index]),
            'cons_text_feats': torch.tensor(self.cons_text_feats[index]),
            'condition_idx': torch.tensor(self.condition_idx[index]),
            'dia_id': self.dia_ids[index],
            'utt_id': self.utt_ids[index],
            'speaker_id': self.speaker_ids[index]
        }
        return sample