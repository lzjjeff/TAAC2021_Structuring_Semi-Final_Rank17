#
# 数据集及训练/预测阶段所需函数


import os
import glob
import h5py
import random
import numpy as np
import torch
import config.config as cfg
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from prefetch_generator import BackgroundGenerator
from utils.data_utils import load_data_from_pkl
from copy import deepcopy
from transformers import AutoTokenizer


# text_tokenizer = AutoTokenizer.from_pretrained(cfg.TaggingModelConfig["text_model_config"]["bert_model_path"])


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class MultimodalDataset(Dataset):
    def __init__(self, data_dir: str, label2id: dict = None, split: list = None, skip: int = 1):
        """
        SceneSeg和Tagging任务共用的数据集结构
        :param data_dir: 特征文件目录
        :param label2id: 标签->编号 的映射字典
        :param split: train_split/dev_split, 为 os.listdir(data_dir) 的子集
        :param skip: 跳帧步长，为k则每k帧抽一次
        """
        self.data_dir = data_dir
        if split is not None:
            self.sample_paths = [os.path.join(data_dir, sample_path) for sample_path in split]
        else:
            files = os.listdir(data_dir)
            rm_files = []
            for file in files:
                if os.path.isfile(os.path.join(data_dir, file)):
                    rm_files.append(file)
            for rm_file in rm_files:
                files.remove(rm_file)
            self.sample_paths = [os.path.join(data_dir, sample_path) for sample_path in files]

        self.label2id = label2id
        self.skip = skip

    def __getitem__(self, idx):
        sample_path = self.sample_paths[idx]
        id = os.path.basename(sample_path)
        video_fea = load_data_from_pkl(os.path.join(sample_path, 'video.pkl'))
        mod = len(video_fea) % self.skip
        expand_num = self.skip - mod if mod > 0 else 0
        video_fea = np.concatenate([video_fea, np.zeros((expand_num, video_fea.shape[1]))], axis=0)
        video_fea = video_fea.reshape((-1, self.skip, video_fea.shape[1]))
        try:
            text = load_data_from_pkl(os.path.join(sample_path, 'text.pkl'))
        except:
            text = []
        if text == []:
            text_ids, text_contents, text_fea = [], [], []
        else:
            text_ids, text_contents, text_fea = text
            text_ids = [text_id // self.skip for text_id in text_ids]

        try:
            shots_ori = load_data_from_pkl(os.path.join(sample_path, 'shots.pkl'))
            shots_skip = [[shot[0] // self.skip, shot[1] // self.skip] for shot in shots_ori]
        except:
            shots_ori, shots_skip = None, None
        try:
            segments_ori = load_data_from_pkl(os.path.join(sample_path, 'segments.pkl'))
            segments_skip = [[seg[0] // self.skip, seg[1] // self.skip] for seg in segments_ori]
            labels = load_data_from_pkl(os.path.join(sample_path, 'labels.pkl'))
            labels = [[self.label2id[l] for l in label_list] for label_list in labels]
            labels4pretrain = load_data_from_pkl(os.path.join(sample_path, 'labels4pretrain.pkl'))
            labels4pretrain = [self.label2id[l] for l in labels4pretrain]
        except:  # for inference
            segments_ori, segments_skip, labels, labels4pretrain = None, None, None, None

        data = {"id": id,
                "video": video_fea,
                "text": [text_ids, text_contents, text_fea],
                "shots_ori": shots_ori,
                "shots_skip": shots_skip,
                "segments_ori": segments_ori,
                "segments_skip": segments_skip,
                "labels": labels,
                "labels4pretrain": labels4pretrain}

        return data

    def __len__(self):
        return len(self.sample_paths)


def collate_fn(batch: list):
    """
    被 torch.utils.data.Dataloader 调用，整理一个batch的数据
    """
    ids = []
    video_frames_list = []
    text_pair_list = []
    shots_ori = []
    shots_skip = []
    segments_ori = []
    segments_skip = []
    labels = []
    labels4pretrain = []

    for inputs in batch:
        video_frames_list.append(inputs["video"])
        text_pair_list.append(inputs["text"])
        ids.append(inputs["id"])
        shots_ori.append(inputs["shots_ori"])
        shots_skip.append(inputs["shots_skip"])
        segments_ori.append(inputs["segments_ori"])
        segments_skip.append(inputs["segments_skip"])
        labels.append(inputs["labels"])
        labels4pretrain.append(inputs["labels4pretrain"])

    return ids, video_frames_list, text_pair_list, shots_ori, shots_skip, segments_ori, segments_skip, labels, labels4pretrain


def sort_sequences(lengths: list, descending=True):
    """
    对一个长度序列进行从小到大排序
    :param lengths: 长度序列
    :param descending: 是否降序
    :return: 排序后的序列, 排序的序号列表, 重排序的序号列表(用于恢复到原来的顺序)
    """
    lengths = torch.LongTensor(lengths)
    lengths_sorted, sorted_idx = lengths.sort(descending=descending)
    _, unsorted_idx = sorted_idx.sort()
    return lengths_sorted, sorted_idx, unsorted_idx


def sort_seq_by_idx(seq, idx):
    """
    根据idx对seq进行排序
    :param seq: 序列, list or torch.Tensor or np.ndarray; shape of (N, ...)
    :param idx: 序号列表; list or torch.Tensor or np.ndarray; shape of (N)
    :return: 排序后的序列
    """
    assert len(seq) == len(idx)
    return [seq[i] for i in idx]


def data_augmentation(ids, video_frames_list, segments):
    new_ids = deepcopy(ids)
    new_video_frames_list = deepcopy(video_frames_list)
    new_segments = segments
    for id, video_frames, segs in zip(ids, video_frames_list, segments):
        scenes = []
        for seg in segs:
            scenes.insert(0, video_frames[seg[0]:seg[1]])
        segs = []
        p = 0
        for scene in scenes:
            segs.append([p, p + len(scene)])
            p += len(scene)
        frames = np.concatenate(scenes, axis=0)
        new_video_frames_list.append(frames)
        new_segments.append(segs)
        new_ids.append(id)

    return new_ids, new_video_frames_list, new_segments


def prepare_for_scene_seg(ids: list,
                          video_frames_list: list,
                          # face_frames_list:list = None,
                          # audio_frames_list: list = None,
                          segments_ori: list = None,
                          segments_skip: list = None,
                          train=True):
    """
    准备SceneSeg模型的输入数据
    :param ids:
    :param video_frames_list:
    :param audio_frames_list:
    :param segments:
    :return:
    """

    seq_lens = [len(video_frames) for video_frames in video_frames_list]  # shot number of video
    video_dim = video_frames_list[0].shape[-1]

    seq_lens = torch.LongTensor(seq_lens)
    video_frames_tensor = pad_sequence([torch.FloatTensor(f) for f in video_frames_list], batch_first=True).cuda()

    return ids, video_frames_tensor, seq_lens, segments_ori, segments_skip


def prepare_for_tagging(ids: list, video_frames_list: list, text_pair_list: list, segments: list, labels: list = None):
    """
    准备Tagging模型的输入数据
    :param ids:
    :param video_frames_list:
    :param audio_frames_list:
    :param segments:
    :param labels:
    :return:
    """
    MAX_SCENE_LEN = cfg.max_scene_len
    MAX_TEXT_LEN = cfg.max_text_len
    SEG_LEN = cfg.frame_skip
    v_dim = video_frames_list[0].shape[-1]
    t_dim = 768

    scene_ids = []
    video_scenes, audio_scenes, text_scenes = [], [], []
    scene_lens = []
    for id, video_frames, (text_ids, text_contents, text_seqs), segs in zip(ids, video_frames_list,
                                                                           text_pair_list, segments):
        for seg in segs:
            text_scene = np.zeros((1, MAX_TEXT_LEN, t_dim))
            p = 0
            while len(text_ids) > 0:
                if text_ids[0] >= seg[0] and text_ids[0] < seg[1]:
                    seq_len = len(text_seqs[0])
                    text_scene[:, p: p+seq_len] = text_seqs[0][:min(MAX_TEXT_LEN-p, seq_len)]
                    text_ids = text_ids[1:]
                    text_seqs = text_seqs[1:]
                    p = min(p+seq_len, MAX_TEXT_LEN)
                else:
                    break

            text_scenes.append(text_scene)

            padded_video_scene = np.zeros((1, MAX_SCENE_LEN, SEG_LEN, v_dim))
            video_scene = video_frames[seg[0]:seg[1]]
            length_v = min(MAX_SCENE_LEN, len(video_scene))
            padded_video_scene[:, :length_v] = video_scene[:length_v]
            video_scenes.append(padded_video_scene)

            scene_ids.append(id)
            scene_lens.append(max(length_v, 1))

    scene_lens = torch.LongTensor(scene_lens)

    if labels is not None:
        scene_labels = [l for lbls in labels for l in lbls]
    else:
        scene_labels = None

    video_scenes_tensor = torch.FloatTensor(np.concatenate(video_scenes, axis=0)).cuda()
    text_scenes_tensor = torch.FloatTensor(np.concatenate(text_scenes, axis=0)).cuda()

    return scene_ids, video_scenes_tensor, text_scenes_tensor, scene_lens, scene_labels


def prepare_for_tagging_pretrain(ids: list, video_frames_list: list, text_pair_list: list, labels: list = None):
    """
    准备Tagging模型的输入数据
    :param ids:
    :param video_frames_list:
    :param audio_frames_list:
    :param segments:
    :param labels:
    :return:
    """
    MAX_SCENE_LEN = cfg.max_scene_len
    MAX_TEXT_LEN = cfg.max_text_len
    v_dim = video_frames_list[0].shape[-1]
    t_dim=768

    scene_ids = []
    video_scenes, audio_scenes, text_scenes = [], [], []
    scene_lens, text_seq_lens = [], []
    for id, video_frames, (text_ids, text_contents, text_seqs) in zip(ids, video_frames_list, text_pair_list):
        padded_video_scene = np.zeros((1, MAX_SCENE_LEN, v_dim))
        length_v = min(MAX_SCENE_LEN, video_frames.shape[0])
        padded_video_scene[:, :length_v] = video_frames[:length_v]
        video_scenes.append(padded_video_scene)

        padded_text_scene = np.zeros((1, MAX_TEXT_LEN, t_dim))
        if len(text_seqs) > 0:
            text_seqs = np.concatenate(text_seqs, axis=0)
            length_t = min(MAX_TEXT_LEN, text_seqs.shape[0])
            padded_text_scene[:, :length_t] = text_seqs[:length_t]
            text_seq_lens.append(max(length_t, 1))

        text_scenes.append(padded_text_scene)

        scene_lens.append(max(length_v, 1))
        scene_ids.append(id)

    scene_lens = torch.LongTensor(scene_lens)

    video_scenes_tensor = torch.FloatTensor(np.concatenate(video_scenes, axis=0)).cuda()
    text_scenes_tensor = torch.FloatTensor(np.concatenate(text_scenes, axis=0)).cuda()

    return scene_ids, video_scenes_tensor, text_scenes_tensor, scene_lens, labels


def convert_segments_to_seq(scene_seg_logits: torch.FloatTensor, segments: list, seq_lens) -> [torch.FloatTensor, torch.LongTensor]:
    """ For SceneSeg train
    将segment pairs映射为0/1序列
    :param scene_seg_logits: SceneSeg模型的输出
    :param segments: SceneSeg任务的ground truth
    :param seq_lens: 长度列表, [frame_num_1, frame_num_2, ...]
    :return: torch.FloatTensor with shape of (sum of frame_num, 2)
             torch.LongTensor with (N)
    """
#     _, scene_pred = torch.softmax(scene_seg_logits, dim=-1).max(-1)
    scene_pred = torch.sigmoid(scene_seg_logits)
    segments_truth = torch.zeros_like(scene_pred)
    mask = torch.zeros_like(scene_pred)

    for i, (segs, seq_len) in enumerate(zip(segments, seq_lens)):
        for seg in segs:
            if seg[0] >= len(segments_truth[i]):
                continue
            segments_truth[i][seg[0]] = 1
        mask[i][:seq_len] = 1

    mask = mask.to(torch.bool)  # 去掉padding的部分
    segments_pred = torch.masked_select(scene_seg_logits, mask).view(-1, 1)
    segments_truth = torch.masked_select(segments_truth, mask).view(-1, 1)


    return segments_pred, segments_truth


def convert_seq_to_segments(scene_seq: torch.BoolTensor, seq_lens) -> list:
    """ For SceneSeg inference
        将SceneSeg模型的输出映射为segment pairs
    :param scene_seg_logits: 模型的输出
    :param seq_lens: 长度列表, [frame_num_1, frame_num_2, ...]
    :return: segment pairs列表 = [[segment_1, segment_2, ...], [...], ...]
    """
    
    segments = []
    for pred, seq_len in zip(scene_seq, seq_lens):
        pred = pred[:seq_len]
        segs = []
        idx = np.argwhere(pred.cpu().numpy() == 1).reshape(-1).tolist()
        idx.append(0)
        idx.append(seq_len.item())
        idx = list(set(idx))
        idx.sort()
        for i in range(len(idx) - 1):
            segs.append([idx[i], idx[i + 1]])
        segments.append(segs)

    return segments


def convert_labels_to_seq(labels: list) -> torch.LongTensor:
    """ For Tagging train
        将标签映射为0/1序列
    :param labels: 标签列表 = [[label_1, label_2, ...], [...], ...]
    :return: 0/1序列，1表示分为该位置序号对应的类别，0则不分
    """
    label_seqs = []
    for label in labels:
        label_seq = torch.zeros((1, 82))
        label_seq[:, torch.LongTensor(label)] = 1
        label_seqs.append(label_seq)

    return torch.cat(label_seqs, dim=0)


def convert_seq_to_labels(tagging_logits: torch.FloatTensor, id2label: dict, topk=20, theta=0.05) -> [list, list]:
    """ For Tagging inference
        将Tagging模型的输出映射为标签列表，取sigmoid处理后 >阈值 的输出
    :param tagging_logits: Tagging模型的输出
    :param id2label: 序号->标签 的映射字典
    :return: list_1 = [[label_1, label_2, ...], [...], ...] 标签列表
             list_2 = [[score_1, score_2, ...], [...], ...] 置信度得分列表(方法的正确性待验证)
    """
    tagging_seq = torch.sigmoid(tagging_logits)
    labels = []
    scores = []
    for logits in tagging_seq:
        idxes = np.argwhere(logits.cpu().numpy() > theta)
        score = torch.take(logits.cpu(), torch.LongTensor(idxes)).view(-1)
        score, idxes2 = score.sort(descending=True)
        score = score[:topk]
        idxes = idxes[idxes2][:topk]
        label = [id2label[idx.item()] for idx in idxes]
        scores.append(score.cpu().numpy().tolist())
        labels.append(label)

    return labels, scores


def get_sim_gt(sim_pred, segments):
    sim_gt = torch.zeros_like(sim_pred)
    for i, segment in enumerate(segments):
        for seg in segment:
            s = seg[0]
            e = seg[1]
            seg_len = e - s
            try:
                sim_gt[i, s:e, s:e] = 1 - torch.eye(seg_len)
            except:
                print(seg)
    return sim_gt
