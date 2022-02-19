# 数据处理相关函数

import os
import sys
import glob
import json
import h5py
import pickle
import time
import random
import cv2
import numpy as np
import librosa as lb
import librosa.display
import config.config as cfg
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def load_data_from_pkl(data_path: str):
    if os.path.getsize(data_path) > 0:
        with open(data_path, 'rb') as f:
            unpickler = pickle.Unpickler(f)
            return unpickler.load()


def save_data_to_pkl(data, save_path: str):
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)


def mp4_to_wav(mp4_path, wav_path, sampling_rate=16000):
    """
    mp4 转 wav
    :param mp4_path: .mp4文件路径
    :param wav_path: .wav文件路径
    :param sampling_rate: 采样率
    :return: .wav文件
    """
    if os.path.exists(wav_path):
        os.remove(wav_path)
    command = "ffmpeg -y -i {} -f wav -ac 1 -ar {} {} -loglevel quiet".format(mp4_path, sampling_rate, wav_path)
    os.system(command)


def read_video_cv2(filename: str, max_num_frames: int = 99999, resize: tuple = None):
    """
    读取视频帧数据，并对图像进行resize
    :param filename: 视频路径
    :param max_num_frames: 最大帧数目
    :param resize: resize大小, (height, weight)
    :return: np.ndarray, shape of (frame_num, h, w, 3)
    """
    video_capture = cv2.VideoCapture()
    if not video_capture.open(filename):
        print(sys.stderr, 'Error: Cannot open video file ' + filename)
        return

    fps = video_capture.get(5)  # 视频的fps信息
    frame_num = int(video_capture.get(7))   # 视频的帧数目
    max_num_frames = min(max_num_frames, frame_num)
    frame_all = []

    for i in range(max_num_frames):
        has_frames, frame = video_capture.read()
        if resize:
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_LINEAR)
        frame_all.append(np.expand_dims(frame, axis=0))

    return np.concatenate(frame_all, axis=0), fps


def read_audio(filename, max_num_frames=99999):
    # 读取视频的音频数据
    if not os.path.exists(filename):
        print("%s 不存在，将从视频中提取 ..." % filename)
        video_file = filename.replace('wav', 'mp4')
        mp4_to_wav(video_file, filename, sampling_rate=16000)
        print("\r已提取音频： %s" % filename)

    wav_input_16khz, sr = lb.load(filename)
    return wav_input_16khz, sr


# def read_text(filename):
#     with open(filename, 'r') as f:
#         res = json.load(f)
#     ocr = res["video_ocr"]
#     asr = res["video_asr"]
#     return ocr, asr
# def read_text(filename):
#     with open(filename, 'r', encoding="utf-8") as f:
#         data = f.read().strip().split('\n')
#         text_list = []
#         for line in data:
#             line = line.replace('\'', '')[1:-1]
#             text, frame_id = line.split(', ')
#             text_list.append((int(frame_id), text.replace(' | ', '')))
#     return text_list
def read_text(text_dict, text_id):
    text_list = text_dict[text_id]
    frame_ids = [p[1] for p in text_list][::-1]
    text_list = [p[0] for p in text_list][::-1]
    return frame_ids, text_list


def read_truth(label_all: dict, label_pretrain_all, video_id: str, fps: float):
    """
    读取ground truth信息, 并将segment转换为在frame上对应的位置
    :param label_all: 从train5k.txt文件读取的dict数据
    :param video_id: 视频id(文件名)
    :param fps: 视频帧率
    :return: list_1 = [segment_1, segment_2, ...], segment_i = (start, end)
             list_2 = [label_1, label_2, ...]
    """
    segments = []
    tags, tags_pretrain = [], []
    for dic in label_all[video_id]["annotations"]:
        s, e = dic['segment']
        segments.append([int(s*fps), int(e*fps)])
        tags.append(dic['labels'])
        tags_pretrain = label_pretrain_all[video_id.split('.mp4')[0]]
    return segments, tags, tags_pretrain


def read_shot_info(shot_dir):
    """
    读取TransNetv2切分的shots边界，返回一个dict
    :param shot_dir: str, shots文件目录
    :return: dict = {vid_1: [shot_1, shot_2, ...], vid_2: [...], ...}, vid是视频的文件名
    """
    shot_files = glob.glob(os.path.join(shot_dir, '*.scenes.txt'))
    shots = {}
    for shot_file in shot_files:
        id = os.path.basename(shot_file).split('.scenes.txt')[0]
        with open(shot_file, 'r', encoding='utf-8') as f:
            shots[id] = [(int(p[0]), int(p[1])) for p in [l.strip().split(' ') for l in f.read().strip().split('\n')]]
    return shots


def read_label_ids(data_path):
    """
    读取tags类别及编号, 共82种tags, 返回两个dict
    :param data_path: str, label_ids.txt文件路径
    :return: dict_1 = {id_1: label_1, id_2: label_2, ...},
             dict_2 = {label_1: id_1, label_2: id_2, ...}
    """
    id2label = {}
    label2id = {}
    with open(data_path, 'r') as f:
        lines = f.read().strip().split('\n')
    for line in lines:
        label, id = line.strip().split('\t')
        id = int(id)
        id2label[id] = label
        label2id[label] = id

    return id2label, label2id


def split_train_dev(data_dir: str, dev_size: float = 0.1, random_state=123456):
    """
    划分训练集和验证集
    :param data_dir: 训练数据的特征文件目录
    :param dev_size: dev集比例
    :return: list_1 = [train_feat_path_1, train_feat_path_2, ...],
             list_2 = [dev_feat_path_1, dev_feat_path_2, ...]
    """
    files = os.listdir(data_dir)
    rm_files = []
    for file in files:
        if os.path.isfile(os.path.join(data_dir, file)):
            rm_files.append(file)
    for rm_file in rm_files:
        files.remove(rm_file)
    # dev_size = int(dev_size * len(files))
    # train_split = files[dev_size:]
    # dev_split = files[:dev_size]
    train_split, dev_split = train_test_split(files, test_size=dev_size, shuffle=True, random_state=random_state)

    return train_split, dev_split


def to_h5py(data_dir, save_dir):
    sample_paths = [os.path.join(data_dir, sample_path) for sample_path in os.listdir(data_dir)]
    id2label, label2id = read_label_ids(data_path=cfg.labels_path)

    shots_dict = {}
    segments_dict = {}
    labels_dict = {}

    with h5py.File(os.path.join(data_dir, 'features.hdf5'), 'w') as f:
        for i in tqdm(range(len(sample_paths))):
            sample_path = sample_paths[i]
            id = os.path.basename(sample_path)

            grp = f.create_group(id)
            grp.attrs["id"] = id

            video_fea = load_data_from_pkl(os.path.join(sample_path, 'video.pkl'))
            audio_fea = load_data_from_pkl(os.path.join(sample_path, 'audio.pkl'))
            # text_fea = load_data_from_pkl(os.path.join(sample_path, 'text.pkl'))
            d_video = grp.create_dataset('video', video_fea.shape)
            d_audio = grp.create_dataset('audio', audio_fea.shape)
            d_video.write_direct(video_fea)
            d_audio.write_direct(audio_fea)

            shots = load_data_from_pkl(os.path.join(sample_path, 'shots.pkl'))
            try:
                segments = load_data_from_pkl(os.path.join(sample_path, 'segments.pkl'))
                labels = load_data_from_pkl(os.path.join(sample_path, 'labels.pkl'))
            except:  # for inference
                segments = None
                labels = None

            shots_dict[id] = shots
            segments_dict[id] = segments
            labels_dict[id] = labels

    save_data_to_pkl(shots_dict, os.path.join(save_dir, "shots.pkl"))
    save_data_to_pkl(segments_dict, os.path.join(save_dir, "segments.pkl"))
    save_data_to_pkl(labels_dict, os.path.join(save_dir, "labels.pkl"))


if __name__ == '__main__':
    to_h5py(cfg.train_data_dir, cfg.train_data_dir)
    to_h5py(cfg.test_data_dir, cfg.test_data_dir)

