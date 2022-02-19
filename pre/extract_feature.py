# 特征提取相关函数

import os
import sys
import glob
import json
import time
import cv2
from tqdm import tqdm
import argparse
import librosa as lb

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModel, AutoTokenizer, ViTModel, Wav2Vec2Model, Wav2Vec2Processor
# from timesformer.models.vit import TimeSformer
from torchvision.models import resnet50

from utils.data_utils import read_shot_info, read_video_cv2, read_text, read_truth, read_audio, save_data_to_pkl, mp4_to_wav
import config.config as cfg


def load_extract_model(load_v=True, load_a=True, load_t=True):
    """
    加载用于提取特征的预训练模型，并返回一个dict
    :param load_v: 加载视频模型
    :param load_a: 加载音频模型
    :param load_t: 加载文本模型
    :return: {"<mode>": <model>, ...}
    """
    extract_model_dict = {"video": None, "audio": None, "text": None}
    if load_v:
        print('Loading video model ...')
        video_model_spatial = ViTModel.from_pretrained('./pretrain_models/vit-base-patch16-224-in21k')
#         video_model_spatial = resnet50()
#         video_model_spatial.load_state_dict(torch.load('/home/tione/notebook/algo-2021/dataset/pretrained_models/sceneseg/resnet50-19c8e357.pth'))
#         video_model_temporal = TimeSformer(img_size=224, num_frames=8, attention_type='divided_space_time',  pretrained_model='/home/tione/notebook/pretrained_models/TimeSformer_divST_8x32_224_K600.pyth')
        extract_model_dict["video"] = video_model_spatial.cuda()
        print('Finished.')
    if load_t:
        print('Loading text model ...')
        text_tokenizer = AutoTokenizer.from_pretrained("./pretrain_models/chinese-roberta-wwm-ext")
        text_model = AutoModel.from_pretrained("./pretrain_models/chinese-roberta-wwm-ext")
        extract_model_dict["text"] = [text_tokenizer, text_model.cuda()]
        print('Finished.')

    return extract_model_dict


def extract_video_spatial(video: np.ndarray, model: torch.nn.Module):
    """
    提取视频特征
    :param video: shape of (frame_num, 224, 224, 3)
    :param model:
    :return: tensor.FloatTensor, shape of (frame_num, dim)
    """

    video = video / 255     # 归一化
    video = torch.tensor(video, dtype=torch.float).permute(0, 3, 1, 2).cuda()

    dataset = TensorDataset(video)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    video_fea = []
    for batch in dataloader:
        batch = [t.cuda() for t in batch]
        model.eval()
        with torch.no_grad():
#             fea = model(batch[0])
            fea = model(batch[0]).pooler_output  # 提取特征
        # print(fea.shape)
        video_fea.append(fea.cpu())
        torch.cuda.empty_cache()

    return torch.cat(video_fea, dim=0).numpy()


def extract_video_temporal(video: np.ndarray, model: torch.nn.Module):
    """
    提取视频特征
    :param video: shape of (frame_num, 224, 224, 3)
    :param model:
    :return: tensor.FloatTensor, shape of (frame_num, dim)
    """

    video = video / 255     # 归一化
    video = torch.tensor(video, dtype=torch.float).permute(0, 3, 1, 2)

    video_stack8 = []
    for i in range(4, 0, -1):
        video_stack8.append(
            torch.cat([torch.zeros((i, video.shape[1], video.shape[2], video.shape[3])), video[:-i]],
                      dim=0).unsqueeze(1)[::12])
        torch.cuda.empty_cache()
    video_stack8.append(video.unsqueeze(1)[::12])
    for i in range(1, 4):
        video_stack8.append(
            torch.cat([video[i:], torch.zeros((i, video.shape[1], video.shape[2], video.shape[3]))],
                      dim=0).unsqueeze(1)[::12])
        torch.cuda.empty_cache()
    video_stack8 = torch.cat(video_stack8, dim=1).permute(0, 2, 1, 3, 4).cuda()

    dataset = TensorDataset(video_stack8)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    video_fea = []
    for batch in dataloader:
        batch = [t.cuda() for t in batch]
        model.eval()
        with torch.no_grad():
            fea = model.model.forward_features(batch[0])  # 提取特征
        # print(fea.shape)
        video_fea.append(fea.cpu())
        torch.cuda.empty_cache()

    return torch.cat(video_fea, dim=0).numpy()


def extract_audio(audio, sr):
    mfcc = lb.feature.mfcc(y=audio, sr=sr, hop_length=512, n_mfcc=13)
    mfcc_delta = lb.feature.delta(mfcc)
    mfcc_delta2 = lb.feature.delta(mfcc, order=2)
    comp_mfcc = np.concatenate((mfcc, mfcc_delta, mfcc_delta2)).T

    return comp_mfcc


def extract_text(text_list, model):
    tokenizer, model = model
    text_fea = []
    for text in text_list:
        inputs = tokenizer(text, max_length=512, return_tensors="pt")
        inputs.data = {k: v.cuda() for (k, v) in inputs.data.items()}
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            last_h = outputs.last_hidden_state
        torch.cuda.empty_cache()
        text_fea.append(last_h.squeeze(0).cpu().numpy())

    return text_fea


def read_and_extract_data(video_dir: str, text_dir: str, save_dir: str,
                          extract_v=True, extract_a=False, extract_t=False,
                          shot_dir: str = None, face_fea_dir: str = None, truth_path: str = None, pretrain_truth_path: str = None):
    """
    读取视频并提取特征并保存
    :param video_dir: 视频目录
    :param shot_dir: shots目录
    :param save_dir: 保存目录
    :param extract_v: 是否提取video特征
    :param extract_a: 是否提取audio特征
    :param extract_t: 是否提取text特征
    :param truth_path: ground truth文件路径，即train5k.txt
    :return: None
    """
    mp4_files = glob.glob(os.path.join(video_dir, '*.mp4'))     # 视频文件路径列表
    if shot_dir:
        shot_dict = read_shot_info(shot_dir)

    # 检查保存路径
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if extract_t:
        with open(os.path.join(text_dir), 'r', encoding='utf-8') as f:
            text_dict = json.load(f)

    if truth_path:
        # read ground truth
        with open(os.path.join(truth_path, 'train5k.txt'), 'r') as f:
            truth = json.loads(''.join(f.read().strip().split('\n')))

        pretrain_truth = {}
        id = None
        for filename in ['train.txt', 'val.txt']:
            with open(os.path.join(pretrain_truth_path, filename), 'r') as f:
                for line in f:
                    line = line.strip()
                    if line == "":
                        continue
                    elif line.startswith(".."):
                        id = os.path.basename(line).split('.')[0]
                    else:
                        pretrain_truth[id] = line.split(',')

    extract_model_dict = load_extract_model(load_v=extract_v, load_a=extract_a, load_t=extract_t)

    tic = time.time()

    print("Reading and extract data ...")
    for mp4 in tqdm(mp4_files):
        video_id = os.path.basename(mp4)
        sample_dir = os.path.join(save_dir, video_id.split('.mp4')[0])  # 该id视频的特征的保存路径
        if not os.path.exists(sample_dir):
            os.mkdir(sample_dir)
        existed_data_list = os.listdir(sample_dir)  # 检查已经提取过的视频

        if extract_v and 'video.pkl' not in existed_data_list:
            # 提取视频特征
            video_np, fps = read_video_cv2(mp4, resize=(224, 224))
            video_fea = extract_video_spatial(video_np, extract_model_dict["video"])
            save_data_to_pkl(video_fea, os.path.join(sample_dir, 'video.pkl'))

        if extract_a and 'audio.pkl' not in existed_data_list:
            # 提取音频特征
            wav = mp4.replace('mp4', 'wav')
            audio_np, sr = read_audio(wav)
            audio_fea = extract_audio(audio_np, sr)
            save_data_to_pkl(audio_fea, os.path.join(sample_dir, 'audio.pkl'))

        if extract_t and 'text.pkl' not in existed_data_list:
            # 提取文本特征
            text_id = video_id.replace('mp4', 'txt')
            try:
                frame_ids, text_list = read_text(text_dict, text_id)
                text_fea = extract_text(text_list, extract_model_dict["text"])
                text_fea = [frame_ids, text_list, text_fea]
            except:
                text_fea = []
            save_data_to_pkl(text_fea, os.path.join(sample_dir, 'text.pkl'))

        if truth_path and ('segments.pkl' not in existed_data_list or 'labels.pkl' not in existed_data_list or
                           'labels4pretrain.pkl' not in existed_data_list):
            # 提取该视频的ground truth信息
            if not extract_v or 'video.pkl' in existed_data_list:
                video_capture = cv2.VideoCapture()
                if not video_capture.open(mp4):
                    print(sys.stderr, 'Error: Cannot open video file ' + mp4)
                    return
                fps = video_capture.get(5)
            segs, tags, pretrain_tags = read_truth(truth, pretrain_truth, video_id, fps)
            save_data_to_pkl(segs, os.path.join(sample_dir, 'segments.pkl'))
            save_data_to_pkl(tags, os.path.join(sample_dir, 'labels.pkl'))
            save_data_to_pkl(pretrain_tags, os.path.join(sample_dir, 'labels4pretrain.pkl'))

        if shot_dir and 'shots.pkl' not in existed_data_list:
            # 提取该视频的shots信息
            save_data_to_pkl(shot_dict[video_id.split('.mp4')[0]], os.path.join(sample_dir, 'shots.pkl'))
    print(time.time()-tic)


def run_mp42wav(mp4_dir):
    mp4_files = glob.glob(os.path.join(mp4_dir, '*.mp4'))  # 视频文件路径列表
    wav_files = [file.replace('mp4', 'wav') for file in mp4_files]  # 音频文件路径列表
    for mp4, wav in tqdm(zip(mp4_files, wav_files), desc="process ..."):
        if os.path.basename(wav) not in os.listdir(mp4_dir):
            mp4_to_wav(mp4, wav)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ex_train", action="store_true", default=False)
    parser.add_argument("--ex_test", action="store_true", default=False)
    parser.add_argument("--mp42wav_train", action="store_true", default=False)
    parser.add_argument("--mp42wav_test", action="store_true", default=False)

    args = parser.parse_args()

    if not os.path.exists(cfg.save_path):
        os.mkdir(cfg.save_path)

    if args.mp42wav_train:
        run_mp42wav(cfg.train_video_dir)

    if args.mp42wav_test:
        run_mp42wav(cfg.test_video_dir)

    if args.ex_train:
        read_and_extract_data(cfg.train_video_dir, cfg.train_text_dir,
                              extract_v=True, extract_a=False, extract_t=False,
                              save_dir=cfg.train_data_dir, truth_path=cfg.train_truth_path, pretrain_truth_path=cfg.train_pretrain_truth_path)

    if args.ex_test:
        read_and_extract_data(cfg.test_video_dir, cfg.test_text_dir,
                              extract_v=True, extract_a=False, extract_t=False,
                              save_dir=cfg.test_data_dir)

