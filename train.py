import os
import sys
import glob
import json
import pickle
from tqdm import tqdm
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_cosine_schedule_with_warmup
import cv2
from sklearn.metrics import f1_score

import config.config as cfg
from src.data.dataset import DataLoaderX, MultimodalDataset, collate_fn, prepare_for_scene_seg, prepare_for_tagging, \
    prepare_for_tagging_pretrain, convert_segments_to_seq, convert_labels_to_seq, convert_seq_to_segments
from utils.data_utils import read_label_ids, split_train_dev
from src.models import SceneSegMODEL, TaggingMODEL
from utils.metrics import get_ap, get_mAP_seq, calculate_gap, get_Miou, get_time_f1, FocalLoss


LOSS_FUNC = {'ce': nn.CrossEntropyLoss(),
             'bce': nn.BCEWithLogitsLoss(),
             'fc': FocalLoss(gamma=2),
             'mse': nn.MSELoss()}


def train_scene_seg():
    """ SceneSeg训练过程 """

    train_split, dev_split = split_train_dev(cfg.train_data_dir, cfg.dev_size, cfg.seed)   # 划分数据集
    id2label, label2id = read_label_ids(data_path=cfg.labels_path)
    print(len(id2label.keys()))

    # 加载数据
    train_dataset = MultimodalDataset(cfg.train_data_dir, label2id, split=train_split, skip=cfg.frame_skip)
    dev_dataset = MultimodalDataset(cfg.train_data_dir, label2id, split=dev_split, skip=cfg.frame_skip)

    print("number of train samples: ", len(train_dataset))
    print("number of dev samples: ", len(dev_dataset))

    train_dataloader = DataLoaderX(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_dataloader = DataLoaderX(dev_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # 创建模型
    scene_seg_model = SceneSegMODEL[cfg.SceneSegModelConfig["name"]](cfg.SceneSegModelConfig).cuda()
#     scene_seg_model.load_state_dict(torch.load(os.path.join(cfg.save_path, "scene_seg_model_pretrain.std")))
    
    loss_func = LOSS_FUNC[cfg.SceneSegTrainConfig["loss_func"]]
    
    optimizer = optim.Adam(scene_seg_model.parameters(), lr=cfg.SceneSegTrainConfig["lr"],
                           weight_decay=cfg.SceneSegTrainConfig["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)

    # 开始训练
    last_lr = float('inf')
    # best_loss = float('inf')
    best_tmf = 0.0
    for epoch in tqdm(range(cfg.SceneSegTrainConfig["epoch"]), desc="Training SceneSeg model ..."):
        train_loss = 0.0
        train_size = 0

        scene_seg_model.train()
        optimizer.zero_grad()
        for batch in tqdm(train_dataloader):
            # print(batch)
            ids, video_frames_list, text_pair_list, _, _, segments_ori, segments_skip, labels, labels4pretrain = batch
            ids, video_frames_tensor, seq_lens, segments_ori, segments_skip = \
                prepare_for_scene_seg(ids, video_frames_list, segments_ori, segments_skip)  # 准备模型的输入数据
            scene_seg_logits = scene_seg_model(video_frames_tensor, seq_lens)

            segments_pred, segments_truth = convert_segments_to_seq(scene_seg_logits, segments_skip, seq_lens)   # 转换预测结果和ground truth形式
            loss = loss_func(segments_pred, segments_truth.to(segments_pred.device))

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # print(loss.item())
            batch_size = len(video_frames_list)
            train_loss += loss.item() * batch_size
            train_size += batch_size

        train_loss /= train_size
        print("Train loss: ", train_loss)
        print("Number of train samples: ", train_size)

        # validation
        preds_skip, targets_skip, preds_span_skip, targets_span_skip = [], [], [], []
        preds_ori, targets_ori, time_spans_pred, time_spans_truth = [], [], [], []
        all_seq_lens = []
        dev_loss = 0.0
        dev_size = 0
        scene_seg_model.eval()
        for batch in tqdm(dev_dataloader):
            ids, video_frames_list, text_pair_list, _, _, segments_ori, segments_skip, labels, labels4pretrain = batch
            ids, video_frames_tensor, seq_lens, segments_ori, segments_skip = \
                prepare_for_scene_seg(ids, video_frames_list, segments_ori, segments_skip, train=False)
            scene_seg_logits = scene_seg_model(video_frames_tensor, seq_lens)
            scene_seq = torch.sigmoid(scene_seg_logits) > cfg.theta

            segments_pred, segments_truth = convert_segments_to_seq(scene_seg_logits, segments_skip, seq_lens)
            segments_pairs_pred = convert_seq_to_segments(scene_seq, seq_lens)
            loss = loss_func(segments_pred, segments_truth.to(segments_pred.device))

            batch_size = len(video_frames_list)
            dev_loss += loss.item() * batch_size
            dev_size += batch_size

#             preds_skip.append(torch.softmax(segments_pred, dim=-1).cpu().detach().numpy())
            preds_skip.append(torch.sigmoid(segments_pred).cpu().detach().numpy())
            targets_skip.append(segments_truth.cpu().detach().numpy())

            preds_span_skip.extend(segments_pairs_pred)
            targets_span_skip.extend(segments_skip)

            for id, segment_pred, segment_ori in zip(ids, segments_pairs_pred, segments_ori):
                id = id + '.mp4'
                mp4 = os.path.join(cfg.train_video_dir, id)
                video_capture = cv2.VideoCapture()
                if not video_capture.open(mp4):
                    print(sys.stderr, 'Error: Cannot open video file ' + mp4)
                    return
                fps = video_capture.get(5)
                max_frame = int(video_capture.get(7))

                time_span_pred = [[round(p[0]*cfg.frame_skip/fps, 3), round(p[1]*cfg.frame_skip/fps, 3)] for p in segment_pred]
                time_span_truth = [[round(p[0]/fps, 3), round(p[1]/fps, 3)] for p in segment_ori]
                time_span_pred[-1][-1] = min(max_frame/fps, time_span_pred[-1][-1])

                time_spans_pred.append(time_span_pred)
                time_spans_truth.append(time_span_truth)

            all_seq_lens.append(seq_lens.cpu().detach().numpy())

        preds_skip = np.concatenate(preds_skip)
        targets_skip = np.concatenate(targets_skip)

        dev_loss /= dev_size
        print("Dev loss: ", dev_loss)

        ap = get_ap(targets_skip, preds_skip)
        print("AP: {:.3f}".format(ap))

        f1 = f1_score(targets_skip, preds_skip > cfg.theta)
        print("F1: {:.3f}".format(f1))

#         miou_skip = get_Miou(preds_span_skip, targets_span_skip)
#         print("Skip Miou: {:.3f}".format(miou_skip))

        miou_ori = get_Miou(time_spans_pred, time_spans_truth)
        print("Time Miou: {:.3f}".format(miou_ori))

        time_f1_mean = get_time_f1(time_spans_pred, time_spans_truth)
        print("Time F1 Mean: {:.3f}".format(time_f1_mean))
        tmf_mean = miou_ori * time_f1_mean
        print("Time Miou x F1 Mean: {:.3f}".format(tmf_mean))
        
        time_f1 = get_time_f1(time_spans_pred, time_spans_truth, mean=False)
        print("Time F1: {:.3f}".format(time_f1))
        tmf = miou_ori * time_f1
        print("Time Miou x F1: {:.3f}".format(tmf))

        # if dev_loss < best_loss:
            # best_loss = dev_loss
        if tmf > best_tmf:
            best_tmf = tmf
            # 保存模型
            torch.save(scene_seg_model.state_dict(), os.path.join(cfg.save_path, "scene_seg_model.std"))
            torch.save(optimizer.state_dict(), os.path.join(cfg.save_path, "scene_seg_optim.std"))
            with open(os.path.join(cfg.save_path, "scene_seg_dev_result.json"), 'w', encoding='utf-8') as f:
                for dev_id, pss, tss, pso, tso in zip(dev_split, preds_span_skip, targets_span_skip, time_spans_pred, time_spans_truth):
                    f.write(dev_id + '\n')
                    f.write('Prediction: ' + json.dumps(pso) + '\n')
                    f.write('GroundTruth: ' + json.dumps(tso) + '\n\n')
            with open(os.path.join(cfg.save_path, "scene_seg_dev_result.txt"), "w", encoding='utf-8') as f:
                f.write("Dev loss: %s\n"
                        "Time Miou: %s\n"
                        "Time F1 Mean: %s\n"
                        "Time Miou x F1 Mean: %s\n"
                        "Time F1: %s\n"
                        "Time Miou x F1: %s\n" % (dev_loss, miou_ori, time_f1_mean, tmf_mean, time_f1, tmf))

        scheduler.step(dev_loss)
        if optimizer.state_dict()['param_groups'][0]['lr'] < last_lr:
            scene_seg_model.load_state_dict(torch.load(os.path.join(cfg.save_path, "scene_seg_model.std")))
            last_lr = optimizer.state_dict()['param_groups'][0]['lr']
            print("Current learning rate: ", last_lr)


def train_tagging(pretrain=False):
    train_split, dev_split = split_train_dev(cfg.train_data_dir, cfg.dev_size, cfg.seed)
    id2label, label2id = read_label_ids(data_path=cfg.labels_path)
    print(len(id2label.keys()))

    train_dataset = MultimodalDataset(cfg.train_data_dir, label2id, split=train_split, skip=cfg.frame_skip)
    dev_dataset = MultimodalDataset(cfg.train_data_dir, label2id, split=dev_split, skip=cfg.frame_skip)

    print("number of train samples: ", len(train_dataset))
    print("number of dev samples: ", len(dev_dataset))

    train_dataloader = DataLoaderX(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_dataloader = DataLoaderX(dev_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    tagging_model = TaggingMODEL[cfg.TaggingModelConfig["name"]](cfg.TaggingModelConfig).cuda()

    if not pretrain and os.path.exists(os.path.join(cfg.save_path, "tagging_model_pretrained.std")):
        print("Load pretrained model from ", os.path.join(cfg.save_path, "tagging_model_pretrained.std"))
        tagging_model.load_state_dict(torch.load(os.path.join(cfg.save_path, "tagging_model_pretrained.std")))

    loss_func = nn.BCEWithLogitsLoss()

    model_params = [p for n, p in list(tagging_model.named_parameters())]

    optimizer_other_parameters = [
        {'params': model_params, 'weight_decay': cfg.TaggingTrainConfig["weight_decay"], 'lr': cfg.TaggingTrainConfig["lr"]}
    ]

    optimizer = optim.Adam(optimizer_other_parameters)

    if not pretrain and os.path.exists(os.path.join(cfg.save_path, "tagging_model_pretrained.std")):
        total_steps = len(train_dataloader) * cfg.TaggingTrainConfig["epoch"]
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_steps,
                                                    num_training_steps=total_steps)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)

#     tagging_model.load_state_dict(torch.load(os.path.join(cfg.save_path, "tagging_model.std")))
#     optimizer.load_state_dict(torch.load(os.path.join(cfg.save_path, "tagging_optim.std")))
    
    last_lr = float('inf')
    best_loss = float('inf')
    for epoch in tqdm(range(cfg.TaggingTrainConfig["epoch"]), desc="Training Tagging model ..."):
        train_loss = 0.0
        train_size = 0

        tagging_model.train()
        optimizer.zero_grad()
        for batch in tqdm(train_dataloader):
            # print(batch)
            ids, video_frames_list, text_pair_list, _, _, segments_ori, segments_skip, labels, labels4pretrain = batch
            if pretrain:
                scene_ids, video_scenes_tensor, text_scenes_tensor, lens, scene_labels = \
                    prepare_for_tagging_pretrain(ids, video_frames_list, text_pair_list, labels4pretrain)
            else:
                scene_ids, video_scenes_tensor, text_scenes_tensor, lens, scene_labels = \
                    prepare_for_tagging(ids, video_frames_list, text_pair_list, segments_skip, labels)

            tagging_logits = tagging_model(video_scenes_tensor, lens)

            tagging_truth = convert_labels_to_seq(scene_labels)
            loss = loss_func(tagging_logits.cpu(), tagging_truth.cpu())
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            if not pretrain and os.path.exists(os.path.join(cfg.save_path, "tagging_model_pretrained.std")):
                scheduler.step()

            # print(loss.item())
            batch_size = len(ids)
            train_loss += loss.item() * batch_size
            train_size += batch_size

        train_loss /= train_size
        print("Train loss: ", train_loss)

        # dev
        preds, targets = [], []
        dev_loss = 0.0
        dev_size = 0
        tagging_model.eval()
        for batch in tqdm(dev_dataloader):
            ids, video_frames_list, text_pair_list, _, _, segments_ori, segments_skip, labels, labels4pretrain = batch
            if pretrain:
                scene_ids, video_scenes_tensor, text_scenes_tensor, lens, scene_labels = \
                    prepare_for_tagging_pretrain(ids, video_frames_list, text_pair_list, labels4pretrain)
            else:
                scene_ids, video_scenes_tensor, text_scenes_tensor, lens, scene_labels = \
                    prepare_for_tagging(ids, video_frames_list, text_pair_list, segments_skip, labels)
            tagging_logits = tagging_model(video_scenes_tensor, lens)

            tagging_truth = convert_labels_to_seq(scene_labels)
            loss = loss_func(tagging_logits.cpu(), tagging_truth.cpu())

            batch_size = len(ids)
            dev_loss += loss.item() * batch_size
            dev_size += batch_size

            preds.append(tagging_logits.cpu().detach().numpy())
            targets.append(tagging_truth.cpu().detach().numpy())

        dev_loss /= dev_size
        print("Dev loss: ", dev_loss)

        gap = calculate_gap(np.concatenate(preds, axis=0), np.concatenate(targets, axis=0))
        print("GAP: ", gap)

        if dev_loss < best_loss:
            best_loss = dev_loss
            if pretrain:
                torch.save(tagging_model.state_dict(), os.path.join(cfg.save_path, "tagging_model_pretrained.std"))
                torch.save(optimizer.state_dict(), os.path.join(cfg.save_path, "tagging_optim_pretrained.std"))
            else:
                torch.save(tagging_model.state_dict(), os.path.join(cfg.save_path, "tagging_model.std"))
                torch.save(optimizer.state_dict(), os.path.join(cfg.save_path, "tagging_optim.std"))

        if pretrain or not os.path.exists(os.path.join(cfg.save_path, "tagging_model_pretrained.std")):
            scheduler.step(dev_loss)
            if optimizer.state_dict()['param_groups'][0]['lr'] < last_lr:
                if pretrain:
                    tagging_model.load_state_dict(torch.load(os.path.join(cfg.save_path, "tagging_model_pretrained.std")))
                else:
                    tagging_model.load_state_dict(torch.load(os.path.join(cfg.save_path, "tagging_model.std")))
                last_lr = optimizer.state_dict()['param_groups'][0]['lr']
                print("Current learning rate: ", last_lr)


if __name__ == '__main__':
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    
    if not os.path.exists(cfg.save_path):
        os.mkdir(cfg.save_path)
    
    train_scene_seg()
    train_tagging(pretrain=False)
