import os
import sys
import glob
import json
import pickle
from tqdm import tqdm
import cv2
import torch
import random
import numpy as np

import config.config as cfg
from src.data.dataset import DataLoaderX, MultimodalDataset, collate_fn, prepare_for_scene_seg, prepare_for_tagging, \
    convert_seq_to_segments, convert_seq_to_labels, sort_seq_by_idx
from utils.data_utils import read_label_ids
from src.models import SceneSegMODEL, TaggingMODEL


def inference_scene_seg():
    id2label, label2id = read_label_ids(data_path=cfg.labels_path)
    print(len(id2label.keys()))
    dataset = MultimodalDataset(cfg.test_data_dir, label2id, skip=cfg.frame_skip)
    print("number of samples: ", len(dataset))
    dataloader = DataLoaderX(dataset, batch_size=1, collate_fn=collate_fn)
    scene_seg_model = SceneSegMODEL[cfg.SceneSegModelConfig["name"]](cfg.SceneSegModelConfig).cuda()
    scene_seg_model.load_state_dict(torch.load(os.path.join(cfg.save_path, "scene_seg_model.std")))

    scene_seg_model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # print(batch)
            ids, video_frames_list, text_pair_list, _, _, _, _, _, _ = batch  # segments, labels is None
            ids, video_frames_tensor, seq_lens, _, _ = prepare_for_scene_seg(ids, video_frames_list, train=False)
            scene_seg_logits = scene_seg_model(video_frames_tensor, seq_lens)
            #             _, scene_seq = torch.softmax(scene_seg_logits, dim=-1).max(-1)
            #             scene_seq = torch.softmax(scene_seg_logits, dim=-1)[:, :, 1] > 0.2
            scene_seq = torch.sigmoid(scene_seg_logits) > cfg.theta

            segments_pred = convert_seq_to_segments(scene_seq, seq_lens)  # 转换预测结果为segment pairs

            print(segments_pred)


def inference():
    id2label, label2id = read_label_ids(data_path=cfg.labels_path)
    print(len(id2label.keys()))
    dataset = MultimodalDataset(cfg.test_data_dir, label2id, skip=cfg.frame_skip)
    print("number of samples: ", len(dataset))
    dataloader = DataLoaderX(dataset, batch_size=1, collate_fn=collate_fn)
    scene_seg_model = SceneSegMODEL[cfg.SceneSegModelConfig["name"]](cfg.SceneSegModelConfig).cuda()
    scene_seg_model.load_state_dict(torch.load(os.path.join(cfg.save_path, "scene_seg_model.std")))
    tagging_model = TaggingMODEL[cfg.TaggingModelConfig["name"]](cfg.TaggingModelConfig).cuda()
    tagging_model.load_state_dict(torch.load(os.path.join(cfg.save_path, "tagging_model.std")))

    result = {}

    tagging_model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # print(batch)
            ids, video_frames_list, text_pair_list, _, _, _, _, _, _ = batch
            # 预测scene segments
            ids, video_frames_tensor, seq_lens, _, _ = prepare_for_scene_seg(ids, video_frames_list, train=False)
            scene_seg_logits = scene_seg_model(video_frames_tensor, seq_lens)
            scene_seq = torch.sigmoid(scene_seg_logits) > cfg.theta

            segments_pred = convert_seq_to_segments(scene_seq, seq_lens)
            # print(segments_pred)

            # 标签预测
            scene_ids, video_scenes_tensor, text_scenes_tensor, lens, _ = \
                prepare_for_tagging(ids, video_frames_list, text_pair_list, segments_pred)
            tagging_logits = tagging_model(video_scenes_tensor, lens)
            tagging_pred, tagging_scores = convert_seq_to_labels(tagging_logits, id2label)

            segments_pred = [segment for segs in segments_pred for segment in segs]

            for id, segment, label, score in zip(scene_ids, segments_pred, tagging_pred, tagging_scores):
                id = id + ".mp4"
                mp4 = os.path.join(cfg.test_video_dir, id)
                video_capture = cv2.VideoCapture()
                if not video_capture.open(mp4):
                    print(sys.stderr, 'Error: Cannot open video file ' + mp4)
                    return
                fps = video_capture.get(5)
                max_frame = int(video_capture.get(7))

                segment = [round(cfg.frame_skip * segment[0] / fps, 3), round(cfg.frame_skip * segment[1] / fps, 3)]
                segment[-1] = min(max_frame / fps, segment[-1])

                cur_result = {"segment": segment, "labels": label, "scores": [round(s, 2) for s in score]}
                result[id] = result.get(id, {})
                result[id]["result"] = result[id].get("result", [])
                result[id]["result"].append(cur_result)

    # print(result)
    # 保存结果
    with open(os.path.join(cfg.save_path, "results.json"), 'w', encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    inference()