

seed = 123456
batch_size = 4
frame_skip =12       # 每几帧抽一
# frame_start = 0
max_scene_len = 100   # 最大的scene长度
max_text_len = 100
dev_size = 0.1        # 验证集大小
theta = 0.2

save_path = "./save/"    # 改
labels_path = '/home/tione/notebook/dataset/label_id.txt'

train_video_dir = "/home/tione/notebook/dataset/videos/train_without_same_region/"   # 训练集的视频文件路径
train_text_dir = "./pre/clean_ocr_train.json"   # 忽略
# train_shot_dir = '/home/tione/notebook/dataset/structuring/structuring_dataset_train_5k/shot_txt/'    # 训练集的shots文件路径
train_data_dir = "/home/tione/notebook/dataset/processed/train_cut/"      # 训练集的特征文件路径,特征提取的保存路径
train_truth_path = "/home/tione/notebook/dataset/structuring/GroundTruth/"   # 训练集的ground truth文件路径
train_pretrain_truth_path = "/home/tione/notebook/algo-2021/dataset/tagging/GroundTruth/datafile/"

test_video_dir = "/home/tione/notebook/dataset/videos/test_2nd_without_same_region/"     # 测试集的视频文件路径
test_text_dir = "./pre/clean_ocr_test.json"         # 忽略
# test_shot_dir = '/home/tione/notebook/dataset/structuring/structuring_dataset_test_5k/shot_txt/'             # 测试集的shots文件路径
test_data_dir = "/home/tione/notebook/dataset/processed/test_2nd_cut/"        # 测试集的特征文件路径,特征提取的保存路径

SceneSegTrainConfig = {
    "epoch": 20,
    "lr": 1e-4,
    "weight_decay": 5e-4,
    "loss_func": "bce",
}

TaggingTrainConfig = {
    "epoch": 30,
    "lr": 1e-4,
    "lr_bert": 1e-5,
    "weight_decay": 5e-4,
}

SceneSegModelConfig = {
    "name": "cnn",
    "video_model_config":{
        "in_size": 768,
        "face_size": 512,
        "h_size": 512,
        "dropout": 0.2,
        "num_clusters": 64,
        "lamb":2,
        "groups":8,
        "win_size": frame_skip,
        "max_len":max_scene_len,
    },
    "h_size": 1024,
    "cos_channel": 128, 
    "head_num": 8,
    "dropout": 0.1,
    'class_num': 1,
}       # SceneSeg模型参数

TaggingModelConfig = {
    "name": "win",
    "video_model_config":{
        "in_size": 768,
        "h_size": 512,
        "dropout": 0.2,
        "num_clusters": 128,
        "lamb":2,
        "groups":16,
        "win_size": frame_skip,
        "max_len":max_scene_len,
    },
    "text_model_config":{
        "in_size": 768,
        "h_size": 512,
        "dropout": 0.2,
    },
    "h_size": 512,
    "dropout": 0.2,
    "gate_reduce": 8,
    "max_frames":max_scene_len,
}       # Tagging模型参数
