import os
import argparse
import cv2
import sys
import glob
import tqdm
import torch
import numpy as np
from multiprocessing.dummy import Pool


def generate_axis_without_fuzzy_region(video_filepath, _left, _right, _top, _down):
    # 取帧
    capture = cv2.VideoCapture(video_filepath)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_number = 4
    samples = list()
    for i in range(sample_number):
        pos_frames = frame_count // (sample_number + 1) * (i + 1)
        capture.set(cv2.CAP_PROP_POS_FRAMES, pos_frames)
        ret, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = gray[_top:_down, _left:_right]
        samples.append(gray)
    height, width = gray.shape
    capture.release()
    # 边缘检测
    edges = [cv2.Canny(sample, 50, 200) for sample in samples]
    
    # 直线检测
    lines = list()
    for edge in edges:
        # line为一个列表 列表中的每个元素都是一条线
        line = cv2.HoughLinesP(edge, 1, np.pi / 180, 100, 10, 10)
        if line is None:
            continue
        for element in line:
            lines.append(element)
    
    # 分割线检测 垂直方向
    vertical_line_axis = list()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2:
            vertical_line_axis.append(x1)
    vertical_line_axis = [element for element in vertical_line_axis if 10 < element < width - 11]
    vertical_line_axis.sort()
    if len(vertical_line_axis) > 2:
        left = vertical_line_axis[0]
        right = vertical_line_axis[-1]
        left_count = vertical_line_axis.count(left) + vertical_line_axis.count(left + 1)
        right_count = vertical_line_axis.count(right) + vertical_line_axis.count(right - 1)

        if left_count <= sample_number + sample_number // 2:
            left = 0
        if right_count <= sample_number + sample_number // 2:
            right = width
    else:
        left = 0
        right = width
    
    # 分割线检测 水平方向
    horizen_line_axis = list()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if y1 == y2:
            horizen_line_axis.append(x1)
    horizen_line_axis = [element for element in horizen_line_axis if 10 < element < height - 11]
    horizen_line_axis.sort()
    if len(horizen_line_axis) > 2:
        top = horizen_line_axis[0]
        down = horizen_line_axis[-1]
        top_count = horizen_line_axis.count(top) + horizen_line_axis.count(top + 1)
        down_count = horizen_line_axis.count(down) + horizen_line_axis.count(down - 1)
        
        if top_count <= 2 * sample_number:
            top = 0
        if down_count <= 2 * sample_number:
            down = height
    else:
        top = 0
        down = height
    
    # 返回结果
    if left == 154:
        right = 565
    if left == 435:
        right = 839
    return _left + left, _left + right, _top + top, _top + down


def detect_same_region_single_direction(filename, direction):
    
    if direction != 'horizen' and direction != 'vertical':
        sys.exit(1)
    
    capture = cv2.VideoCapture(filename)
    total_frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 4
    frame_list = list()
    for i in range(1, frame_count + 1):
        capture.set(cv2.CAP_PROP_POS_FRAMES, total_frame * i // (frame_count + 1))
        ret, frame = capture.read()
        tensor = torch.from_numpy(frame).int()
        gray = (0.3 * tensor[:, :, 0] + 0.59 * tensor[:, :, 1] + 0.11 * tensor[:, :, 2]).int()
        if direction == 'vertical':
            gray = gray.T
        frame_list.append(gray)

    capture.release()

    differences = list()
    height, width = frame_list[0].shape
    for i in range(frame_count - 1):
        temp = abs((frame_list[i] - frame_list[i + 1]))
        difference = sum(temp <= 5) > 0.9 * height
        differences.append(difference)

        if sum(difference) > width / 3:
            continue
        else:
            break
    else:
        difference = sum(differences) == frame_count - 1
        difference = list(difference.numpy())
        return True, difference

    return False, [False]*width

def reverse(difference):
    times = 0
    places = list()
    
    for i in range(len(difference) - 1):
        if difference[i] != difference[i + 1]:
            places.append(i)
            times += 1
    
    return times, places

def display(difference):
    current = difference[0]
    count = 1
    data = list()
    for item in difference[1:]:
        if item == current:
            count += 1
        else:
            data.append((current, count))
            current = item
            count = 1
    data.append((current, count))

    return data

def calculate_different_region_axis(flag, difference):
    times, places = reverse(difference)
    length = len(difference)
    
    if flag is False or times > 20:
        return 0, length    
    
    data = display(difference)
    start_axis = length
    end_axis = 0
    current_axis = 0
    for flag, count in data:
        if flag is False and count > 30:
            start_axis = min(start_axis, current_axis)
            end_axis = max(end_axis, current_axis)
        current_axis += count
    if end_axis > start_axis:
        for i in range(start_axis, end_axis):
            difference[i] = False
    
    while times > 2:
        places.insert(0, -1)
        places.append(length-1)
        
        interval = [places[i+1] - places[i] for i in range(len(places) - 1)]
        
        min_interval = min(interval)
        min_index = interval.index(min_interval)
        
        left = places[min_index]
        right = places[min_index + 1]
        
        for place in range(left + 1, right + 1):
            difference[place] = difference[place] is False
        
        times, places = reverse(difference)
    
    if times == 0:
        return 0, length
    
    if times == 1:
        index = [difference.index(False), difference.index(True)]
        if max(index) - min(index) < 200:
            return 0, length
        return min(index), max(index)
    
    if times == 2:
        left = difference.index(False)
        right = difference.index(True, left) - 1
        if right - left < 200:
                return 0, length
        return left, right

def detect_same_region(input_path):
    global changed_video_save_dir, unchanged_video_save_dir, display_jpg_save_dir
    
    # 不变区域切除
    flag, horizen = detect_same_region_single_direction(input_path, direction='horizen')
    left, right = calculate_different_region_axis(flag, horizen)    
    
    flag, vertical = detect_same_region_single_direction(input_path, direction='vertical')
    top, down = calculate_different_region_axis(flag, vertical)
    
    capture = cv2.VideoCapture(input_path)
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    capture.set(cv2.CAP_PROP_FRAME_COUNT, int(frame_count))
    ret, frame = capture.read()
    height, width, channel = frame.shape
    capture.release()
    
    
    # 模糊区域切除
    left, right, top, down = generate_axis_without_fuzzy_region(input_path, left, right, top, down)
    
    # 替换不合理的视频
    if right - left < 200:
        left = 0
        right = height
    if down - top < 200:
        top = 0
        down = width

    # 保存视频
    if width == right - left and height == down - top:
        command = 'cp {} {} > /dev/null 2>&1'.format(input_path, unchanged_video_save_dir)
        os.system(command)
        return
    
    filename = input_path.split('/')[-1]
    output_path = os.path.join(changed_video_save_dir, filename)
    command = 'ffmpeg -i {} -vf crop={}:{}:{}:{} -c:a copy {} -loglevel quiet > /dev/null'.format(input_path, right-left, down-top, left, top, output_path)
    os.system(command)
    
    cv2.rectangle(frame, (left, top), (right, down), (0,0,255), 4)
    filename = input_path.split('/')[-1][:-3] + 'jpg'
    jpg_save_filepath = os.path.join(display_jpg_save_dir, filename)
    cv2.imwrite(jpg_save_filepath, frame)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-id", "--input_dir", type=str, required=True)
    parser.add_argument("-od", "--output_dir", type=str, required=True)
    parser.add_argument("-dd", "--display_dir", type=str, default='/home/tione/notebook/dataset/videos/display_jpg/train/')
    args = parser.parse_args()
    
#     input_video_dir = '../dataset/videos/test_5k_2nd'
#     changed_video_save_dir = '../dataset/videos/test_5k_2nd_without_same_region'
#     unchanged_video_save_dir = '../dataset/videos/test_5k_2nd_without_same_region'
#     display_jpg_save_dir = './test_5k_2nd/display_jpg/'
    input_video_dir = args.input_dir
    changed_video_save_dir = args.output_dir
    unchanged_video_save_dir = args.output_dir
    display_jpg_save_dir = args.display_dir

    if not os.path.exists(changed_video_save_dir):
        os.makedirs(changed_video_save_dir)
    if not os.path.exists(unchanged_video_save_dir):
        os.makedirs(unchanged_video_save_dir)
    if not os.path.exists(display_jpg_save_dir):
        os.makedirs(display_jpg_save_dir)
    
    filenames = glob.glob(os.path.join(input_video_dir, '*.mp4'))
    pool_size = 8
    pool = Pool(pool_size)
    
    try:
        for _ in tqdm.tqdm(pool.imap_unordered(detect_same_region, filenames), total=len(filenames)):
            pass
    finally:
        pool.close()
        pool.join()
