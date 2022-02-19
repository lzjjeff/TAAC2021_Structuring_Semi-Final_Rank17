import os
import json
import argparse


def merge_to_pre (data, interval):
    new_json = {}
    for mp4_name, result_dict in data.items():
        # print(mp4_name, result_dict)
        for res, seg_list in result_dict.items():
            # print(res, seg_list)
            new_seg_list = [seg_list[len(seg_list)-1]]
            for i in range(len(seg_list)-2, 0, -1):
                # print(seg_list[i])
                item = seg_list[i]
                diff = round(item['segment'][1], 3)-round(item['segment'][0], 3)
                if diff > interval:
                    # print(diff)
                    new_seg_list.insert(0, item)
                else:
                    seg_list[i-1]['segment'][1] += diff
                    # for label_i in range(0, len(seg_list[i-1]['labels'])):
                    #     for label_j in range(0, len(item['labels'])):
                    #         if item['labels'][label_j] == seg_list[i-1]['labels'][label_i]:
                    #             if item['scores'][label_j] >
            if len(seg_list) > 1:
                new_seg_list.insert(0, seg_list[0])
            print(len(seg_list), seg_list)
            print(len(new_seg_list), new_seg_list)
        new_result_dict = {"result": new_seg_list}
        new_json[mp4_name] = new_result_dict
    # print(new_json)
    print("merged to pre")

    return new_json


# def merge_to_app (data):
#     new_json = {}
#     for mp4_name, result_dict in data.items():
#         # print(mp4_name, result_dict)
#         for res, seg_list in result_dict.items():
#             # print(res, seg_list)
#             new_seg_list = []
#             for i in range(0, len(seg_list)-1):
#                 # print(seg_list[i])
#                 item = seg_list[i]
#                 diff = round(item['segment'][1], 3)-round(item['segment'][0], 3)
#                 if diff > 0.5:
#                     # print(diff)
#                     new_seg_list.append(item)
#                 else:
#                     seg_list[i+1]['segment'][0] -= diff
#                     # for label_i in range(0, len(seg_list[i-1]['labels'])):
#                     #     for label_j in range(0, len(item['labels'])):
#                     #         if item['labels'][label_j] == seg_list[i-1]['labels'][label_i]:
#                     #             if item['scores'][label_j] >
#             new_seg_list.append(seg_list[len(seg_list)-1])
#             print(len(seg_list), seg_list)
#             print(len(new_seg_list), new_seg_list)
#         new_result_dict = {"result": new_seg_list}
#         new_json[mp4_name] = new_result_dict
#     # print(new_json)
#     print("merged to append")
#
#     with open('result_to_app.json', 'w', encoding='utf8') as json_f:
#         json.dump(new_json, json_f, ensure_ascii=False, indent=4)
#         # json.dump(new_json, json_f, ensure_ascii=False)
#     print("json_file is written")
#
#     return 21


def merge_to_pre_label (data, interval):
    new_json = {}
    for mp4_name, result_dict in data.items():
        # print(mp4_name, result_dict)
        for res, seg_list in result_dict.items():
            # print(res, seg_list)
            new_seg_list = [seg_list[len(seg_list)-1]]
            for i in range(len(seg_list)-2, 0, -1):
                # print(seg_list[i])
                item = seg_list[i]
                diff = round(item['segment'][1], 3)-round(item['segment'][0], 3)
                if diff > interval:
                    # print(diff)
                    new_seg_list.insert(0, item)
                else:
                    seg_list[i-1]['segment'][1] += diff
                    for label_j in range(0, len(item['labels'])):
                        dup = 0 # duplicate flag
                        for label_i in range(0, len(seg_list[i-1]['labels'])):
                            if item['labels'][label_j] == seg_list[i-1]['labels'][label_i]:
                                dup = 1
                                # print("Duplicate label", item['labels'][label_j], seg_list[i-1]['labels'][label_i])
                        if dup == 0 and len(seg_list[i-1]['labels']) < 20:
                            seg_list[i-1]['labels'].append(item['labels'][label_j])
                            seg_list[i-1]['scores'].append(item['scores'][label_j])
                            print("new label", item['labels'][label_j], "score", item['scores'][label_j])
                            # create a tuple to sort
                            tuple_list = []
                            for sort_idx in range(0, len(seg_list[i-1]['labels'])):
                                tuple_list.append((seg_list[i-1]['labels'][sort_idx], seg_list[i-1]['scores'][sort_idx]))
                            # print(tuple_list)
                            Sort_Tuple(tuple_list)  # sort tuple code
                            # print("Sort_Tuple", tuple_list)
                            temp_label = []
                            temp_score = []
                            for tup_idx in range(0, len(tuple_list)):
                                temp_label.append(tuple_list[tup_idx][0])
                                temp_score.append(tuple_list[tup_idx][1])
                            seg_list[i-1]['labels'] = temp_label
                            seg_list[i-1]['scores'] = temp_score
            if len(seg_list) > 1:
                new_seg_list.insert(0, seg_list[0])

            # print(len(seg_list), seg_list)
            # print(len(new_seg_list), new_seg_list)
        new_result_dict = {"result": new_seg_list}
        new_json[mp4_name] = new_result_dict
    # print(new_json)
    print("merged to pre with labels")

    # with open('result_to_pre.json', 'w', encoding='utf8') as json_f:
    #     json.dump(new_json, json_f, ensure_ascii=False, indent=4)
    #     # json.dump(new_json, json_f, ensure_ascii=False)
    # print("json_file is written")

    return new_json


def Sort_Tuple(tup_list):
    # reverse = None (Sorts in Ascending order)
    # key is set to sort using second element of
    # sublist lambda has been used
    tup_list.sort(key = lambda x: x[1], reverse=True)
    return tup_list

# printing the sorted list of tuples


def clean_labels(data):

    for mp4_name, result_dict in data.items():
        # print(mp4_name, result_dict)
        for res, seg_list in result_dict.items():
            # print(res, seg_list)

            # clean "推广页” in segments except for the last segment
            for i in range(0, len(seg_list)-1):
                # print(seg_list[i])
                item = seg_list[i]
                item_labels = item['labels']
                item_scores = item['scores']
                new_item_labels = []
                new_item_scores = []
                if len(item_labels)>1:
                    for label_idx in range(0, len(item_labels)):
                        if item_labels[label_idx] == '推广页':
                            print(item)
                        else:
                            new_item_labels.append(item_labels[label_idx])
                            new_item_scores.append(item_scores[label_idx])
                    item['labels'] = new_item_labels
                    item['scores'] = new_item_scores
                    print(item['labels'], item['scores'])

    # print(new_json)
    print("clean")

    return data


# def clean_labels2(data):
#     # 最后一个segment只留“推广页”
#     for mp4_name, result_dict in data.items():
#         # print(mp4_name, result_dict)
#         for res, seg_list in result_dict.items():
#             # print(res, seg_list)
#             idx = len(seg_list)-1
#             seg_list[idx]['labels'] = ["推广页"]
#             seg_list[idx]['scores'] = [1.0]
#     print("clean other labels in the last segment")
#     return data


def json_dump(data, filename):
    with open(filename, 'w', encoding='utf8') as json_f:
        json.dump(data, json_f, ensure_ascii=False, indent=4)
    print("json_file is written")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-fp", "--file_path", type=str, required=True)
    parser.add_argument("-fn", "--file_name", type=str, default='results.json')
    args = parser.parse_args()
    
    filepath = os.path.join(args.file_path, args.file_name)
    f = open(filepath, "r", encoding="utf-8")
    data = json.load(f)
    print(type(data))
    data = clean_labels(data)
    # data = merge_to_pre(data, 0.6)
    data = merge_to_pre_label(data, 0.6)
    data = clean_labels(data)
    json_dump(data, os.path.join(args.file_path, 'results_post.json'))

    # f = open('results-15_merge_pre_label_0-6.json', "r", encoding="utf-8")
    # data = json.load(f)
    # #
    # cnt = 0
    # for k,val in data.items():
    #     for res, seg_list in val.items():
    #         for i in range(0, len(seg_list)-1):
    #             if len(seg_list[i]['labels']) != len(seg_list[i]['scores']):
    #                 print(len(seg_list[i]['labels']), len(seg_list[i]['scores']))
    #             if len(seg_list[i]['labels']) > 20:
    #                 print((seg_list[i]['labels']))
    #             for lab_id in range(0, len(seg_list[i]['labels'])):
    #                 if seg_list[i]['labels'][lab_id] == '推广页':
    #                     print(k, i, len(seg_list), seg_list[i]['labels'][lab_id], len(seg_list[i]['labels']))
    #
    #             if seg_list[i]['segment'][1] != seg_list[i+1]['segment'][0]:
    #                 cnt+=1
    #                 print(cnt, k, i, seg_list[i]['segment'], seg_list[i+1]['segment'])
            # if seg_list[len(seg_list)-2]['segment'][1] != seg_list[len(seg_list)-1]['segment'][0]:
            #     print(k, i, seg_list[i]['segment'], seg_list[i + 1]['segment'])



