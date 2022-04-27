import json
import os

def json_ext(path, flag):
    if flag is 'train':
        dataset_path = os.path.join(path, 'train', 'mix')
        json_path = os.path.join(path, 'Json', 'train')
        os.makedirs(json_path, exist_ok=True)
    else:
        dataset_path = os.path.join(path, 'dev', 'mix')
        json_path = os.path.join(path, 'Json', 'dev')
        os.makedirs(json_path, exist_ok=True)
    data_dir = os.listdir(dataset_path)
    data_dir.sort()
    data_num = len(data_dir)
    data_list = []

    for i in range(data_num):
        file_name = data_dir[i]
        file_name = os.path.splitext(file_name)[0]
        data_list.append(file_name)

    with open(os.path.join(json_path, 'files.json'),'w') as f :
        json.dump(data_list, f, indent=4)


path = '/media/Dataset/WSJ-SI-84-Trans'
json_ext(path, flag='train')
json_ext(path, flag='dev')