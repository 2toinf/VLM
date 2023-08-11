import json
import torch
from tqdm import tqdm
from petrel_client.client import Client as CephClient
from torch.utils.data import Dataset
from PIL import Image 
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import io
import numpy as np
import random
import logging
from torchvision import transforms
logger = logging.getLogger('global')

class BridgeDataV1(Dataset):
    def __init__(self, 
                input_shape = 128, 
                root_dir = "bridgedata:s3://", 
                meta_file = "/mnt/lustre/zhengjinliang/vision-language-model/frame_predictor/BridgeDataV1.json",
                target_range = range(3, 4),
                with_timeline = False,
                **kwargs) -> None:
        super().__init__()
        self.input_shape = (input_shape, input_shape)
        self.root_dir = root_dir
        self.with_timeline = with_timeline
        self.target_range = target_range
        self.meta_file = meta_file
        with open(meta_file, "r") as f:
            self.datalist = json.load(f)

        self.img_list = []
        for idx, traj in enumerate(self.datalist):
            for i in range(traj['length']):
                self.img_list.append(
                    {
                        'frame_idx': i,
                        'traj_idx': idx
                    }
                )
        self.initialized = False
        self.transform = None
        self.build_transform()

    def _init_ceph(self):
        from petrel_client.client import Client as CephClient
        if not self.initialized:
            self.mclient = CephClient()
            self.initialized = True

    def __len__(self):
        return len(self.img_list)
    
    def get_single_img(self, path, idx):
        self._init_ceph()
        img_path = f'{self.root_dir}{path}_{idx}.jpg'
        value = self.mclient.Get(img_path)
        img_bytes = np.fromstring(value, np.uint8)
        buff = io.BytesIO(img_bytes)
        with Image.open(buff) as img:
            img = img.convert('RGB')

        if self.transform is not None:
                img = self.transform(img)
        return img

    def __getitem__(self, idx):
        try:
            traj_idx = self.img_list[idx]['traj_idx']
            frame_idx = self.img_list[idx]['frame_idx']
            target_idx = frame_idx + random.sample(self.target_range, 1)[0]

            # clip idx
            if target_idx > self.datalist[traj_idx]['length']:
                frame_idx = frame_idx - (self.datalist[traj_idx]['length'] - target_idx)
                target_idx = self.datalist[traj_idx]['length']

            text_instruction = self.datalist[traj_idx]['instruction'].replace('_', ' ') 
            if self.with_timeline:
                f', it is {frame_idx}th step'
            input_img = self.get_single_img(self.datalist[traj_idx]['path'],frame_idx)
            target_img = self.get_single_img(self.datalist[traj_idx]['path'], target_idx)
            return input_img, target_img, text_instruction
        except Exception as e:
            logger.info(f'Error when load {idx}')
            logger.info(e)
            return self.__getitem__(random.randint(0, len(self.img_list) - 1))

    
    def build_transform(self):
        t = []
        t.extend([
            transforms.Resize(self.input_shape, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ]
        )

        self.transform = transforms.Compose(t)





if __name__ == '__main__':
    # generate json files for BridgeData
    print("begin_read!!")
    path = "bridgedata:s3://mydata/bridgedata/raw/bridge_data_v1/"
    client = CephClient()
    files = [x[0] for x in client.get_file_iterator(path) if x[0][-4:] == '.jpg']
    print("read end!!")
    json_dict = {}
    for file in  tqdm(files):
        try:
            split = file.split('/')
            traj_path = '/'.join(split[:-1]) + '/' + split[-1].split('_')[0]
            img_num = int(split[-1].split('_')[-1][:-4])
            if traj_path in json_dict.keys():
                json_dict[traj_path]['length'] = max(img_num, json_dict[traj_path]['length'])

            else:
                json_dict[traj_path] = {
                    'path': traj_path,
                    'length': img_num,
                    'instruction': split[6]
                }
        except:
            print(f'Error when load {file}')

    print(len(json_dict.values()))
    with open("BridgeDataV1.json", "w") as f:
        json.dump(list(json_dict.values()), f, indent=4)


