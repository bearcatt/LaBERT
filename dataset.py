import json
import os.path as osp
from collections import namedtuple

import h5py
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from utils.tokenizer import EOS, MASK, PAD, tokenizer

Sample = namedtuple("Sample", ["caption", "image_id"])


class COCOCaptionDataset(Dataset):

    def __init__(self, root, split, boundaries):
        self.split = split
        self.root = root
        self.boundaries = boundaries

        if self.split == 'test':
            self.load_fn = self._get_item_infer
            self.build_infer_samples()
        else:
            self.load_fn = self._get_item_train
            self.build_train_samples()

    def build_infer_samples(self):
        with open(osp.join(self.root, 'dataset_coco.json')) as f:
            captions = json.load(f)
            captions = captions['images']

        file_id2captions_test = osp.join(self.root, 'id2captions_test.json')
        file_test_samples = osp.join(self.root, 'test_samples.json')
        if not osp.exists(file_id2captions_test):
            samples = list()
            id2captions = dict()
            for item in captions:
                if item['split'] in self.split:
                    image_id = item['filename'].split('.')[0]
                    samples.append(image_id)
                    image_id = str(int(image_id[-12:]))
                    id2captions[image_id] = list()
                    for c in item['sentences']:
                        caption = ' '.join(c['tokens']) + '.'
                        id2captions[image_id].append({'caption': caption})

            with open(file_id2captions_test, 'w') as f:
                json.dump(id2captions, f)
            with open(file_test_samples, 'w') as f:
                json.dump({'ids': samples}, f)
        else:
            with open(file_test_samples) as f:
                samples = json.load(f)['ids']

        self.samples = samples

    def build_train_samples(self):
        with open(osp.join(self.root, 'dataset_coco.json')) as f:
            captions = json.load(f)
            captions = captions['images']

        file_train_data = osp.join(self.root, f'{self.split}_data.pth')
        if not osp.exists(file_train_data):
            samples = list()
            for item in captions:
                if item['split'] in self.split:
                    image_id = item['filename'].split('.')[0]
                    for c in item['sentences']:
                        caption = ' '.join(c['tokens']) + '.'
                        caption = tokenizer.encode(caption)
                        if len(caption) > self.boundaries[-1][-1]:
                            continue
                        sample = Sample(caption=caption, image_id=image_id)
                        samples.append(sample)
            torch.save(samples, file_train_data)
        else:
            samples = torch.load(file_train_data)

        self.samples = samples

    def get_region_feature(self, name):
        with h5py.File(osp.join(self.root, f'feat{name[-3:]}.h5'), 'r') as features, \
                h5py.File(osp.join(self.root, f'cls{name[-3:]}.h5'), 'r') as classes, \
                h5py.File(osp.join(self.root, f'region_bbox.h5'), 'r') as bboxes:
            region_feature = torch.from_numpy(features[name][:])
            region_class = torch.from_numpy(classes[name][:])
            region_spatial = torch.from_numpy(bboxes[name][:])
        return region_feature, region_class, region_spatial

    def __getitem__(self, index):
        return self.load_fn(index)

    def _get_item_train(self, index):
        sample = self.samples[index]
        input_token_id = sample.caption

        length = len(input_token_id)
        for i, (l, h) in enumerate(self.boundaries, 1):
            if l <= length <= h:
                length_level = i
                break

        high = self.boundaries[length_level - 1][1]
        offset = high - length

        token_type_id = [length_level] * high
        token_type_id = torch.tensor(token_type_id, dtype=torch.long)

        input_token_id = input_token_id + [EOS] * offset

        # masked_token_id = input_token_id.copy()
        # num_masks = random.randint(max(1, int(0.1 * high)), high)
        # selected_idx = random.sample(range(high), num_masks)
        # for i in selected_idx:
        #     masked_token_id[i] = MASK

        input_token_id = torch.tensor(input_token_id, dtype=torch.long)
        # masked_token_id = torch.tensor(masked_token_id, dtype=torch.long)

        region_feature, region_class, region_spatial = \
            self.get_region_feature(sample.image_id)

        return token_type_id, input_token_id, region_feature, \
            region_class, region_spatial

    def _get_item_infer(self, index):
        sample = self.samples[index]
        region_feature, region_class, region_spatial = \
            self.get_region_feature(sample)
        image_id = torch.tensor(int(sample[-12:]), dtype=torch.long)
        return region_feature, region_class, region_spatial, image_id

    def __len__(self):
        return len(self.samples)


def collate_fn_train(batch):
    batch = list(zip(*batch))

    token_type_id = pad_sequence(batch[0], batch_first=True, padding_value=0)
    input_token_id = pad_sequence(batch[1], batch_first=True, padding_value=PAD)

    region_feature = torch.stack(batch[2], dim=0)
    region_class = torch.stack(batch[3], dim=0)
    region_spatial = torch.stack(batch[4], dim=0)

    return token_type_id, input_token_id, region_feature, \
        region_class, region_spatial
