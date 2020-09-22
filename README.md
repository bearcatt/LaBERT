# Length-Controllable Image Captioning (ECCV2020)

This repo provides the implemetation of the paper [Length-Controllable Image Captioning](https://arxiv.org/abs/2007.09580).

## Install

```bash
conda create --name labert python=3.7
conda activate labert

conda install pytorch=1.3.1 torchvision cudatoolkit=10.1 -c pytorch
pip install h5py tqdm transformers==2.1.1
pip install git+https://github.com/salaniz/pycocoevalcap
```

#### Data & Pre-trained Models

* Prepare MSCOCO data follow [link](https://github.com/LuoweiZhou/VLP#-data-preparation).
* Download pretrained Bert and Faster-RCNN from [Baidu Cloud Disk](https://pan.baidu.com/s/14DRNGGOSMVfO9Vz5CCEdEg) [code: 0j9f]. 
  * It's an unified checkpoint file, containing a pretrained `Bert-base` and the `fc6` layer of the Faster-RCNN.
* Download our pretrained LaBERT model from [Baidu Cloud Disk](https://pan.baidu.com/s/12FujGSvDBQQROJOYrtDXsw) [code: fpke].


## Scripts
Train
```bash
python -m torch.distributed.launch \
  --nproc_per_node=$NUM_GPUS \
  --master_port=4396 train.py \
  save_dir $PATH_TO_TRAIN_OUTPUT \
  samples_per_gpu $NUM_SAMPLES_PER_GPU
```
Continue train
```bash
python -m torch.distributed.launch \
  --nproc_per_node=$NUM_GPUS \
  --master_port=4396 train.py \
  save_dir $PATH_TO_TRAIN_OUTPUT \
  samples_per_gpu $NUM_SAMPLES_PER_GPU \
  model_path $PATH_TO_MODEL
```
Inference
```bash
python inference.py \
  model_path $PATH_TO_MODEL \
  save_dir $PATH_TO_TEST_OUTPUT \
  samples_per_gpu $NUM_SAMPLES_PER_GPU
```
Evaluate
```bash
python evaluate.py \
  --gt_caption data/id2captions_test.json \
  --pd_caption $PATH_TO_TEST_OUTPUT/caption_results.json \
  --save_dir $PATH_TO_TEST_OUTPUT
```

## Cite
Please consider citing our paper in your publications if the project helps your research. 
```
@article{deng2020length,
  title={Length-Controllable Image Captioning},
  author={Deng, Chaorui and Ding, Ning and Tan, Mingkui and Wu, Qi},
  journal={arXiv preprint arXiv:2007.09580},
  year={2020}
}
```
