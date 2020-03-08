# Length-Controllable Image Captioning

## Install

```bash
conda create --name labert python=3.7
conda activate labert

conda install pytorch=1.3.1 torchvision cudatoolkit=10.1 -c pytorch
conda install h5py tqdm transformers=2.1.1
pip install git+https://github.com/salaniz/pycocoevalcap
```

## Scripts
Train
```bash
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=4396 train.py save_dir $PATH_TO_TRAIN_OUTPUT samples_per_gpu $NUM_SAMPLES_PER_GPU
```
Continue train
```bash
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=4396 train.py save_dir $PATH_TO_TRAIN_OUTPUT samples_per_gpu $NUM_SAMPLES_PER_GPU model_path $PATH_TO_MODEL
```
Inference
```bash
python inference.py model_path $PATH_TO_MODEL save_dir $PATH_TO_TEST_OUTPUT samples_per_gpu $NUM_SAMPLES_PER_GPU
```
Evaluate
```bash
python evaluate.py --gt_caption data/id2captions_test.json --pd_caption $PATH_TO_TESR_OUTPUT/caption_results.json --save_dir $PATH_TO_TESR_OUTPUT
```

## Notes
Download pretrained Bert model from [link]()

Download pretrained LaBERT model from [link]()