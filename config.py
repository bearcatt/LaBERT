from yacs.config import CfgNode as CN

_C = CN()
_C.device = 'cuda'
_C.distributed = True
_C.log_time = 20
_C.checkpoint_time = 2500
_C.save_dir = 'output'
_C.data_dir = 'data'
_C.num_workers = 3
_C.boundaries = ((7, 9), (10, 14), (15, 19), (20, 25))
_C.samples_per_gpu = 36  # 64 for inference
_C.model_path = ''
_C.pretrained_bert = 'pretrained/bert.pth'

_C.solver = CN()
_C.solver.lr = 5e-5
_C.solver.weight_decay = 1e-2
_C.solver.betas = (0.9, 0.999)
_C.solver.grad_clip = 1.0

_C.scheduler = CN()
_C.scheduler.warmup_steps = 1000
_C.scheduler.max_steps = 100000

_C.loss = CN()
_C.loss.balance_weight = 0.5
_C.loss.label_smoothing = 0.1

_C.infer = CN()
_C.infer.steps = (10, 15, 20, 20)
_C.infer.eos_decay = (1.0, 0.88, 0.95, 1.0)
