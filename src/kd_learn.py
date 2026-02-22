import argparse
from typing import Dict
import os
import torch
from torch import optim
import pickle

torch.cuda.empty_cache()
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from datasets import Dataset
from models import ComplExMDR
from regularizers import F2, N3
from kd import KD, KD_RL, KD_SingleTeacher, \
    KD_BestConf, KD_BestTeacher, KD_BestStrategy, KD_RandTeacher, KD_AvgTeacher
from kd_optimizers import KDOptimizer
from datetime import datetime

import json
import numpy as np
import time
import ast
# import optuna
from config import device

# os.environ["CUDA_VISIBLE_DEVICES"] = device
seed = 2022
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

big_datasets = ['DB15K', 'YAGO15K',
                'WN9', 'WN18',
                'FB15K-237', 'MMKB-DB15K', 'MKG-Y', 'MKG-W']
datasets = big_datasets

parser = argparse.ArgumentParser(
    description="Relational learning contraption"
)

parser.add_argument(
    '--dataset', choices=datasets,
    default='WN18RR',
    help="Dataset in {}".format(datasets)
)

models = ['ComplExMDR']
parser.add_argument(
    '--model', choices=models,
    default='ComplExMDR',
    help="Model in {}".format(models)
)

parser.add_argument(
    '--alpha', default=1, type=float,
    help="Modality/local attention embedding ratio in modality_structure fusion."
)
parser.add_argument(
    '--img_info', default='../data/FB15K-237/img_vec.pickle'
)
parser.add_argument(
    '--dscp_info', default='../data/FB15K-237/dscp_vec.pickle'
)
parser.add_argument(
    '--modality_split', default=True, type=ast.literal_eval,
    help="Whether split modalities."
)

parser.add_argument(
    '--fusion_img', default=True, type=ast.literal_eval,
    help="Whether fusion img modality graph."
)

parser.add_argument(
    '--fusion_dscp', default=True, type=ast.literal_eval,
    help="Whether fusion description modality graph."
)

parser.add_argument(
    '--scale', default=20, type=float,
    help="temp parameter in contrastive loss"
)

parser.add_argument(
    '--kd_temp', default=4.0, type=float,
    help="temp parameter in KD"
)

parser.add_argument(
    '--kd_ce_weight', default=1.0, type=float,
    help="ce weight in KD"
)

parser.add_argument(
    '--kd_kl_weight', default=1.0, type=float,
    help="kl weight in KD"
)

parser.add_argument(
    '--kd_loss_type', default='kd', type=str,
    help="kd loss type in KD"
)  # kd, dkd


regularizers = ['N3', 'F2']
parser.add_argument(
    '--regularizer', choices=regularizers, default='N3',
    help="Regularizer in {}".format(regularizers)
)

optimizers = ['Adagrad', 'Adam', 'SGD']
parser.add_argument(
    '--optimizer', choices=optimizers, default='Adagrad',
    help="Optimizer in {}".format(optimizers)
)

parser.add_argument(
    '--max_epochs', default=200, type=int,
    help="Number of epochs."
)
parser.add_argument(
    '--valid', default=3, type=float,
    help="Number of epochs before valid."
)
parser.add_argument(
    '--rank', default=2000, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--batch_size', default=1000, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--reg', default=0.01, type=float,
    help="Regularization weight"
)
parser.add_argument(
    '--init', default=1e-3, type=float,
    help="Initial scale"
)
parser.add_argument(
    '--learning_rate', default=1e-1, type=float,
    help="Learning rate"
)
parser.add_argument(
    '--decay1', default=0.9, type=float,
    help="decay rate for the first moment estimate in Adam"
)
parser.add_argument(
    '--decay2', default=0.999, type=float,
    help="decay rate for second moment estimate in Adam"
)
parser.add_argument(
    '--note', default=None,
    help="model setting or ablation note for ckpt save"
)
parser.add_argument(
    '--early_stopping', default=10, type=int,
    help="stop training until MRR stop increasing after early stopping epoches"
)
parser.add_argument(
    '--ckpt_dir', default='./kdckpt/wn9/test/'
)
parser.add_argument(
    '--rl_hs', default=1024, type=int,
    help="RL MLP hidden state"
)
parser.add_argument(
    '--distiller', default=None, help="Distiller model in KD, KD_BestConf, KD_BestTeacher, KD_BestStrategy, KD_RandTeacher, KD_AvgTeacher")
parser.add_argument(
    '--rl_reward', default=1., type=float,
    help="RL reward"
)
parser.add_argument(
    '--rl_penalize', default=10., type=float,
    help="RL penalize"
)
parser.add_argument(
    '--rl_weight', default=1., type=float,
    help="RL weight"
)
parser.add_argument(
    '--policy_temp', default=1., type=float,
    help="policy MLP temperature"
)
parser.add_argument(
    '--dkd_alpha', default=1., type=float,
    help="decoupled KD alpha"
)
parser.add_argument(
    '--dkd_beta', default=1., type=float,
    help="decoupled KD beta"
)
parser.add_argument(
    '--modal_missing', default=0.0, type=float,
    help="modal fraction"
)


args = parser.parse_args()
print("running setting args: ", args)

dataset = Dataset(args.dataset)
examples = torch.from_numpy(dataset.get_train().astype('int64'))

print(dataset.get_shape())

teacher_model_path = {'FB15K-237': 'YOUR_PATH_TO_TEACHER_MODEL_ComplExMDR'}[args.dataset]


print(teacher_model_path)
teacher = torch.load(teacher_model_path)
student = ComplExMDR(dataset.get_shape(), args.rank, args.init, False, False, False,
                     args.alpha,
                     args.dataset, args.scale, args.img_info, args.dscp_info, args) # single modal student model

if args.kd_loss_type == 'kdrl':
    distiller = KD_RL(student, teacher, dataset.n_entities, args.rl_hs,
                      reward=args.rl_reward,
                      penalize=args.rl_penalize,
                      policy_temp=args.policy_temp,
                      args=args)
elif args.distiller != None:
    distiller = {'best_teacher': lambda: KD_BestTeacher(student, teacher),
                 'best_conf': lambda: KD_BestConf(student, teacher),
                 'best_strategy': lambda: KD_BestStrategy(student, teacher),
                 'rand_teacher': lambda: KD_RandTeacher(student, teacher),
                 'avg_teacher': lambda: KD_AvgTeacher(student, teacher),
                 'single_teacher': lambda: KD_SingleTeacher(student, teacher, args),
                 'soft_kl': lambda: KD_SoftKL(student, teacher)}[args.distiller]()
else:
    distiller = KD(student, teacher, assistant)

device = 'cuda'
distiller.to(device)
# teacher.to(device)
student.to(device)

regularizer = {
    'F2': F2(args.reg),
    'N3': N3(args.reg),
}[args.regularizer]

optim_method = {
    'Adagrad': lambda: optim.Adagrad(distiller.get_learnable_parameters(), lr=args.learning_rate),
    'Adam': lambda: optim.Adam(distiller.get_learnable_parameters(), lr=args.learning_rate,
                               betas=(args.decay1, args.decay2)),
    'SGD': lambda: optim.SGD(distiller.get_learnable_parameters(), lr=args.learning_rate)
}[args.optimizer]()

optimizer = KDOptimizer(distiller, regularizer, optim_method, args.batch_size,
                        args.kd_ce_weight, args.kd_kl_weight, args.kd_temp, args.kd_loss_type,
                        args=args, to_skip=dataset.train_to_skip)

scheduler = ReduceLROnPlateau(optim_method, 'min', factor=0.5, verbose=True, patience=10, threshold=1e-3)


def avg_both(mrs: Dict[str, float], mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
    """
    aggregate metrics for missing lhs and rhs
    :param mrrs: d
    :param hits:
    :return:
    """
    mr = (mrs['lhs'] + mrs['rhs']) / 2
    mrr = (mrrs['lhs'] + mrrs['rhs']) / 2.
    h = (hits['lhs'] + hits['rhs']) / 2.
    return {'MR': mr, 'MRR': mrr, 'hits@[1,3,10]': h.tolist()}


def create_subfolder(log_dir):
    index = len(os.listdir(log_dir))
    try:
        new_folder_path = os.path.join(log_dir, str(index))
        os.makedirs(new_folder_path)
    except Exception as e:
        print(e)
        print(new_folder_path)
    return new_folder_path


ckpt_dir = args.ckpt_dir
if not ckpt_dir.endswith('/'):
    ckpt_dir = ckpt_dir + '/'
run_dir = create_subfolder(ckpt_dir)

cur_loss = 0
best_loss = 10000
curve = {'train': [], 'valid': [], 'test': []}
curve_loss = []
best_mr = 100000
best_mrr = 0
best_hits = [0, 0, 0]
best_epoch = 0
best_hits10 = 0
best_val_model_test_result = {}
train, test, valid = [0, 0, 0]
model_path = run_dir + '/distiller-' + datetime.now().strftime("%Y%m%d_%H%M") + '-n-' + str(args.note) + '.pth'
student_path = run_dir + '/student-' + datetime.now().strftime("%Y%m%d_%H%M") + '-n-' + str(args.note) + '.pth'

since = time.time()
print(student_path)

for e in range(args.max_epochs):
    cur_loss = optimizer.epoch(examples, e).tolist()

    curve_loss.append(cur_loss)
    if cur_loss < best_loss:
        best_loss = cur_loss
    if (e + 1) % args.valid == 0:
        valid, test, train = [
            avg_both(*dataset.eval(student, split, -1 if split != 'train' else 3000))
            for split in ['valid', 'test', 'train']
        ]

        curve['valid'].append(valid)
        curve['test'].append(test)
        curve['train'].append(train)

        print("epoch: %d" % (e + 1))
        print(args.note)
        print("\t TRAIN: ", train)
        print("\t TEST : ", test)
        print("%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t" % (test['MR'],
                                                  test['MRR'] * 100.0,
                                                  test['hits@[1,3,10]'][0] * 100.0,
                                                  test['hits@[1,3,10]'][1] * 100.0,
                                                  test['hits@[1,3,10]'][2] * 100.0))
        print("\t VALID : ", valid)

        # if valid['MR'] < best_mr:

        if valid['MRR'] > best_mrr:
            best_hits10 = valid['hits@[1,3,10]'][-1]
            best_mrr = valid['MRR']
            best_mr = valid['MR']
            best_hits = valid['hits@[1,3,10]']
            best_epoch = e + 1
            best_val_model_test_result = test
            torch.save(distiller, model_path)
            torch.save(student, student_path)

        scheduler.step(valid['hits@[1,3,10]'][2])
        print("Learning rate at epoch {}: {}".format(e + 1, scheduler._last_lr))
    if (e + 1 - best_epoch) > args.early_stopping:
        print(optimizer.reward_logs)
        break

time_elapsed = time.time() - since
sec_per_epoch = time_elapsed / float(e + 1)
print('Time consuming: {:.3f}s, average sec per epoch: {:.3f}s'.format(time_elapsed, sec_per_epoch))
print('last_lr: ', scheduler._last_lr)

print('Test result on best Valid MRR model: ', best_val_model_test_result)

print("%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t" % (best_val_model_test_result['MR'],
                                          best_val_model_test_result['MRR'] * 100.0,
                                          best_val_model_test_result['hits@[1,3,10]'][0] * 100.0,
                                          best_val_model_test_result['hits@[1,3,10]'][1] * 100.0,
                                          best_val_model_test_result['hits@[1,3,10]'][2] * 100.0))

result = {'epoch': e + 1,
          'best_loss': best_loss, 'best_epoch': best_epoch,
          'best_mrr': best_mrr, 'best_mr': best_mr, 'best_hits10': best_hits,
          'curve': curve, 'curve_loss': curve_loss,
          'train': train, 'test': test, 'val': valid, 'final_result': best_val_model_test_result,
          'sec_per_epoch': sec_per_epoch, 'last_lr': scheduler._last_lr, 'run_dir': run_dir}
result.update(args.__dict__)

print(result)

if run_dir:
    print(run_dir)
    with open(os.path.join(run_dir,
                           'result.json'),
              'w') as f:
        json.dump(result, f, indent=2)
