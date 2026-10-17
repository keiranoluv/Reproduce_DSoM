import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd


def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss


def multi_dkd_loss(
    logits_student, logits_teacher, target, alpha, beta, temperature, to_skip=None
):
    # gt_mask = _get_gt_mask(logits_student, target)
    # other_mask = _get_other_mask(logits_student, target)
    gt_mask = _get_gt_mask_multi(logits_student, target, to_skip)
    other_mask = _get_other_mask_multi(logits_student, target, to_skip)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_gt_mask_multi(logits, target, to_skip):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    for i in range(len(to_skip)):
        mask[i, to_skip[i]] = True
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def _get_other_mask_multi(logits, target, to_skip=None):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    for i in range(len(to_skip)):
        mask[i, to_skip[i]] = False
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


import tqdm
from torch import optim

from kd import Distiller
from models import KBCModel
from regularizers import Regularizer


class KDOptimizer(object):
    def __init__(
        self,
        model: Distiller,
        regularizer: Regularizer,
        optimizer: optim.Optimizer,
        batch_size: int = 256,
        ce_weight=1.0,
        kd_weight=1.0,
        temperature=1.0,
        kd_loss="kd",
        verbose: bool = True,
        args=None,
        to_skip=None,
    ):
        self.model = model
        self.regularizer = regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose

        self.ce_weight = ce_weight
        self.kd_weight = kd_weight
        self.temperature = temperature
        self.kd_loss = kd_loss
        if args != None:
            self.args = args
        if to_skip != None:
            self.all_to_skip = to_skip
        self.reward_logs = []
        self.strategy_logs = []

    def epoch(self, examples: torch.LongTensor, e: int):
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        ce_loss = nn.CrossEntropyLoss(reduction="mean")
        mse_loss = nn.MSELoss(reduction="mean")
        self.strategy_logs = []
        with tqdm.tqdm(
            total=examples.shape[0], unit="ex", disable=not self.verbose, ncols=80
        ) as bar:
            bar.set_description(f"train loss")
            b_begin = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[
                    b_begin : b_begin + self.batch_size
                ].cuda()  # [batch, 3]
                truth = input_batch[:, 2]
                batch_num = int(b_begin / self.batch_size)
                if self.kd_loss == "kdrl" or self.kd_loss == "kdrl_step":
                    (
                        preds_student,
                        fac_student,
                        preds_selected,
                        reward_loss,
                        reward,
                        advantage,
                        strategy_selected,
                    ) = self.model.forward_train(input_batch, e)

                    l_reward = self.args.rl_weight * reward_loss
                    l_ce = self.ce_weight * ce_loss(preds_student, truth)
                    l_reg = self.regularizer.forward(fac_student)
                    golden_truths = [
                        self.all_to_skip[tuple(input_batch[i][:2].tolist())]
                        for i in range(len(truth))
                    ]
                    # l_kd = self.kd_weight * kd_loss(preds_student, preds_selected, self.temperature)
                    # l_kd = self.kd_weight * dkd_loss(preds_student, preds_selected, truth,
                    #                                  self.args.dkd_alpha, self.args.dkd_beta, self.temperature)
                    l_kd = self.kd_weight * multi_dkd_loss(
                        preds_student,
                        preds_selected,
                        truth,
                        self.args.dkd_alpha,
                        self.args.dkd_beta,
                        self.temperature,
                        golden_truths,
                    )
                    l = l_ce + l_reg + l_kd + l_reward
                    # self.reward_logs.append(l_reward.item())
                    # self.reward_logs.append(advantage.sum().item())
                    # self.strategy_logs.append(strategy_selected.detach().cpu().numpy())
                else:
                    preds_student, fac_student, preds_assistant = (
                        self.model.forward_train(input_batch, batch_num)
                    )
                    l_ce = self.ce_weight * ce_loss(preds_student, truth)
                    l_reg = self.regularizer.forward(fac_student)
                    if self.kd_loss == "kd":
                        l_kd = self.kd_weight * kd_loss(
                            preds_student, preds_assistant, self.temperature
                        )
                        p_student = torch.softmax(preds_student, -1) / 2.0
                        p_teacher = torch.softmax(preds_assistant, -1) / 2.0
                        C_batch_student = p_student.mm(p_student.t())
                        C_batch_teacher = p_teacher.mm(p_teacher.t())
                        C_ent_student = p_student.t().mm(p_student)
                        C_ent_teacher = p_teacher.t().mm(p_teacher)
                        if self.args.kd_batch:
                            l_kd += mse_loss(C_batch_student, C_batch_teacher) * 1e4
                        if self.args.kd_class:
                            l_kd += mse_loss(C_ent_student, C_ent_teacher) * 1e7
                    elif self.kd_loss == "dkd":
                        l_kd = self.kd_weight * dkd_loss(
                            preds_student,
                            preds_assistant,
                            truth,
                            self.args.dkd_alpha,
                            self.args.dkd_beta,
                            self.temperature,
                        )
                    l = l_ce + l_reg + l_kd

                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                bar.set_postfix(loss=f"{l.item():.0f}")
        return l
