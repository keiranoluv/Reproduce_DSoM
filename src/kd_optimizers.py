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
        policy_optimizer=None,
    ):
        self.model = model
        self.regularizer = regularizer
        self.optimizer = optimizer
        self.policy_optimizer = policy_optimizer
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
        self.model.train()
        rc_checked_batches = 0
        rc_nonzero_adv_batches = 0
        rc_nonzero_grad_batches = 0
        rc_updated_batches = 0
        actual_examples = examples[torch.randperm(examples.shape[0]), :]

        def probe_policy(batch):
            policy = self.model.rewarder
            was_training = policy.training

            try:
                policy.eval()

                with torch.no_grad():
                    teacher_out = self.model.teacher.forward(batch)
                    preds_stack = torch.stack(
                        [teacher_out[0], teacher_out[2], teacher_out[4]],
                        dim=-1,
                    )

                    probs = self.model.policy_probs(preds_stack)
                    entropy = (
                        -(probs * probs.clamp_min(1e-12).log())
                        .sum(dim=-1)
                        .mean()
                        .item()
                    )

                    stats = policy.last_forward_stats.copy()

                    return {
                        "entropy": entropy,
                        "hidden_rms": stats["hidden_rms"],
                        "logit_range": stats["raw_logit_range"],
                        "mean_probs": probs.mean(dim=0).cpu().tolist(),
                    }
            finally:
                policy.train(was_training)

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
                    elif self.kd_loss == "fixed_ndkd":
                        golden_truths = [
                            self.all_to_skip[tuple(input_batch[i][:2].tolist())]
                            for i in range(len(truth))
                        ]

                        l_kd = self.kd_weight * multi_dkd_loss(
                            preds_student,
                            preds_assistant,
                            truth,
                            self.args.dkd_alpha,
                            self.args.dkd_beta,
                            self.temperature,
                            golden_truths,
                        )

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

                # Detect non-finite training loss.
                if not torch.isfinite(l).item():
                    raise FloatingPointError(
                        f"Non-finite loss at epoch {e + 1}, "
                        f"batch {batch_num}: {l.item()}"
                    )

                check_policy = self.kd_loss in ("kdrl", "kdrl_step")

                if check_policy:
                    policy_weight = self.model.rewarder.fc2.weight
                    policy_before = policy_weight.detach().clone()

                self.optimizer.zero_grad()

                if self.policy_optimizer is not None:
                    self.policy_optimizer.zero_grad()

                l.backward()

                if check_policy:
                    rc_checked_batches += 1

                    nonzero_adv = (advantage.detach() != 0).sum().item()
                    grad = policy_weight.grad

                    grad_norm = grad.detach().norm().item() if grad is not None else 0.0

                    if nonzero_adv > 0:
                        rc_nonzero_adv_batches += 1

                    if grad_norm > 0:
                        rc_nonzero_grad_batches += 1

                probe_this_batch = self.kd_loss == "kdrl" and e == 0 and batch_num < 20

                if probe_this_batch:
                    before_probe = probe_policy(input_batch)

                self.optimizer.step()

                if self.policy_optimizer is not None:
                    self.policy_optimizer.step()

                if probe_this_batch:
                    after_probe = probe_policy(input_batch)

                    print("\n===== FIXED-BATCH POLICY PROBE =====")

                    for key in ("entropy", "hidden_rms", "logit_range"):
                        print(
                            f"{key}: "
                            f"before={before_probe[key]:.6f} "
                            f"after={after_probe[key]:.6f}"
                        )

                    print("====================================")
                    before_probs = before_probe["mean_probs"]
                    after_probs = after_probe["mean_probs"]

                    print(f"===== RC DIRECTION batch={batch_num} =====")
                    print(
                        "Sampled actions:",
                        torch.bincount(
                            strategy_selected.detach(),
                            minlength=7,
                        )
                        .cpu()
                        .tolist(),
                    )

                    with torch.no_grad():
                        actions = strategy_selected.detach()
                        adv = advantage.detach()

                        pos = torch.bincount(
                            actions[adv > 0],
                            minlength=7,
                        )

                        neg = torch.bincount(
                            actions[adv < 0],
                            minlength=7,
                        )

                        adv_sum = torch.zeros(7, device=adv.device)
                        adv_sum.scatter_add_(0, actions, adv)

                        print("Positive advantage actions:", pos.cpu().tolist())
                        print("Negative advantage actions:", neg.cpu().tolist())
                        print("Advantage sum by strategy:", adv_sum.cpu().tolist())

                    print(
                        "Advantage counts:",
                        {
                            "positive": int((advantage > 0).sum().item()),
                            "negative": int((advantage < 0).sum().item()),
                            "zero": int((advantage == 0).sum().item()),
                        },
                    )

                    for idx in range(7):
                        delta = after_probs[idx] - before_probs[idx]
                        print(
                            f"Strategy {idx}: "
                            f"before={before_probs[idx]:.6f} "
                            f"after={after_probs[idx]:.6f} "
                            f"delta={delta:+.6f}"
                        )

                    print("====================================")

                if check_policy:
                    weight_change = (
                        (policy_weight.detach() - policy_before).abs().max().item()
                    )

                    if weight_change > 0:
                        rc_updated_batches += 1

                    print(
                        f"\nRC batch={batch_num} "
                        f"reward_mean={reward.detach().float().mean().item():.4f} "
                        f"adv_nonzero={nonzero_adv}/{len(advantage)} "
                        f"grad_norm={grad_norm:.6f} "
                        f"weight_change={weight_change:.8f}"
                    )

                    stats = self.model.last_policy_stats

                    print(
                        f"POLICY batch={batch_num} "
                        f"entropy={stats['entropy']:.6f} "
                        f"max_prob={stats['max_prob']:.6f} "
                        f"near_deterministic="
                        f"{stats['near_deterministic']}/{len(truth)} "
                        f"actions={stats['actions']}"
                    )

                    mlp_stats = self.model.rewarder.last_forward_stats

                    print(
                        f"MLP batch={batch_num} "
                        f"teacher_mean={mlp_stats['teacher_mean']:.4f} "
                        f"teacher_std={mlp_stats['teacher_std']:.4f} "
                        f"teacher_absmax={mlp_stats['teacher_absmax']:.4f} "
                        f"hidden_rms={mlp_stats['hidden_rms']:.4f} "
                        f"hidden_absmax={mlp_stats['hidden_absmax']:.4f} "
                        f"logit_absmax={mlp_stats['raw_logit_absmax']:.4f} "
                        f"logit_range={mlp_stats['raw_logit_range']:.4f}"
                    )

                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                bar.set_postfix(loss=f"{l.item():.0f}")

        if self.kd_loss in ("kdrl", "kdrl_step"):
            print("\n===== RC DIAGNOSTICS =====")
            print("Checked batches:", rc_checked_batches)
            print("Batches with nonzero advantage:", rc_nonzero_adv_batches)
            print("Batches with nonzero policy gradient:", rc_nonzero_grad_batches)
            print("Batches with updated policy weights:", rc_updated_batches)
            print("==========================")

        return l
