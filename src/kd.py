import torch
import torch.nn as nn
import torch.nn.functional as F


class Distiller(nn.Module):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher

    def train(self, mode=True):
        # teacher as eval mode by default
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.teacher.eval()
        return self

    def get_learnable_parameters(self):
        # if the method introduces extra parameters, re-impl this function
        return [v for k, v in self.student.named_parameters()]

    def get_extra_parameters(self):
        # calculate the extra parameters introduced by the distiller
        return 0

    def forward_train(self, **kwargs):
        # training function for the distillation method
        raise NotImplementedError()

    def forward_test(self, x):
        return self.student(x)

    def forward(self, x):
        if self.training:
            return self.forward_train(x)
        return self.forward_test(x)


def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd


def kd_loss_samplewise(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none")
    # loss_kd *= temperature ** 2
    return loss_kd


class KD(Distiller):
    def __init__(self, student, teacher, assistant):
        super(KD, self).__init__(student, teacher)
        self.assistant = assistant

    def forward_train(self, x, epoch):
        preds_student, fac_student = self.student(x)
        with torch.no_grad():
            preds_str, fac_str, preds_img, fac_img, preds_dscp, fac_dscp = (
                self.teacher.forward(x)
            )  # get_ranking_score or forward
            # assistant trainable or fixed
            preds_teacher = torch.stack([preds_str, preds_img, preds_dscp]).permute(
                1, 2, 0
            )  # [b, 3]
        preds_assistant = self.assistant(preds_teacher)

        return preds_student, fac_student, preds_assistant


class KD_BestTeacher(Distiller):
    def __init__(self, student, teacher):
        super(KD_BestTeacher, self).__init__(student, teacher)
        self.ce = nn.CrossEntropyLoss(reduction="none")
        print("init KD_BestTeacher")

    def forward_train(self, x, epoch):
        truth = x[:, 2]
        preds_student, fac_student = self.student(x)
        with torch.no_grad():
            preds_str, fac_str, preds_img, fac_img, preds_dscp, fac_dscp = (
                self.teacher.forward(x)
            )  # get_ranking_score or forward [b, Ents]
            ce_str = self.ce(preds_str, truth)  # [b]
            ce_img = self.ce(preds_img, truth)
            ce_dscp = self.ce(preds_dscp, truth)
            min_n_ce = torch.min(ce_str, torch.min(ce_img, ce_dscp))
            mask = torch.stack(
                [(min_n_ce == ce_str), (min_n_ce == ce_img), (min_n_ce == ce_dscp)]
            ).float()
            mask = mask / torch.sum(mask, dim=0)  # [3, b]
            mask = mask.unsqueeze(-1)  # [3, b, 1]
            preds_stack = torch.stack([preds_str, preds_img, preds_dscp])  # [3,b,E]
            preds_best_teacher = (mask * preds_stack).sum(dim=0)  # [b,E]
        return preds_student, fac_student, preds_best_teacher


class KD_SingleTeacher(Distiller):
    def __init__(self, student, teacher, args):
        super(KD_SingleTeacher, self).__init__(student, teacher)
        self.ce = nn.CrossEntropyLoss(reduction="none")
        self.args = args

        print("init KD_SingleTeacher")

    def forward_train(self, x, epoch):
        truth = x[:, 2]
        preds_student, fac_student = self.student(x)
        with torch.no_grad():
            preds_str, fac_str, preds_img, fac_img, preds_dscp, fac_dscp = (
                self.teacher.forward(x)
            )  # get_ranking_score or forward [b, Ents]
            ce_str = self.ce(preds_str, truth)  # [b]
            ce_img = self.ce(preds_img, truth)
            ce_dscp = self.ce(preds_dscp, truth)
        if self.args.fusion_img == True and self.args.fusion_dscp == False:
            preds_teacher = preds_img
        elif self.args.fusion_img == False and self.args.fusion_dscp == True:
            preds_teacher = preds_dscp
        elif self.args.fusion_img == False and self.args.fusion_dscp == False:
            preds_teacher = preds_str
        return preds_student, fac_student, preds_teacher


class KD_RandTeacher(Distiller):
    def __init__(self, student, teacher):
        super(KD_RandTeacher, self).__init__(student, teacher)
        self.ce = nn.CrossEntropyLoss(reduction="none")
        print("init KD_RandTeacher")

    def forward_train(self, x, epoch):
        truth = x[:, 2]
        preds_student, fac_student = self.student(x)
        with torch.no_grad():
            preds_str, fac_str, preds_img, fac_img, preds_dscp, fac_dscp = (
                self.teacher.forward(x)
            )  # get_ranking_score or forward [b, Ents]\
            teacher_selected = torch.randint(0, 3, (preds_str.shape[0],))
            preds_stack = torch.stack([preds_str, preds_img, preds_dscp])  # [3,b,E]
            preds_best_teacher = preds_stack[
                teacher_selected, range(preds_str.shape[0])
            ]
        return preds_student, fac_student, preds_best_teacher


class KD_AvgTeacher(Distiller):
    def __init__(self, student, teacher):
        super(KD_AvgTeacher, self).__init__(student, teacher)
        self.ce = nn.CrossEntropyLoss(reduction="none")
        print("init KD_AvgTeacher")

    def forward_train(self, x, epoch):
        truth = x[:, 2]
        preds_student, fac_student = self.student(x)
        with torch.no_grad():
            preds_str, fac_str, preds_img, fac_img, preds_dscp, fac_dscp = (
                self.teacher.forward(x)
            )  # get_ranking_score or forward [b, Ents]\
            preds_stack = torch.stack([preds_str, preds_img, preds_dscp])  # [3,b,E]
            preds_best_teacher = preds_stack.mean(dim=0)
        return preds_student, fac_student, preds_best_teacher


class KD_BestStrategy(Distiller):
    def __init__(self, student, teacher):
        super(KD_BestStrategy, self).__init__(student, teacher)
        self.ce = nn.CrossEntropyLoss(reduction="none")
        self.strategy = torch.tensor(
            [
                [1.0, 0, 0],
                [0, 1.0, 0],
                [0, 0, 1.0],
                [0.5, 0.5, 0],
                [0.5, 0, 0.5],
                [0, 0.5, 0.5],
                [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            ]
        ).cuda()
        print("init KD_BestStrategy")

    def forward_train(self, x, epoch):
        truth = x[:, 2]
        preds_student, fac_student = self.student(x)
        with torch.no_grad():
            preds_str, fac_str, preds_img, fac_img, preds_dscp, fac_dscp = (
                self.teacher.forward(x)
            )  # get_ranking_score or forward [b, Ents]
            preds_stack = torch.stack([preds_str, preds_img, preds_dscp]).permute(
                1, 0, 2
            )  # [b,3,E]
            bsz, msz, esz = preds_stack.shape
            strategy_count = self.strategy.shape[0]
            mod_mask = self.strategy.unsqueeze(0).broadcast_to(
                bsz, strategy_count, msz
            )  # b, 7, 3
            preds_strategy_all = mod_mask.bmm(preds_stack)  # b,7,E
            presd_strategy_all_reshape = preds_strategy_all.reshape(
                bsz * strategy_count, esz
            )
            truth_strategy_all = (
                truth.unsqueeze(0)
                .broadcast_to(strategy_count, bsz)
                .t()
                .reshape(bsz * strategy_count)
            )
            ce_strategy = self.ce(
                presd_strategy_all_reshape, truth_strategy_all
            ).reshape(bsz, strategy_count)  # b*7
            strategy_selected = ce_strategy.argmin(dim=-1)
            preds_strategy_selected = preds_strategy_all[range(bsz), strategy_selected]
        return preds_student, fac_student, preds_strategy_selected


class KD_BestConf(Distiller):
    def __init__(self, student, teacher):
        super(KD_BestConf, self).__init__(student, teacher)
        print("init KD_BestConf")

    def forward_train(self, x, epoch):
        truth = x[:, 2]
        preds_student, fac_student = self.student(x)
        with torch.no_grad():
            preds_str, fac_str, preds_img, fac_img, preds_dscp, fac_dscp = (
                self.teacher.forward(x)
            )  # get_ranking_score or forward [b, Ents]
            p_str = F.softmax(preds_str, dim=-1)
            p_img = F.softmax(preds_img, dim=-1)
            p_dscp = F.softmax(preds_dscp, dim=-1)
            conf_str = p_str.max(dim=-1)[0]
            conf_img = p_img.max(dim=-1)[0]
            conf_dscp = p_dscp.max(dim=-1)[0]
            confs = torch.stack([conf_str, conf_img, conf_dscp])
            modal_best_conf = confs.max(dim=0)[1]
            preds_stack = torch.stack([preds_str, preds_img, preds_dscp])  # [3,b,E]
            preds_best_teacher = preds_stack.permute(1, 0, 2)[
                range(p_str.shape[0]), modal_best_conf
            ]
        return preds_student, fac_student, preds_best_teacher


class MLP(torch.nn.Module):
    def __init__(self, E_size, hidden_size, temp=1.0, p_dropout=0.1):
        """
        input : T, E, M
        output: T, M, 1 -> T, M
        """
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(E_size * 3, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, 7)
        self.dropout = torch.nn.Dropout(p=p_dropout)
        self.temp = temp
        print("hidden_size", hidden_size)  # S, V, T, SV, ST, VT, SVT
        for n, p in self.named_parameters():
            print(n, p.shape)

    def forward(self, logits):
        # logits # T, E, M -> T, E*M
        logits = logits.reshape(logits.shape[0], -1)
        logits = self.dropout(logits)
        logits = F.relu(self.fc1(logits))  # T, 3
        logits = self.fc2(logits)
        output = F.softmax(logits / self.temp, dim=-1)
        return output


class MLP2(torch.nn.Module):
    def __init__(self, E_size, hidden_size, temp=1.0, p_dropout=0.1):
        """
        input : T, E, M
        output: T, M, 1 -> T, M
        """
        super(MLP2, self).__init__()
        self.fc1 = torch.nn.Linear(E_size * 2, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, 3)
        self.dropout = torch.nn.Dropout(p=p_dropout)
        self.temp = temp
        print("hidden_size", hidden_size)  # S, V, T, SV, ST, VT, SVT
        for n, p in self.named_parameters():
            print(n, p.shape)

    def forward(self, logits):
        # logits # T, E, M -> T, E*M
        logits = logits.reshape(logits.shape[0], -1)
        logits = self.dropout(logits)
        logits = F.relu(self.fc1(logits))  # T, 3
        logits = self.fc2(logits)
        output = F.softmax(logits / self.temp, dim=-1)
        return output


class MLP_value(torch.nn.Module):
    def __init__(self, E_size, hidden_size, temp=1.0, p_dropout=0.1):
        """
        input : T, E, M
        output: T, M, 1 -> T, M
        """
        super(MLP_value, self).__init__()
        self.fc1 = torch.nn.Linear(E_size * 3, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, 1)
        self.dropout = torch.nn.Dropout(p=p_dropout)
        self.temp = temp
        # self.fc3 = torch.nn.Linear(hidden_size, 3)
        print("hidden_size", hidden_size)  # S, V, T, SV, ST, VT, SVT
        for n, p in self.named_parameters():
            print(n, p.shape)

    def forward(self, logits):
        # logits # T, E, M -> T, E*M
        logits = logits.reshape(logits.shape[0], -1)
        logits = self.dropout(logits)
        logits = F.relu(self.fc1(logits))  # T, 3
        logits = self.fc2(logits)
        return logits


class KD_RL(Distiller):
    def __init__(
        self,
        student,
        teacher,
        E_size,
        rl_hs=128,
        reward=1.0,
        penalize=-10.0,
        policy_temp=1.0,
        args=None,
    ):
        super(KD_RL, self).__init__(student, teacher)
        self.ce = nn.CrossEntropyLoss(reduction="none")
        self.args = args
        if args.fusion_img == True and args.fusion_dscp == True:
            self.strategy = torch.tensor(
                [
                    [1.0, 0, 0],
                    [0, 1.0, 0],
                    [0, 0, 1.0],
                    [0.5, 0.5, 0],
                    [0.5, 0, 0.5],
                    [0, 0.5, 0.5],
                    [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                ]
            ).cuda()
            self.rewarder = MLP(E_size=E_size, hidden_size=rl_hs, temp=policy_temp)
        if args.fusion_img == False or args.fusion_dscp == False:
            self.strategy = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]).cuda()
            self.rewarder = MLP2(E_size=E_size, hidden_size=rl_hs, temp=policy_temp)

        self.reward = reward
        self.penalize = -penalize
        self.valuenet = MLP_value(E_size=E_size, hidden_size=rl_hs, temp=policy_temp)

    def forward_train(self, x, epoch):
        truth = x[:, 2]
        preds_student, fac_student = self.student(x)
        with torch.no_grad():
            preds_str, fac_str, preds_img, fac_img, preds_dscp, fac_dscp = (
                self.teacher.forward(x)
            )  # get_ranking_score or forward [b, Ents]
        if self.args.fusion_img == False:
            preds_stack = torch.stack([preds_str, preds_dscp]).permute(1, 2, 0)
        elif self.args.fusion_dscp == False:
            preds_stack = torch.stack([preds_str, preds_img]).permute(1, 2, 0)
        else:
            preds_stack = torch.stack([preds_str, preds_img, preds_dscp]).permute(
                1, 2, 0
            )
        logits = self.rewarder(preds_stack)
        strategy_selected = torch.multinomial(logits, 1).squeeze()
        mod_mask = (
            self.strategy[strategy_selected]
            .unsqueeze(1)
            .broadcast_to(preds_stack.shape)
        )
        reverse_mod_mask = 1.0 - mod_mask
        preds_selected = preds_stack.mul(mod_mask).sum(dim=-1)
        preds_reverse = preds_stack.mul(reverse_mod_mask).sum(dim=-1)
        avg_preds = preds_stack.mean(dim=-1)
        reward = torch.where(
            (self.ce(preds_selected, truth) < self.ce(preds_student, truth)),
            self.reward,
            self.penalize,
        )
        reward_baseline = torch.where(
            (self.ce(avg_preds, truth) < self.ce(preds_student, truth)),
            self.reward,
            self.penalize,
        )

        advantage = reward - reward_baseline.detach()
        log_prob = torch.log(logits[range(len(preds_stack)), strategy_selected])
        reward_loss = (-log_prob * advantage).mean()
        return (
            preds_student,
            fac_student,
            preds_selected,
            reward_loss,
            reward,
            advantage,
            strategy_selected,
        )
