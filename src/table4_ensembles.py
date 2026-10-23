"""Isolated teacher-ensemble ablations; no changes to existing training paths."""

import torch
from torch import nn
from torch.nn import functional as F

STRATEGIES = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.5, 0.5, 0.0),
    (0.5, 0.0, 0.5),
    (0.0, 0.5, 0.5),
    (1 / 3, 1 / 3, 1 / 3),
)


class MetaLearner(nn.Module):
    """MoSE entity-wise 3 -> 32 -> 1 ReLU scorer (batch axis preserved)."""

    def __init__(self, hidden=32):
        super().__init__()
        self.fc1 = nn.Linear(3, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, logits):
        return F.relu(self.fc2(F.relu(self.fc1(logits)))).squeeze(-1)


class ValidationRC(nn.Module):
    """DSoM seven-action policy, learned directly against average-teacher CE."""

    def __init__(self, entities, hidden=1024):
        super().__init__()
        self.fc1 = nn.Linear(entities * 3, hidden)
        self.fc2 = nn.Linear(hidden, 7)
        self.dropout = nn.Dropout(0.1)
        self.register_buffer("strategies", torch.tensor(STRATEGIES))

    def policy_logits(self, logits):
        return self.fc2(F.relu(self.fc1(self.dropout(logits.flatten(1)))))

    def combine(self, logits, actions):
        return (logits * self.strategies[actions, None, :]).sum(-1)

    def forward(self, logits):
        # Deployment sees teacher predictions only, never target labels.
        return self.combine(logits, self.policy_logits(logits).argmax(-1))

    def reinforce_loss(self, logits, targets):
        policy = torch.distributions.Categorical(logits=self.policy_logits(logits))
        actions = policy.sample()
        with torch.no_grad():
            selected_ce = F.cross_entropy(
                self.combine(logits, actions), targets, reduction="none"
            )
            average_ce = F.cross_entropy(logits.mean(-1), targets, reduction="none")
            # Binary +1/-10 versus fixed average reference, baseline reward -10.
            advantage = torch.where(selected_ce < average_ce, 11.0, 0.0)
        return -(policy.log_prob(actions) * advantage).mean()


def filtered_ranks(scores, queries, filters):
    """Pessimistic ties, exclude known answers and target after score fusion."""
    targets = scores.gather(1, queries[:, 2:3]).clone()
    masked = scores.clone()
    for i, (head, relation, tail) in enumerate(queries.tolist()):
        # Benchmark pickle filters contain NumPy uint64 scalars. Mixed unsigned
        # scalars and Python ints cannot be promoted by PyTorch indexing.
        excluded = [int(entity) for entity in filters.get((head, relation), ())]
        excluded.append(tail)
        indices = torch.tensor(excluded, dtype=torch.long, device=masked.device)
        masked[i, indices] = -torch.inf
    return 1 + (masked >= targets).sum(-1)


def metrics(ranks):
    ranks = ranks.double()
    return {
        "MR": ranks.mean().item(),
        "MRR": ranks.reciprocal().mean().item(),
        "hits@[1,3,10]": [(ranks <= k).double().mean().item() for k in (1, 3, 10)],
    }


def average_metrics(rows):
    return {
        "MR": sum(r["MR"] for r in rows) / len(rows),
        "MRR": sum(r["MRR"] for r in rows) / len(rows),
        "hits@[1,3,10]": [
            sum(r["hits@[1,3,10]"][i] for r in rows) / len(rows) for i in range(3)
        ],
    }


@torch.no_grad()
def teacher_logits(teacher, queries):
    out = teacher(queries)
    logits = torch.stack((out[0], out[2], out[4]), dim=-1).detach()
    if not torch.isfinite(logits).all():
        raise ValueError("Non-finite teacher logits")
    return logits


def fit_epoch(teacher, model, optimizer, queries_by_side, device, batch_size):
    model.train()
    total, count = 0.0, 0
    updates = 0
    # Match MoSE order: shuffled RHS, then independently shuffled LHS.
    for queries in queries_by_side.values():
        for indices in torch.randperm(len(queries)).split(batch_size):
            batch = queries[indices].to(device)
            logits = teacher_logits(teacher, batch)
            loss = (
                model.reinforce_loss(logits, batch[:, 2])
                if isinstance(model, ValidationRC)
                else F.cross_entropy(model(logits), batch[:, 2])
            )
            if not torch.isfinite(loss):
                raise ValueError("Non-finite validation training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nonzero = any(
                p.grad is not None and bool(p.grad.abs().max() > 0)
                for p in model.parameters()
            )
            optimizer.step()
            updates += int(nonzero)
            total += loss.item() * len(batch)
            count += len(batch)
    return {"loss": total / count, "batches_with_nonzero_gradient": updates}


@torch.no_grad()
def evaluate(teacher, models, queries_by_side, filters, device, batch_size):
    for model in models.values():
        model.eval()
    ranks = {
        name: []
        for name in (
            "structural",
            "visual",
            "description",
            "teacher_mean_logits",
            *models,
        )
    }
    for side, queries in queries_by_side.items():
        for batch in queries.split(batch_size):
            batch = batch.to(device)
            logits = teacher_logits(teacher, batch)
            scores = {
                name: logits[..., i]
                for i, name in enumerate(("structural", "visual", "description"))
            }
            scores["teacher_mean_logits"] = logits.mean(-1)
            scores.update({name: model(logits) for name, model in models.items()})
            for name, values in scores.items():
                if not torch.isfinite(values).all():
                    raise ValueError(f"Non-finite scores: {name}")
                ranks[name].append(filtered_ranks(values, batch, filters[side]).cpu())
    result = {name: metrics(torch.cat(values)) for name, values in ranks.items()}
    result["teacher_mean_metrics"] = average_metrics(
        [result[name] for name in ("structural", "visual", "description")]
    )
    return result
