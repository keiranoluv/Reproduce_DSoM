"""Raw/filtered structural student ranking, independent of teacher oracle ranks."""

import torch


def ranking_pair(scores, targets, queries, filters):
    """Pessimistic ties, target excluded in both protocols, other answers raw-only."""
    scores = scores.clone()
    scores.scatter_(1, queries[:, 2:3], -torch.inf)
    raw = 1 + (scores >= targets).sum(-1)
    for i, (head, relation, _) in enumerate(queries.tolist()):
        excluded = [int(e) for e in filters.get((head, relation), ())]
        indices = torch.tensor(excluded, dtype=torch.long, device=scores.device)
        scores[i, indices] = -torch.inf
    filtered = 1 + (scores >= targets).sum(-1)
    return raw, filtered


def metrics(ranks):
    ranks = ranks.double()
    return {
        "MR": ranks.mean().item(),
        "MRR": ranks.reciprocal().mean().item(),
        "hits@[1,3,10]": [(ranks <= k).double().mean().item() for k in (1, 3, 10)],
    }


@torch.inference_mode()
def evaluate_student(
    model, examples, filters, device, batch_size=2000, limit=0, seed=2022
):
    if model.modality_split or model.fusion_img or model.fusion_dscp:
        raise ValueError(
            "Table 7 must evaluate the structural student, not multimodal teacher"
        )
    model.eval()
    queries = torch.as_tensor(examples.astype("int64"))
    if limit > 0:
        indices = torch.randperm(
            len(queries), generator=torch.Generator().manual_seed(seed)
        )[:limit]
        queries = queries[indices]
    ranks = {protocol: {} for protocol in ("raw", "filtered")}
    for side in ("rhs", "lhs"):
        side_queries = queries.clone()
        if side == "lhs":
            side_queries[:, 0], side_queries[:, 2] = queries[:, 2], queries[:, 0]
            side_queries[:, 1] += model.sizes[1] // 2
        current = {protocol: [] for protocol in ranks}
        rhs = model.get_rhs(0, model.sizes[2])[0]
        for batch in side_queries.split(batch_size):
            batch = batch.to(device)
            scores = model.get_queries(batch)[0] @ rhs
            targets = model.score(batch)[0]
            if not torch.isfinite(scores).all() or not torch.isfinite(targets).all():
                raise ValueError("Non-finite student scores")
            raw, filtered = ranking_pair(scores, targets, batch, filters[side])
            current["raw"].append(raw.cpu())
            current["filtered"].append(filtered.cpu())
        for protocol in ranks:
            ranks[protocol][side] = torch.cat(current[protocol])
    result = {}
    for protocol, by_side in ranks.items():
        result[protocol] = {
            "both": metrics(torch.cat(list(by_side.values()))),
            **{side: metrics(values) for side, values in by_side.items()},
        }
    result["queries_per_direction"] = len(queries)
    return result
