from typing import Dict

import torch


def avg_both(
    mrs: Dict[str, float], mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]
):
    mr = (mrs["lhs"] + mrs["rhs"]) / 2
    mrr = (mrrs["lhs"] + mrrs["rhs"]) / 2.0
    h = (hits["lhs"] + hits["rhs"]) / 2.0

    return {"MR": mr, "MRR": mrr, "hits@[1,3,10]": h.tolist()}
