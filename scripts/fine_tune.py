from copy import deepcopy
from typing import Any, Dict, List, Tuple
from collections import deque

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

import nethook



def apply_ft_to_model(
    model,
    tok: AutoTokenizer,
    requests: List[Dict],
    args: Dict,
    device: torch.device,
    copy=True,
    return_orig_weights=False,
    keep_original_weight=False,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) the weights that changed
    """
    weights_copy = {}
    if copy:
        model = deepcopy(model)

    execute_ft(model, tok, args, requests, device)
    # deltas = execute_ft(model, tok, args, requests, device)

    # with torch.no_grad():
    #     for w_name, upd_matrix in deltas.items():
    #         w = nethook.get_parameter(model, w_name)
    #         if return_orig_weights and w_name not in weights_copy:
    #             weights_copy[w_name] = w.detach().clone()

    #         w[...] += upd_matrix

    # print(f"New weights successfully inserted.")

    if not keep_original_weight:
        weights_copy = {}

    return model, weights_copy


def execute_ft(
    model,
    tok: AutoTokenizer,
    args: Dict,
    requests: List[Dict],
    device: torch.device,
    **kwargs: Any,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the FT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """
    # model = model.to(device)
    # Update target and print info
    requests = deepcopy(requests)
    print(requests)
    for request in requests:
        if request["target_new"] != " ":
            # Space required for correct tokenization
            request["target_new"] = " " + request["target_new"]
        print(
            f"Executing FT algo for: "
            f"[{request['prompt']}] -> [{request['target_new']}]"
        )
    
    # Retrieve weights that user desires to change
    weights = {n:p for n, p in model.named_parameters()}
    
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    # print(f"Weights to be updated: {list(weights.keys())}")

    # Define inputs
    texts = [r["prompt"] for r in requests]
    targets = [r["target_new"] for r in requests]
    
    # Configure optimizer / gradients
    opt = torch.optim.SGD(
        [v for _, v in weights.items()],
        lr=args["lr_ft"],
        weight_decay=args["weight_decay_ft"],
    )

    # print('masked model parameters-----------------------------')
    # for name, param in list(model.named_parameters()):
    #     print(name)

    # for name, w in model.named_parameters():
    #     w.requires_grad = name in weights

    # Update loop: intervene at layers simultaneously
    loss_meter = AverageMeter()
    for it in range(args["num_steps_ft"]):
        print(20 * "=")
        print(f"Epoch: {it}")
        print(20 * "=")
        loss_meter.reset()

        for txt, tgt in zip(
            chunks(texts, args["batch_size_ft"]), chunks(targets, args["batch_size_ft"])
        ):
            inputs = tok(txt, return_tensors="pt", padding=True).to(device)
            target_ids = tok(tgt, return_tensors="pt", padding=True)["input_ids"].to(
                device
            )
            inputs_targets = [txt_ + tgt_ for txt_, tgt_ in zip(txt, tgt)]
            inputs_targets = tok(inputs_targets, return_tensors="pt", padding=True).to(device)
            # last_token_inds = inputs["attention_mask"].sum(dim=1) - 1
            # loss_mask = inputs != tok.unk_token_id
            # loss_mask = [:, ]
            num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in inputs['input_ids'].cpu()]
            num_pad_toks = [int((i == tok.pad_token_id).sum()) for i in inputs_targets['input_ids'].cpu()]
            prompt_len = [x + y for x, y in zip(num_pad_toks, num_prompt_toks)]
            prompt_target_len = inputs_targets['input_ids'].size(1)
            label_mask = torch.tensor([[False] * length + [True] * (prompt_target_len - length) for length in prompt_len]).to(device)
            opt.zero_grad()
            bs = inputs["input_ids"].shape[0]
            
            logits = model(**inputs_targets).logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs_targets['input_ids'][..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(bs, -1)
            loss = (loss * label_mask[:,1:]).sum(1) / label_mask[:,1:].sum(1)
            loss = loss.mean()
            print(f"Batch loss {loss.item()}")
            loss_meter.update(loss.item(), n=bs)

            if loss.item() >= 1e-2:
                loss.backward()
                for n, p in model.model.named_parameters():
                    if n in model.mask_param_names:
                        mask = model.masks[n]
                        p.grad.data = mask * p.grad.data
                        # print('-grad-'*20)
                        # print(n)
                        # print(torch.eq(p, 0).sum().item())
                opt.step()

            if type(args["norm_constraint_ft"]) is float:
                eps = args["norm_constraint_ft"]
                with torch.no_grad():
                    for k, v in weights.items():
                        v[...] = torch.clamp(
                            v, min=weights_copy[k] - eps, max=weights_copy[k] + eps
                        )

        print(f"Total loss {loss_meter.avg}")

        if loss_meter.avg < 1e-2:
            break

    # deltas = {k: (weights[k] - weights_copy[k]).detach() for k in weights}

    # # Restore state of original model
    # with torch.no_grad():
    #     for k, v in weights.items():
    #         v[...] = weights_copy[k]

    # print(f"Deltas successfully computed.")

    # return deltas


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count