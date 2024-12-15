import torch 
from tqdm.auto import tqdm 


def collate_fn(batch):
    """
    Outputs:
        batch_x: a list of string
        batch_y: a list of int
    """
    batch_x = [item["text"] for item in batch]
    batch_y = [item["label"] for item in batch]
    return batch_x, batch_y 


def get_target_module_keys(model):
    keys = []
    for n, p in model.named_parameters():
        # if 'transformer.h' in n:
        if '.h' in n:
            keys.append(n)
        # if 'roberta.encoder.layer' in n:
        #     if len(p.shape) == 2:
        #         keys.append(n)
        #     elif 'bias' in n:
        #         keys.append(n)
    if len(keys) == 0:
        print("Warning: utils.get_target_module_keys() returns an empty list. Is this intended, or did you forget to change the name of the model?")
    return keys


def eval_forward_pass(model, tokenizer, dl, device):

    preds, ys_true = [], []
    with torch.no_grad():
        for batch_x, batch_y in dl:
            tok = tokenizer(batch_x, return_tensors='pt', padding=True).to(device)
            logits = model(**tok).logits 
            maxval, maxid = torch.max(logits, -1)
            preds.append(maxid)  # (B)
            ys_true += batch_y
    preds = torch.cat(preds, 0)
    ys_true = torch.tensor(ys_true).to(device)
    return preds, ys_true 


def eval_mask_model(masked_model, eval_dl, tokenizer, device):
    masked_model.model.eval()
    with torch.no_grad():
        masked_model.apply_masks()
        pruned_model_density = masked_model.get_pruned_model_density()

        # Mask: only keep the params in the circuit
        preds_circ, ys_true = eval_forward_pass(masked_model, tokenizer, eval_dl, device)
        masked_model.remove_masks()

        # Reverse mask: keep only the params "other than" the circuit
        masked_model.apply_masks(reverse_mask=True)
        preds_other, _ = eval_forward_pass(masked_model, tokenizer, eval_dl, device)
        masked_model.remove_masks()

        # No mask: the full model
        preds_full, _ = eval_forward_pass(masked_model, tokenizer, eval_dl, device)

        N = len(preds_circ)
        acc_circ = (preds_circ == ys_true).sum() / N
        acc_other = (preds_other == ys_true).sum() / N
        acc_faith = (preds_circ == preds_full).sum() / N

    return {
        'acc_circ': acc_circ.item(),
        'acc_other': acc_other.item(),
        'acc_faith': acc_faith.item(),
        'model_density': pruned_model_density.item()
    }
