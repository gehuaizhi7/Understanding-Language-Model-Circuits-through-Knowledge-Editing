from fine_tune import apply_ft_to_model
import torch
import numpy as np
import typing



def edit(prompts, target_new, ground_truth, model, tok, args, device):

    requests = [{
        'prompt': prompt,
        'target_new': target_new_,
        'ground_truth': ground_truth_} for prompt, target_new_, ground_truth_ in zip(prompts, target_new, ground_truth)]
    
    def test_batch_prediction(
        model,
        tok,
        prefixes: typing.List[str],
        target_new: str,
        target_true: str,
    ):
    
        prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
        prompt_tok = tok(
            [
                f"{prefix} {suffix}"
                for prefix in prefixes
                for suffix in [target_new, target_true]
            ],
            padding=True,
            return_tensors="pt",
        ).to("cuda")
    
        a_tok, b_tok = (tok(f" {n}")["input_ids"] for n in [target_new, target_true])
        choice_a_len, choice_b_len = (len(n) for n in [a_tok, b_tok])
    
        with torch.no_grad():
            logits = model(**prompt_tok).logits
        results = np.zeros((logits.size(0),), dtype=np.float32)
    
        for i in range(logits.size(0)):
            cur_len = choice_a_len if i % 2 == 0 else choice_b_len
            for j in range(cur_len):
                cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]
                results[i] += -torch.nn.functional.log_softmax(
                    logits[i, prefix_lens[i // 2] + j - 1, :], dim=0
                )[cur_tok].item()
            results[i] /= cur_len
    
        return [
            {"target_new": results[i].item(), "target_true": results[i + 1].item()}
            for i in range(0, len(results), 2)
        ]
    
    metrics = []

    for i, request in enumerate(requests):
        print(f"Request {i}")
        print(request)
        print("preprepre")
        preprob = test_batch_prediction(model, tok, [request["prompt"]], request["target_new"], request["ground_truth"])
        print(preprob)

        # input_tensor = torch.tensor([tok.encode("A cat is a kind of")]).to(device)
        # generated_sequence = model.generate(input_tensor, max_length=20)
        # generated_text = tok.decode(generated_sequence[0])
        # print(generated_text)
        
        print("preedit"*20)
        iii = 0
        for k,v in list(model.named_parameters()):
            if iii == 6 or iii == 7:
                print(k)
                print(torch.eq(v, 0).sum().item())
            iii += 1

        edited_model, _ = apply_ft_to_model(
            model=model, 
            tok=tok,
            requests=[request],
            args=args,
            device=device,
            copy=True
            )
        

        print("postedit"*20)
        iii = 0
        for k,v in list(model.named_parameters()):
            if iii == 6 or iii == 7:
                print(k)
                print(torch.eq(v, 0).sum().item())
            iii += 1

        # generated_sequence = edited_model.generate(input_tensor, max_length=20)
        # generated_text = tok.decode(generated_sequence[0])
        # print(generated_text)
        print("postpostpost")
        postprob = test_batch_prediction(edited_model, tok, [request["prompt"]], request["target_new"], request["ground_truth"])
        print(postprob)

        metrics.append({
            "id": i,
            "preprob": preprob,
            "postprob": postprob
        })


    return metrics