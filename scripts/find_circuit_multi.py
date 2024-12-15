# Based on the differentiable mask repo (Jade, Frank, Zining)

import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from masked_model import MaskedModel
from masked_generative_model import MaskedGenerativeModel
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, AutoModelForCausalLM
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils import collate_fn, get_target_module_keys, eval_forward_pass, eval_mask_model
from editor import edit
import os
import json
from transformers import AutoTokenizer, GPT2Model, GPT2Config
from torch.nn import CrossEntropyLoss, Linear, Dropout
from transformers.modeling_outputs import SequenceClassifierOutput





def main(args):
    print(args['task_name'])
    device = torch.device('cuda') 
    
    # model = AutoModelForSequenceClassification.from_pretrained(args['model_name']).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args['model_name'])
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
    if args['model_name']=='gpt2-xl':
        class GPT2ForSequenceClassification(torch.nn.Module):
            def __init__(self, config):
                super(GPT2ForSequenceClassification, self).__init__()
                self.config = config  # Store the config in the model
                self.num_labels = config.num_labels
                self.gpt2 = GPT2Model(config)
                self.dropout = Dropout(config.resid_pdrop)
                self.classifier = Linear(config.hidden_size, config.num_labels)
                self.score = self.classifier  # Alias for compatibility
            
            def forward(self, input_ids, attention_mask=None, labels=None):
                outputs = self.gpt2(input_ids, attention_mask=attention_mask)
                hidden_states = outputs.last_hidden_state
                pooled_output = hidden_states[:, -1, :]  # Take the hidden state of the last token
                pooled_output = self.dropout(pooled_output)
                logits = self.classifier(pooled_output)

                loss = None
                if labels is not None:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                
                return SequenceClassifierOutput(loss=loss, logits=logits)
        config = GPT2Config.from_pretrained('gpt2-xl', num_labels=2)
        model = GPT2ForSequenceClassification(config).to(device)
        # state_dict = model.state_dict()
        # new_state_dict = {}
        # for key, value in state_dict.items():
        #     new_key = key.replace("gpt2.", "transformer.")
        #     new_state_dict[new_key] = value

        # # Load the modified state dict back into the model
        # model.load_state_dict(new_state_dict)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args['model_name']).to(device)

    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.unk_token_id
    # Assuming tokenizer.pad_token has been set to tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    args['param_keys'] = get_target_module_keys(model)

    ds = load_dataset("csv", data_files=args["data_dir"])
    ds = ds["train"].train_test_split(test_size=0.2)

    train_dl = DataLoader(
        ds["train"],
        batch_size=args['batch_size'],
        collate_fn=collate_fn
    )
    eval_dl = DataLoader(
        ds["test"],
        batch_size=args['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )

    model.train()

    optim_m = torch.optim.AdamW([model.score.weight], lr=args['classification_lr'])
    for epoch in range(args['original_model_train_epochs']):
        for batch_x, batch_y in train_dl:
            tok = tokenizer(batch_x, return_tensors="pt", padding=True).to(device)
            batch_y = torch.tensor(batch_y).to(device)

            # with torch.no_grad():
            #     full_logits = model(**tok).logits  # (bsz, n_class)
            #     _, full_model_preds = torch.max(full_logits, 1)  

            logits = model(**tok).logits
            loss = F.cross_entropy(logits, batch_y)
            loss.backward()
            optim_m.step()
            optim_m.zero_grad()
            torch.cuda.empty_cache()
            
        print(f"Epoch {epoch+1}/{args['original_model_train_epochs']}, Loss: {loss.item()}")
    

    if args['resume_epoch'] > 0:
        masked_logits_fn = os.path.join(
            args['model_dir'], f"mask_logits/{args['task_name']}/{args['model_name']}-{args['resume_epoch']}.pt")
        masked_logits_dict = torch.load(masked_logits_fn)
        masked_logits_dict = {k: v.to(model.device) for k, v in masked_logits_dict.items()}
    else:
        masked_logits_dict = None

    masked_model = MaskedModel(model, args, masked_logits_dict)
    optim = torch.optim.AdamW(masked_model.get_trainable_parameters(), lr=args['lr'])

    masked_model.eval()
    preds, ys_true = eval_forward_pass(model, tokenizer, eval_dl, device)
    acc = (preds == ys_true).sum().item() / len(preds)

    print('model test accuracy: {:.4f}'.format(acc))

    # clear the cache if necessary-----------------------------------------------------
    # torch.cuda.empty_cache()
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    # masked_model = torch.nn.DataParallel(masked_model)
    # masked_model = masked_model.to(device)

    target_density = [0.5,0.35,0.25,0.15,0.05]
    point = 0

    for epoch in range(args['train_epochs']):
        losses = {
            'faithfulness': [],
            'faithfulness_v2': [],
            'sparseness': [],
        }
        # linear increasing of sparseness loss weight
        if args['lambda_sparse'] < 1000:
            args['lambda_sparse'] += 1

        step = 0
        for batch_x, batch_y in train_dl:
            tok = tokenizer(batch_x, return_tensors="pt", padding=True).to(device)
            batch_y = torch.tensor(batch_y).to(device)

            assert masked_model.is_masked == False 
            with torch.no_grad():
                full_logits = masked_model(**tok).logits  # (bsz, n_class)
                _, full_model_preds = torch.max(full_logits, 1)  

            # compute logits with param masks and the sparseness loss
            masked_model.apply_masks()
            masked_logits = masked_model(**tok).logits

            # faithfulness loss v1: cross entropy
            faith_loss = F.cross_entropy(masked_logits, full_model_preds)
            losses['faithfulness'].append(faith_loss.detach().item())

            # faithfulness loss v2: MSE
            # faith_loss_2 = F.mse_loss(masked_logits, full_logits)
            # losses['faithfulness_v2'].append(faith_loss_2.detach().item())

            # compute sparseness loss and then backprop the first two loss terms
            sparse_loss = masked_model.get_sparseness_loss()
            losses['sparseness'].append(sparse_loss.detach().item())
            
            # Aggregate the losses and optimize
            loss = sparse_loss * args['lambda_sparse'] + faith_loss
            loss.backward()
            optim.step()
            optim.zero_grad()
            masked_model.remove_masks()

            step += 1
            torch.cuda.empty_cache()

        results = eval_mask_model(
            masked_model, eval_dl, tokenizer, device)
        print(
            "Epoch {}. mean faithfulness loss {:.2f}, mean sparseness loss {:.2f}".format(
                epoch + 1, np.mean(losses['faithfulness']), np.mean(losses['sparseness']),
                ))
        print(
            "\tmean pruned model accuracy {:.2f}, mean complementary model accuracy {:.2f}; pruned_model_density: {:.4f}".format(
                results['acc_faith'], results['acc_other'], results['model_density']))
        print('\n')

        # Save a checkpoint
        if results['model_density'] <= target_density[point]:


            #do a model edit task here ------------------------------------------------!!






            masked_model.apply_masks()

            # Convert tensors to lists
            to_save_masks = {masked_model.param_names[i]: p.tolist() for i, p in masked_model.masks.items()}
            
            # Save the masks
            file_path = os.path.join('./output', f"{args['task_name']}_{target_density[point]}_masks.json")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(to_save_masks, f, indent=4)

            masks = {masked_model.param_names[i]:p for i,p in masked_model.masks.items()}
            if args['model_name']=='gpt2-xl':
                new_masks = {}
                for key, value in masks.items():
                    new_key = key.replace("gpt2.", "transformer.")
                    new_masks[new_key] = value
                masks = new_masks
            # masks = {masked_model.param_names[i]:(torch.rand_like(p) < args['target_density']).float() for i,p in masked_model.masks.items()}



            model_g = AutoModelForCausalLM.from_pretrained(args['model_name']).to(device)
            args['param_keys'] = get_target_module_keys(model_g)
            masked_generative_model = MaskedGenerativeModel(model_g, args, masks)
            # tokenizer_g = AutoTokenizer.from_pretrained(args['model_name'])

            location = args["data_dir_ft"]
            with open(location, "r") as f:
                data = json.load(f)

            prompts = []
            ground_truth = []
            target_new = []
            
            for datai in data:
                prompts.append(datai["requested_rewrite"]["prompt"].format(datai["requested_rewrite"]["subject"]))
                ground_truth.append(datai["requested_rewrite"]["target_true"]["str"])
                target_new.append(datai["requested_rewrite"]["target_new"]["str"])

            metrics = edit(
                prompts=prompts, 
                target_new=target_new, 
                ground_truth=ground_truth, 
                model=masked_generative_model, 
                tok=tokenizer,
                args=args, 
                device=device
            )

            if not os.path.exists('./output'):
                os.makedirs('./output')

            file_path = os.path.join('./output', f"{args['task_name']}_{target_density[point]}_editing_results.json")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(metrics, f, indent=4)


        #test the complementary model
            model_g_c = AutoModelForCausalLM.from_pretrained(args['model_name']).to(device)
            args['param_keys'] = get_target_module_keys(model_g_c)
            masked_generative_model_c = MaskedGenerativeModel(model_g_c, args, masks, complementary=True)

            metrics_c = edit(
                prompts=prompts, 
                target_new=target_new, 
                ground_truth=ground_truth, 
                model=masked_generative_model_c, 
                tok=tokenizer,
                args=args, 
                device=device
            )

            if not os.path.exists('./output'):
                os.makedirs('./output')

            file_path = os.path.join('./output', f"{args['task_name']}_{target_density[point]}_editing_results_complementary.json")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(metrics_c, f, indent=4)




            masked_model.remove_masks()
            #model edit task here end here ------------------------------------------------
            
            point = point+1

        if point == 5:
            break
        # if results['model_density'] <= args['target_density']:
        #     break
    # print("="*20)
    # iii = 0
    # for k,v in list(masked_model.named_parameters()):
    #     if iii == 6 or iii == 7:
    #         print(k)
    #         print(v)
    #     iii += 1
        



    # do a generation task








if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--args_fn', default='hparams_ra.yaml', type=str)
    with open(parser.parse_args().args_fn) as f:
        args = yaml.safe_load(f)
    main(args)
