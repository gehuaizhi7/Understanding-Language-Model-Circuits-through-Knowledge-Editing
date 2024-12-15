# Based on the differentiable mask repo (Jade, Frank, Zining)

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


# def gumbel_sigmoid(logits, gs_temp=1., eps=1e-10):
#     uniform = logits.new_empty([2]+list(logits.shape)).uniform_(0, 1)
#     noise = -((uniform[1] + eps).log() / (uniform[0] + eps).log() + eps).log()
#     res = torch.sigmoid((logits + noise) / gs_temp)
#     res = ((res > 0.5).type_as(res) - res).detach() + res
#     return res


class MaskedGenerativeModel(nn.Module):
    def __init__(self, model, args, masks, complementary = False):
        super().__init__()
        self.model = model
        self.mask_param_names = args['param_keys']


        # self.mask_logits = {}
        # self.masks = {}
        # self.unmasked_params = {}  # a copy of the unmasked params
        # # self.is_masked = False  # no masks applied upon initialization
        # self.gs_temp = args['temperature']

        # self.param_names = [n for n, _ in list(
        #     model.named_parameters())]  # full list of param names of the transformer (some may not be masked)
        # self.model_param_list = [p for _, p in list(
        #     model.named_parameters())]  # full list of params of the transformer (some may not be masked)
        # self.param_name2param_id = {n: i for i, n in enumerate(self.param_names)}


        # for param_name in self.mask_param_names:
        #     param_id = self.param_name2param_id[param_name]
        #     m = self.model_param_list[param_id]
        #     self.unmasked_params[param_id] = m.detach().clone()

        #     masks_logits = nn.Parameter(mask_logits_dict[param_id], requires_grad=False)
        #     self.mask_logits[param_id] = masks_logits
        #     self.masks[param_id] = gumbel_sigmoid(self.mask_logits[param_id], gs_temp=self.gs_temp)

        #     print(param_name)
        #     print(torch.eq(self.masks[param_id], 0).sum().item())

        #     m.register_hook(lambda grad, mask=self.masks[param_id]: grad * mask)
            
        #     m.data = self.masks[param_id] * self.unmasked_params[param_id]
        #     # print(m)


        self.masks = copy.deepcopy(masks)
        if complementary:
            for n, p in self.masks.items():
                # print("before inversion:")
                # print(p)
                p[:] = 1 - p
                # print("after inversion:")
                # print(p)
            # print('masks:')
            # print(self.masks)
            # print('mask_param_names:')
            # print(self.mask_param_names)
        for n, p in list(model.named_parameters()):
            if n in self.mask_param_names:
                mask = self.masks[n]
                # # p.register_hook(lambda grad, mask=mask: grad * mask)
                p.data = mask * p.data
                # print(n)
                # print(torch.eq(p, 0).sum().item())

                
            

    def forward(self, **inputs):
        return self.model.forward(**inputs)

    def generate(self, inputs, max_length=20):
        return self.model.generate(inputs, max_length=max_length)
    
    # def get_unmasked_parameters(self):
    #     return (logits for _, logits in self.unmasked_params.items())



    # def save_mask_logits(self, fn):
    #     mask_logits = {k: v.detach().cpu() for k, v in self.mask_logits.items()}
    #     torch.save(mask_logits, fn)