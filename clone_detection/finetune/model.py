import torch
import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

    def forward(self, input_ids=None, labels=None):
        input_ids = input_ids.view(-1, self.args.block_size)
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]
        outputs = (outputs * input_ids.ne(1)[:, :, None]).sum(1) / input_ids.ne(1).sum(1)[:, None]
        outputs = outputs.reshape(-1, 2, outputs.size(-1))
        outputs = torch.nn.functional.normalize(outputs, p=2, dim=-1)
        cos_sim = (outputs[:, 0] * outputs[:, 1]).sum(-1)

        if labels is not None:
            loss = ((cos_sim - labels.float()) ** 2).mean()
            return loss, cos_sim
        else:
            return cos_sim
