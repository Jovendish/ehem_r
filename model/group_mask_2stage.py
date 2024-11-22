import math

import lightning as l
import torch
import torch.nn as nn
import yaml

from model.attention_model import SelfAttentionModule
from model.attention_model_local import SelfAttentionModule_local
from model.edge_conv import EdgeConv
from model.mlp_model import MLP

from utils.utils import AverageMeter, accuracy

cfg = yaml.safe_load(open("config.yaml", "r"))

class AncestralModel(l.LightningModule):
    def __init__(self, nlevel, ntoken, ninp, nhead, nhid, nout, nlayers, dropout):
        super().__init__()

        self.nout = nout

        self.total_loss = AverageMeter()
        self.total_acc1 = AverageMeter()
        self.total_acc5 = AverageMeter()

        self.attention_local_0 = SelfAttentionModule_local(ninp, nhead, nhid, dropout, nlayers)
        self.attention_local_1 = SelfAttentionModule_local(ninp, nhead, nhid, dropout, nlayers)

        self.attention_global_0 = SelfAttentionModule(ninp, nhead, nhid, dropout, nlayers)
        self.attention_global_1 = SelfAttentionModule(ninp, nhead, nhid, dropout, nlayers)

        self.encoder0 = nn.Embedding(ntoken + 1, 56)
        self.encoder1 = nn.Embedding(nlevel + 1, 4)
        self.encoder2 = nn.Embedding(8 + 1, 4)

        self.edge_conv = EdgeConv(16)

        self.linear_0 = nn.Linear(768, 552)

        self.decoder = MLP(ninp, nhid, nout, dropout)

        self.save_hyperparameters(logger=False)

    def forward(self, source):
        l, b, _, _ = source.size()

        occupy = source[:, :,  :, 0].long()
        level  = source[:, :,  :, 1].long()
        octant = source[:, :,  :, 2].long()
        pos    = source[:, :, -1, 3: ].float()

        # normalize position
        pos_min = torch.min(pos, dim=0, keepdim=True)[0]
        pos_max = torch.max(pos, dim=0, keepdim=True)[0]
        pos_norm = (pos - pos_min) / (pos_max - pos_min + 1e-7)

        # feature embedding
        occupy_parent = occupy.clone()
        level_parent  = level.clone()
        octant_parent = octant.clone()

        occupy_parent[: ,: ,  -1] = occupy_parent[: ,: , -2]
        level_parent[: ,: ,  -1]  = level_parent[: ,: , -2]
        octant_parent[: ,: ,  -1] = octant_parent[: ,: , -2]

        occupy_emb = self.encoder0(occupy)
        level_emb  = self.encoder1(level)
        octant_emb = self.encoder2(octant)

        # geometric feature extraction
        parent_pos_emb = self.edge_conv(occupy_parent, level_parent, octant_parent, pos_norm)

        emb = torch.cat((occupy_emb, level_emb, octant_emb), dim=-1).reshape(l, b, -1)
        emb = torch.cat((emb, parent_pos_emb), dim=-1)
        emb = self.linear_0(emb)

        # 4 Group
        # mask = torch.tril(torch.ones((4, 4))).bool().to(emb.device)
        # mask = mask.repeat(256,256)

        # 2 Group
        mask = torch.tril(torch.ones((2, 2))).bool().to(emb.device)
        mask = mask.repeat(512, 512)
        mask = ~mask

        # 4 Group
        # mask_local = torch.tril(torch.ones((4, 4))).bool().to(parent_emb.device)
        # mask_local = mask_local.repeat(64, 64)

        # 2 Group
        mask_local = torch.tril(torch.ones((2, 2))).bool().to(emb.device)
        mask_local = mask_local.repeat(128, 128)
        mask_local = ~mask_local

        output = self.attention_global_0(emb, mask=mask)  # (l, b, c)
        output = self.attention_local_0(output, mask=mask_local)

        output = self.attention_global_1(output, mask=mask)  # (l, b, c)
        output = self.attention_local_1(output, mask=mask_local)

        output = self.decoder(output)

        return output


    def process_batch_data(self, batch):
        batch = batch.permute(1, 0, 2, 3)
        target = batch[1:, :, -1, 0].clone()
        source = batch[1:, :, :, :].clone()

        batch[0::2, :, -1, 0] = batch[0::2, :, -2, 0]
        source[:, :, -1, 0] = batch[:-1, :, -1, 0]

        return source, target

    def training_step(self, batch, batch_idx):
        source, target = self.process_batch_data(batch)

        output = self.forward(source)

        output = output.reshape(-1, self.nout)
        target = target.reshape(-1).long()

        criterion = nn.CrossEntropyLoss(label_smoothing=0.)

        loss = criterion(output, target)
        loss = loss / math.log(2)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        self.total_loss.update(loss.item())
        self.total_acc1.update(acc1.item())
        self.total_acc5.update(acc5.item())

        if batch_idx % cfg["log_on_bar_interval"] == 0:
            self.log("loss", self.total_loss.avg, prog_bar=True)
            self.log("acc1", self.total_acc1.avg, prog_bar=True)
            self.log("acc5", self.total_acc5.avg, prog_bar=True)
            self.log("lr",   self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)

            self.total_loss.reset()
            self.total_acc1.reset()
            self.total_acc5.reset()

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=cfg["lr"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)

        return [optimizer], [scheduler]
