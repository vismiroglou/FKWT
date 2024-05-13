from typing import Any

import lightning as L
import torch
from src.KWT import KWT, KWTFNet
from torch import nn, optim
from transformers import get_cosine_schedule_with_warmup
from torchmetrics.classification import MulticlassAccuracy


class LightningKWT(L.LightningModule):
    def __init__(self, config, useFnet=False):
        super().__init__()
        self.model = (
            KWT(**config["hparams"]["KWT"])
            if not useFnet
            else KWTFNet(**config["hparams"]["KWTFNet"])
        )
        self.config = config
        self.num_classes= (
            self.config['hparams']['KWT']['num_classes']
            if not useFnet
            else self.config['hparams']['KWTFNet']['num_classes']
        )
        self.train_precision = MulticlassAccuracy(num_classes=self.num_classes) #logging multiclass accuracy
        self.val_precision = MulticlassAccuracy(num_classes=self.num_classes) #logging multiclass accuracy
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, specs):
        return self.model(specs)

    def training_step(self, batch, batch_idx):
        specs, targets = batch
        outputs = self(specs)
        loss = self.criterion(outputs, targets)
        self.train_precision(outputs,targets)
        self.log("train_acc", self.train_precision, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log_dict({"train_loss": loss, "lr": self.optimizer.param_groups[0]["lr"]}, on_epoch=True, on_step=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        specs, targets = batch
        outputs = self(specs)
        val_loss = self.criterion(outputs, targets)
        self.val_precision(outputs, targets)
        self.log("val_acc",self.val_precision, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log_dict({"val_loss": val_loss}, on_epoch=True, on_step=True, sync_dist=True)
        return val_loss
   
    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        specs, targets = batch
        outputs = self(specs)
        test_loss = self.criterion(outputs, targets)
        self.val_precision(outputs, targets)
        self.log("test_acc",self.val_precision, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log_dict({"test_loss": test_loss}, on_epoch=True, on_step=True, sync_dist=True)
        return test_loss
    
    def test_epoch_end(self, outputs):
        avg_test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.log('avg_test_loss', avg_test_loss, sync_dist=True)
        
    def configure_optimizers(self):
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config["hparams"]["optimizer"]["lr"],
                           weight_decay=self.config["hparams"]["optimizer"]["weight_decay"])
        scheduler = get_cosine_schedule_with_warmup(self.optimizer, self.config["hparams"]["scheduler"]["n_warmup"], self.config["hparams"]["n_epochs"])
        return [self.optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
 
    
