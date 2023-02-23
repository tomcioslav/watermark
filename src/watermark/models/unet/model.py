import pytorch_lightning as pl
import torch
import torch.nn as nn
from watermark.models.unet.model_parts import DoubleConv, Down, OutConv, Up


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.inc = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, 3)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class WatermarkNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.unet = UNet()

    def training_step(self, batch):
        input, target = batch
        output = self.unet(input)
        loss = nn.MSELoss(reduction="mean")(output, target)
        self.log("train_loss", loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input, target = batch
        output = self.unet(input)
        loss = nn.MSELoss(reduction="mean")(output, target)
        self.log("validation_loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        return optimizer
