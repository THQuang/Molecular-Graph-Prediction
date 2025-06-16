import os
import time

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import DataLoader
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        option,
        model,
        train_dataset,
        valid_dataset,
        test_dataset=None,
    ):
        self.option = option
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"  # noqa: E501
        )
        self.model = model.to(self.device)
        
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=self.option["batch_size"],
            shuffle=True
        )
        self.valid_dataloader = DataLoader(
            valid_dataset, batch_size=self.option["batch_size"]
        )
        if test_dataset:
            self.test_dataloader = DataLoader(
                test_dataset, batch_size=self.option["batch_size"]
            )
        self.save_path = self.option["exp_path"]
        self.criterion = torch.nn.BCELoss()
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.option["lr"],
            weight_decay=option["weight_decay"],
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.7,
            patience=self.option["lr_scheduler_patience"],
            min_lr=1e-6,
        )

    def train_iterations(self):
        self.model.train()
        losses = []
        for data in tqdm(self.train_dataloader):
            self.optimizer.zero_grad()

            data = data.to(self.device)
            output = self.model(data)
            y_label = data.y
            loss = self.criterion(output.view_as(y_label), y_label.float())

            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        
        trn_loss = np.array(losses).mean()
        return trn_loss

    def valid_iterations(self, mode="valid"):
        self.model.eval()
        
        losses = []
        for data in tqdm(self.valid_dataloader):
            self.optimizer.zero_grad()

            data = data.to(self.device)
            output = self.model(data)
            y_label = data.y
            loss = self.criterion(output.view_as(y_label), y_label.float())
            losses.append(loss.item())
        val_loss = np.array(losses).mean()
        return val_loss

    def train(self):
        print("Training start...")
        best_val_loss = float('inf')
        
        for epoch in range(self.option["epochs"]):
            trn_loss = self.train_iterations()
            val_loss = self.valid_iterations()
            self.scheduler.step(val_loss)
            
            # Log metrics
            print(f"Epoch {epoch+1}/{self.option['epochs']}")
            print(f"Train Loss: {trn_loss:.4f}, Valid Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save model checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, os.path.join(self.save_path, 'best_model.pt'))
                print(f"Saved best model with validation loss: {val_loss:.4f}")
                
        print("Training completed!")

   