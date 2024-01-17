import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class LanguageModel(nn.Module):
    def __init__(self, lmodel, embedding_dim, num_classes, dropout=0.15):
        super(LanguageModel, self).__init__()
        self.lmodel = lmodel  # language model
        self.p = dropout
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(embedding_dim, num_classes)  # prediction head 

    def forward(self, batch):
        ids, masks = batch["ids"], batch["masks"]
        _, pool = self.lmodel(input_ids=ids, attention_mask=masks)  # use pooled embedding
        z = self.dropout(pool)
        z = self.head(z)
        return z

    @torch.inference_mode()
    def predict(self, batch):
        self.eval()
        z = self(batch)
        y_pred = torch.argmax(z, dim=1).cpu().numpy()
        return y_pred

    @torch.inference_mode()
    def predict_proba(self, batch):
        self.eval()
        z = self(batch)
        y_probs = F.softmax(z, dim=1).cpu().numpy()
        return y_probs

    def save(self, dp):
        with open(Path(dp, "args.json"), "w") as fp:
            contents = {
                "dropout": self.p,
                "embedding_dim": self.embedding_dim,
                "num_classes": self.num_classes,
            }
            json.dump(contents, fp, indent=4, sort_keys=False)
        torch.save(self.state_dict(), os.path.join(dp, "model.pt"))

    @classmethod
    def load(cls, args_fp, state_dict_fp):
        with open(args_fp, "r") as fp:
            kwargs = json.load(fp=fp)
        bert = BertModel.from_pretrained("allenai/scibert_scivocab_uncased", return_dict=False)
        model = cls(lmodel=bert, **kwargs)
        model.load_state_dict(torch.load(state_dict_fp, map_location=torch.device("cpu")))
        return model