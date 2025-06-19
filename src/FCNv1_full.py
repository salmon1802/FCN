# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import torch
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding


class FCNv1_full(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="FCNv1_full",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=16,
                 exp_num_layers=3,
                 lin_num_layers=3,
                 net_dropout=0,
                 batch_norm=False,
                 layer_norm=False,
                 num_heads=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(FCNv1_full, self).__init__(feature_map,
                                    model_id=model_id,
                                    gpu=gpu,
                                    embedding_regularizer=embedding_regularizer,
                                    net_regularizer=net_regularizer,
                                    **kwargs)
        self.embedding_layer = MultiHeadFeatureEmbedding(feature_map, embedding_dim * num_heads, num_heads)
        input_dim = feature_map.sum_emb_out_dim()
        self.ECN = ExponentialCrossNetwork(input_dim=input_dim,
                                           num_layers=exp_num_layers,
                                           net_dropout=net_dropout,
                                           batch_norm=batch_norm,
                                           layer_norm=layer_norm,
                                           num_heads=num_heads)
        self.LCN = LinearCrossNetwork(input_dim=input_dim,
                                      num_layers=lin_num_layers,
                                      net_dropout=net_dropout,
                                      batch_norm=batch_norm,
                                      layer_norm=layer_norm,
                                      num_heads=num_heads)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)  # B × H × FD/H
        ecn_logit = self.ECN(feature_emb).mean(dim=1)
        lcn_logit = self.LCN(feature_emb).mean(dim=1)
        logit = (ecn_logit + lcn_logit) * 0.5
        y_pred = self.output_activation(logit)
        return_dict = {"y_pred": y_pred,
                       "y_d": self.output_activation(ecn_logit),
                       "y_s": self.output_activation(lcn_logit)}
        return return_dict

    def add_loss(self, inputs):
        return_dict = self.forward(inputs)
        y_true = self.get_labels(inputs)
        y_pred = return_dict["y_pred"]
        y_d = return_dict["y_d"]
        y_s = return_dict["y_s"]
        loss = self.loss_fn(y_pred, y_true, reduction='mean')
        loss_d = self.loss_fn(y_d, y_true, reduction='mean')
        loss_s = self.loss_fn(y_s, y_true, reduction='mean')
        weight_d = loss_d - loss
        weight_s = loss_s - loss
        weight_d = torch.where(weight_d > 0, weight_d, torch.zeros(1).to(weight_d.device))
        weight_s = torch.where(weight_s > 0, weight_s, torch.zeros(1).to(weight_s.device))
        loss = loss + loss_d * weight_d + loss_s * weight_s
        return loss


class MultiHeadFeatureEmbedding(nn.Module):
    def __init__(self, feature_map, embedding_dim, num_heads=2):
        super(MultiHeadFeatureEmbedding, self).__init__()
        self.num_heads = num_heads
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)

    def forward(self, X):  # H = num_heads
        feature_emb = self.embedding_layer(X)  # B × F × D
        multihead_feature_emb = torch.tensor_split(feature_emb, self.num_heads, dim=-1)
        multihead_feature_emb = torch.stack(multihead_feature_emb, dim=1)  # B × H × F × D/H
        multihead_feature_emb1, multihead_feature_emb2 = torch.tensor_split(multihead_feature_emb, 2,
                                                                            dim=-1)  # B × H × F × D/2H
        multihead_feature_emb1, multihead_feature_emb2 = multihead_feature_emb1.flatten(start_dim=2), \
                                                         multihead_feature_emb2.flatten(
                                                             start_dim=2)  # B × H × FD/2H; B × H × FD/2H
        multihead_feature_emb = torch.cat([multihead_feature_emb1, multihead_feature_emb2], dim=-1)
        return multihead_feature_emb  # B × H × FD/H


class ExponentialCrossNetwork(nn.Module):
    def __init__(self,
                 input_dim,
                 num_layers=3,
                 batch_norm=True,
                 layer_norm=False,
                 net_dropout=0.1,
                 num_heads=1):
        super(ExponentialCrossNetwork, self).__init__()
        self.num_layers = num_layers
        self.layer_norm = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.w = nn.ModuleList()
        self.b = nn.ParameterList()
        self.gamma = nn.ParameterList()
        self.beta = nn.ParameterList()
        for i in range(num_layers):
            self.w.append(nn.Linear(input_dim, input_dim, bias=False))
            self.b.append(nn.Parameter(torch.empty((input_dim,))))
            if batch_norm:
                self.batch_norm.append(nn.BatchNorm1d(num_heads))
            if net_dropout > 0:
                self.dropout.append(nn.Dropout(net_dropout))
            nn.init.uniform_(self.b[i].data)
        self.fc = nn.Linear(input_dim, 1)

    def _cross_layer(self, x, x_anchor, i):
        H = self.w[i](x)
        if len(self.batch_norm) > i:
            H = self.batch_norm[i](H)
        x = x_anchor * (H + self.b[i]) + x
        if len(self.dropout) > i:
            x = self.dropout[i](x)
        return x

    def forward(self, x):
        for i in range(self.num_layers):
            x = self._cross_layer(x, x, i)
        logit = self.fc(x)
        return logit


class LinearCrossNetwork(nn.Module):
    def __init__(self,
                 input_dim,
                 num_layers=3,
                 batch_norm=True,
                 layer_norm=False,
                 net_dropout=0.1,
                 num_heads=1):
        super(LinearCrossNetwork, self).__init__()
        self.num_layers = num_layers
        self.layer_norm = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.w = nn.ModuleList()
        self.b = nn.ParameterList()
        self.gamma = nn.ParameterList()
        self.beta = nn.ParameterList()
        for i in range(num_layers):
            self.w.append(nn.Linear(input_dim, input_dim, bias=False))
            self.b.append(nn.Parameter(torch.empty((input_dim,))))
            if batch_norm:
                self.batch_norm.append(nn.BatchNorm1d(num_heads))
            if net_dropout > 0:
                self.dropout.append(nn.Dropout(net_dropout))
            nn.init.uniform_(self.b[i].data)
        self.fc = nn.Linear(input_dim, 1)

    def _cross_layer(self, x, x_anchor, i):
        H = self.w[i](x)
        if len(self.batch_norm) > i:
            H = self.batch_norm[i](H)
        x = x_anchor * (H + self.b[i]) + x
        if len(self.dropout) > i:
            x = self.dropout[i](x)
        return x

    def forward(self, x):
        x_a = x
        for i in range(self.num_layers):
            x = self._cross_layer(x, x_a, i)
        logit = self.fc(x)
        return logit
