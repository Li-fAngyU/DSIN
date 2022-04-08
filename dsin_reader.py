# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function
import numpy as np

from paddle.io import IterableDataset
import pandas as pd

sparse_features = ['userid', 'adgroup_id', 'pid', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level',
                    'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level ', 'campaign_id',
                    'customer', 'cate_id', 'brand']

dense_features = ['price']

class RecDataset(IterableDataset):
    def __init__(self, file_list, config):
        super().__init__()
        self.file_list = file_list
        feat_input = pd.read_pickle(self.file_list[0])
        self.sess_input = pd.read_pickle(self.file_list[3])
        self.sess_length = pd.read_pickle(self.file_list[2])
        self.label = pd.read_pickle(self.file_list[1]).to_numpy().astype('float32')
        self.num_samples = self.label.shape[0]
        self.sparse_input = feat_input[sparse_features].to_numpy().astype('int64')
        self.dense_input = feat_input[dense_features].to_numpy().reshape(-1)

    def __iter__(self):
        for i in range(self.num_samples):
            yield (self.sparse_input[i, :], self.dense_input[i], self.sess_input[i, :, :], self.sess_length[i]), self.label[i]
