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

import os
import time
import yaml
import paddle
import paddle.nn as nn
from net import DSIN_layer
from dsin_reader import RecDataset
from paddle.io import DataLoader
import logging

if __name__ == "__main__":
    # set random seed
    paddle.seed(12345)

    # read config
    with open('config_bigdata.yaml', 'r') as f:
        config = yaml.load(f.read())
    train_batch_size = config['runner']['train_batch_size']
    test_batch_size = config['runner']['infer_batch_size']
    epochs = config['runner']['epochs']

    # read dataset
    train_data_dir = config['runner']['train_data_dir']
    test_data_dir = config['runner']['test_data_dir']
    dataloader_train = DataLoader(RecDataset(train_data_dir, mode='train'), batch_size=train_batch_size, shuffle=False)
    dataloader_test = DataLoader(RecDataset(test_data_dir, mode='test'), batch_size=test_batch_size, shuffle=False)

    # set logger config
    logging.basicConfig(filename='train&test.log', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('--------------common.configs--------------')
    logger.info(f"train_batch_size: {train_batch_size}, test_batch_size: {test_batch_size}, feature_embed_size: {config['hyper_parameters']['feat_embed_size']} ")
    logger.info('--------------common.configs--------------')

    # create model
    feat_size = config['hyper_parameters']
    model = DSIN_layer(user_size = feat_size['user_size'],adgroup_size = feat_size['adgroup_size'],pid_size = feat_size['pid_size'],
                        cms_segid_size = feat_size['cms_segid_size'],cms_group_size = feat_size['cms_group_size'],final_gender_size = feat_size['final_gender_size'],
                        age_level_size = feat_size['age_level_size'],pvalue_level_size = feat_size['pvalue_level_size'],shopping_level_size = feat_size['shopping_level_size'],
                        occupation_size = feat_size['occupation_size'],new_user_class_level_size = feat_size['new_user_class_level_size'],campaign_size = feat_size['campaign_size'],
                        customer_size = feat_size['customer_size'],cate_size = feat_size['cate_size'],brand_size = feat_size['brand_size'], l2_reg_embedding=1e-6)

    # set loss and opt
    model.train()
    criterion = nn.BCELoss()
    optimizer = paddle.optimizer.Adam(learning_rate=config['hyper_parameters']['optimizer']['learning_rate'], parameters=model.parameters())

    # training
    best_test_auc = 0
    for i in range(epochs):
        start_time = time.asctime()
        auc_metric = paddle.metric.Auc("ROC")
        for batch_id, datas in enumerate(dataloader_train):
            data, label = datas[0], datas[1]
            label = label.reshape([-1,1])
            output = model(data)
            pred_2d = paddle.concat(x=[1-output, output], axis=1)
            auc_metric.update(preds=pred_2d.numpy(), labels=label.numpy())
            loss = criterion(output,label)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            if batch_id %20 == 0:
                logger.info(f"epoch:{i}, batch_id:{batch_id}, log_loss:{loss.numpy()}, train_auc: {auc_metric.accumulate()}")
            if batch_id %50 == 0 and auc_metric.accumulate()>=0.6:
                model.eval()
                test_metric = paddle.metric.Auc('ROC')
                for test_batch_id, test_datas in enumerate(dataloader_test):
                    data, label = test_datas[0], test_datas[1]
                    label = label.reshape([-1,1])
                    output = model(data)
                    pred_2d = paddle.concat(x=[1-output, output], axis=1)
                    test_metric.update(preds=pred_2d.numpy(), labels=label.numpy())
                test_auc = test_metric.accumulate()
                best_test_auc = test_auc if test_auc>best_test_auc else best_test_auc
                logger.info("------------test stage------------")
                logger.info(f"epoch:{i}, batch_id:{batch_id}, test_auc: {test_auc}")
                logger.info(f"best_test_auc:{best_test_auc}")
                logger.info("------------test stage------------")
                model.train()
                if best_test_auc >=0.6375:
                    break
        if best_test_auc >=0.63:
            break
    print(f'After {i} epochs, best_test_auc: {best_test_auc}, log_loss:{loss}, train_auc:{auc_metric.accumulate()}')
    print(f'training start at {start_time} finish at {time.asctime()}')
