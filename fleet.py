from __future__ import print_function
from argparse import parse_args
import os
import sys

import paddle
import paddle.distributed.fleet.base.role_maker as role_maker
import paddle.distributed.fleet as fleet

from network_conf import ctr_dnn_model_dataset

dense_feature_dim = 13
sparse_feature_dim = 10000001

batch_size = 100
thread_num = 10
embedding_size = 10

args = parse_args()

def main_function(is_local):

    # common code for local training and distributed training
    dense_input = paddle.static.data(
      name="dense_input", shape=[dense_feature_dim], dtype='float32')
    sparse_input_ids = [
          paddle.static.data(name="C" + str(i), shape=[1], lod_level=1,
                            dtype="int64") for i in range(1, 27)]

    label = paddle.static.data(name="label", shape=[1], dtype="int64")

    dataset = paddle.distributed.QueueDataset()
    dataset.init(
          batch_size=batch_size,
          thread_num=thread_num,
          input_type=0,
          pipe_command="python criteo_reader.py %d" % sparse_feature_dim,
          use_var=[dense_input] + sparse_input_ids + [label])

    whole_filelist = ["raw_data/part-%d" % x
                       for x in range(len(os.listdir("raw_data")))]
    dataset.set_filelist(whole_filelist)

    loss, auc_var, batch_auc_var = ctr_dnn_model_dataset(
        dense_input, sparse_input_ids, label, embedding_size,
        sparse_feature_dim)

    exe = paddle.static.Executor(paddle.CPUPlace())

    def train_loop(epoch=20):
        for i in range(epoch):
            exe.train_from_dataset(program=paddle.static.default_main_program(),
                                   dataset=dataset,
                                   fetch_list=[auc_var],
                                   fetch_info=["auc"],
                                   debug=False)

    # local training
    def local_train():
        optimizer = paddle.optimizer.SGD(learning_rate=1e-4)
        optimizer.minimize(loss)
        exe.run(paddle.static.default_startup_program())
        train_loop()

  # distributed training
    def dist_train():
        role = role_maker.PaddleCloudRoleMaker()
        fleet.init(role)

        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.a_sync = True
        optimizer = paddle.optimizer.SGD(learning_rate=1e-4)
        optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(loss)

        if fleet.is_server():
            fleet.init_server()
            fleet.run_server()

        elif fleet.is_worker():
            fleet.init_worker()
            exe.run(paddle.static.default_startup_program())
            train_loop()

    if is_local:
        local_train()
    else:
        dist_train()

if __name__ == '__main__':
    main_function(args.is_local)
