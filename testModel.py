import DQN as dqn

# !/usr/bin/env python
import keras, numpy as np, sys, copy, argparse, random
import matplotlib.pyplot as plt
import multiprocessing  #多线程模块
import threading  #线程模块
import tensorflow.compat.v1 as tf
import Initializer as init
import TwoPinAStarSearch as AStarSearch
import math
import os
import GridGraph as env
import TestDRLSolution as test
from tensorflow.python.client import device_lib
import os

tf.compat.v1.disable_eager_execution()

np.random.seed(10701)
tf.set_random_seed(10701)
random.seed(10701)

if __name__ == '__main__':
    # 初始化环境部分
    environment_name = 'grid'
    filename = 'test_benchmark_1.1.gr'
    grid_info = init.read(filename)
    gridParameters = init.gridParameters(grid_info)
    gridgraph=env.GridGraph(gridParameters)

    # Setting the session to allow growth, so it doesn't allocate all GPU memory.
    #gpu_ops = tf.GPUOptions(allow_growth=True)
    #config = tf.ConfigProto(gpu_options=gpu_ops)
    #sess = tf.Session(config=config)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    config = tf.ConfigProto(gpu_options=gpu_options)
    sess = tf.Session(config=config)


    ## 初始化智能体
    agent = dqn.DQN_Agent(environment_name, sess, gridgraph, render=False)

    # 测试
    model_path = './doubleModel/'
    model_file = model_path + 'my_net.ckpt'
    agent.test(model_file)

    sess.close()