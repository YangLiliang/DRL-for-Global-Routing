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
tf.compat.v1.disable_eager_execution()
N_workers=multiprocessing.cpu_count()    #并行数量

np.random.seed(10701)
tf.set_random_seed(10701)
random.seed(10701)


class QNetwork():

    # This class essentially defines the network architecture.
    # The network should take in state of the world as an input,
    # and output Q values of the actions available to the agent as the output.

    def __init__(self, environment_name, networkname):
        # Define your network architecture here. It is also a good idea to define any training operations
        # and optimizers here, initialize your variables, or alternately compile your model here.
        if environment_name == 'grid':
            self.nObservation = 12
            self.nAction = 6
            self.learning_rate = 0.0001
            self.architecture = [32, 64, 32]

        kernel_init = tf.random_uniform_initializer(-0.5, 0.5)
        bias_init = tf.constant_initializer(0)
        self.input = tf.placeholder(tf.float32, shape=[None, self.nObservation], name='input')
        if networkname=='target_netWork':
            with tf.variable_scope(networkname):
                layer1 = tf.layers.dense(self.input, self.architecture[0], tf.nn.relu, kernel_initializer=kernel_init,
                                     bias_initializer=bias_init, name='layer1')
                layer2 = tf.layers.dense(layer1, self.architecture[1], tf.nn.relu, kernel_initializer=kernel_init,
                                     bias_initializer=bias_init, name='layer2')
                layer3 = tf.layers.dense(layer2, self.architecture[2], tf.nn.relu, kernel_initializer=kernel_init,
                                     bias_initializer=bias_init, name='layer3')
                self.output = tf.layers.dense(layer3, self.nAction, kernel_initializer=kernel_init,
                                          bias_initializer=bias_init, name='output')
                self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=networkname)
        else:
            with tf.variable_scope(networkname):
                layer1 = tf.layers.dense(self.input, self.architecture[0], tf.nn.relu, kernel_initializer=kernel_init,
                                     bias_initializer=bias_init, name='layer1')
                layer2 = tf.layers.dense(layer1, self.architecture[1], tf.nn.relu, kernel_initializer=kernel_init,
                                     bias_initializer=bias_init, name='layer2')
                layer3 = tf.layers.dense(layer2, self.architecture[2], tf.nn.relu, kernel_initializer=kernel_init,
                                     bias_initializer=bias_init, name='layer3')
                self.output = tf.layers.dense(layer3, self.nAction, kernel_initializer=kernel_init,
                                          bias_initializer=bias_init, name='output')
                self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=networkname)

                self.targetQ = tf.placeholder(tf.float32, shape=[None, self.nAction], name='target')
                self.loss = tf.losses.mean_squared_error(self.targetQ, self.output)
                self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

                with tf.name_scope('local_grad'):
                    self.grads=tf.gradients(self.loss,self.params)   #实现loss对params每一个参数的求导，返回一个list


        with tf.variable_scope(networkname, reuse=True):
            self.w1 = tf.get_variable('layer1/kernel')
            self.b1 = tf.get_variable('layer1/bias')
            self.w2 = tf.get_variable('layer2/kernel')
            self.b2 = tf.get_variable('layer2/bias')
            self.w3 = tf.get_variable('layer3/kernel')
            self.b3 = tf.get_variable('layer3/bias')
            self.w4 = tf.get_variable('output/kernel')
            self.b4 = tf.get_variable('output/bias')



class Replay_Memory():

    def __init__(self, memory_size=50000, burn_in=10000):
        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
        # A simple (if not the most efficient) was to implement the memory is as a list of transitions.
        self.memory = []
        self.is_burn_in = False
        self.memory_max = memory_size
        self.burn_in = burn_in

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
        # You will feed this to your model to train.
        index = np.random.randint(len(self.memory), size=batch_size)
        batch = [self.memory[i] for i in index]
        return batch

    def append(self, transition):
        # Appends transition to the memory.
        self.memory.append(transition)
        if len(self.memory) > self.memory_max:
            self.memory.pop(0)


class DQN_Agent():

    # In this class, we will implement functions to do the following.
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
    #		(a) Epsilon Greedy Policy.
    # 		(b) Greedy Policy.
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.

    def __init__(self,workerName ,environment_name, sess, gridgraph, render=False):

        # Create an instance of the network itself, as well as the memory.
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc.
        self.epsilon = 0.05
        self.workerName=workerName

        if environment_name == 'grid':
            self.gamma = 0.95
        self.max_episodes = 100  # 20000 #200
        self.batch_size = 32
        self.render = render

        self.netWork = QNetwork(environment_name,workerName)

        self.replay = Replay_Memory()

        self.gridgraph = gridgraph

        self.as_w1 = tf.assign(targetNetwork.w1, 0.95*self.netWork.w1+0.05*targetNetwork.w1)
        self.as_b1 = tf.assign(targetNetwork.b1, 0.95*self.netWork.b1+0.05*targetNetwork.b1)
        self.as_w2 = tf.assign(targetNetwork.w2, 0.95*self.netWork.w2+0.05*targetNetwork.w2)
        self.as_b2 = tf.assign(targetNetwork.b2, 0.95*self.netWork.b2+0.05*targetNetwork.b2)
        self.as_w3 = tf.assign(targetNetwork.w3, 0.95*self.netWork.w3+0.05*targetNetwork.w3)
        self.as_b3 = tf.assign(targetNetwork.b3, 0.95*self.netWork.b3+0.05*targetNetwork.b3)
        self.as_w4 = tf.assign(targetNetwork.w4, 0.95*self.netWork.w4+0.05*targetNetwork.w4)
        self.as_b4 = tf.assign(targetNetwork.b4, 0.95*self.netWork.b4+0.05*targetNetwork.b4)

        self.init = tf.global_variables_initializer()

        self.sess = sess
        # tf.summary.FileWriter("logs/", self.sess.graph)
        self.sess.run(self.init)
        self.saver = tf.train.Saver(max_to_keep=20)

    def epsilon_greedy_policy(self, q_values):
        # Creating epsilon greedy probabilities to sample from.
        rnd = np.random.rand()
        if rnd <= self.epsilon:
            return np.random.randint(len(q_values))
        else:
            return np.argmax(q_values)

    def greedy_policy(self, q_values):
        # Creating greedy policy for test time.
        return np.argmax(q_values)

    def network_assign(self):
        # pass the weights of evaluation network to target network
        self.sess.run([self.as_w1, self.as_b1, self.as_w2, self.as_b2, self.as_w3, self.as_b3, self.as_w4, self.as_b4])

    def train(self, savepath, model_file=None):
        # ! savepath: "../model_(train/test)"
        # ! if model_file = None, training; if given, testing
        # ! if testing using training function, comment burn_in in Router.py

        # In this function, we will train our network.
        # If training without experience replay_memory, then you will interact with the environment
        # in this function, while also updating your network parameters.

        # If you are using a replay memory, you should interact with environment here, and store these
        # transitions to memory, while also updating your model.

        # the model will be saved to ../model/
        # the training/testing curve will be saved as a .npz file in ../data/

        twoPinNum=self.gridgraph.twopinNum
        twoPinNumEachNet=self.gridgraph.net_pair
        netSort=self.gridgraph.netOrder
        # print(netSort)
        # print(twoPinNumEachNet)
        # print(self.gridgraph.twopin_combo)

        if model_file is not None:
            self.saver.restore(self.sess, model_file)

        reward_log = []
        test_reward_log = []
        test_episode = []
        # if not self.replay.is_burn_in:
        # 	self.burn_in_memory()
        solution_combo = []

        reward_plot_combo = []
        reward_plot_combo_pure = []

        episode=0
        while not COORD.should_stop() and episode<self.max_episodes:
            for loop in range(len(self.gridgraph.twopin_combo)):

                solution_combo.append(self.gridgraph.route)

                state, reward_plot = self.gridgraph.reset()
                reward_plot_pure = reward_plot - self.gridgraph.posTwoPinNum * 100
                # print('reward_plot-self.gridgraph.posTwoPinNum*100',reward_plot-self.gridgraph.posTwoPinNum*100)

                if (episode*loop) % twoPinNum == 0 and episode!=0:
                    reward_plot_combo.append(reward_plot)
                    reward_plot_combo_pure.append(reward_plot_pure)
                is_terminal = False
                rewardi = 0.0
                if episode*loop % 100 == 0:
                    self.network_assign()

                rewardfortwopin = 0
                while not is_terminal:
                    observation = self.gridgraph.state2obsv()
                    q_values = self.sess.run(self.netWork.output, feed_dict={self.netWork.input: observation})
                    action = self.epsilon_greedy_policy(q_values)
                    # print(action)
                    nextstate, reward, is_terminal, debug = self.gridgraph.step(action)
                    # print(nextstate)
                    observation_next = self.gridgraph.state2obsv()
                    self.replay.append([observation, action, reward, observation_next, is_terminal])
                    state = nextstate
                    rewardi = rewardi + reward
                    rewardfortwopin = rewardfortwopin + reward

                    batch = self.replay.sample_batch(self.batch_size)
                    batch_observation = np.squeeze(np.array([trans[0] for trans in batch]))
                    batch_action = np.array([trans[1] for trans in batch])
                    batch_reward = np.array([trans[2] for trans in batch])
                    batch_observation_next = np.squeeze(np.array([trans[3] for trans in batch]))
                    batch_is_terminal = np.array([trans[4] for trans in batch])
                    q_batch = self.sess.run(self.netWork.output, feed_dict={self.netWork.input: batch_observation})
                    q_batch_next = self.sess.run(targetNetwork.output,
                                             feed_dict={targetNetwork.input: batch_observation_next})
                    y_batch = batch_reward + self.gamma * (1 - batch_is_terminal) * np.max(q_batch_next, axis=1)

                    targetQ = q_batch.copy()
                    targetQ[np.arange(self.batch_size), batch_action] = y_batch
                    _, train_error = self.sess.run([self.netWork.opt, self.netWork.loss],
                                               feed_dict={self.netWork.input: batch_observation,self.netWork.targetQ: targetQ})
                reward_log.append(rewardi)  # comment in test; do not save model test

                self.gridgraph.instantrewardcombo.append(rewardfortwopin)
                if self.gridgraph.clearCapacityFlag==1:
                    self.gridgraph.passby=np.zeros_like(self.gridgraph.capacity)
                    self.gridgraph.clearCapacityFlag=0

            episode += 1

        score = self.gridgraph.best_reward
        solution = self.gridgraph.best_route
        capacity=self.gridgraph.bestCapacity

        solutionDRL=[]
        for i in range(len(netSort)):
            solutionDRL.append([])


        print('twoPinNum:', twoPinNum)

        Dump=0
        for i in range(len(netSort)):
            netToDump = netSort[i]
            for j in range(twoPinNumEachNet[i]):
                solutionDRL[netToDump].append(solution[Dump])
                Dump+=1


        tf.reset_default_graph()
        print(netSort)

        return solutionDRL, reward_plot_combo, reward_plot_combo_pure, solution, self.gridgraph.posTwoPinNum,score,capacity

    def test(self, model_file=None, no=20, stat=False):
        # Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using a memory.

        # uncomment this line below for videos
        # self.env = gym.wrappers.Monitor(self.env, "recordings", video_callable=lambda episode_id: True)
        if model_file is not None:
            self.saver.restore(self.sess, model_file)
        reward_list = []
        cum_reward = 0.0
        for episode in np.arange(no):
            episode_reward = 0.0
            state = self.gridgraph.reset()
            is_terminal = False
            while not is_terminal:
                observation = self.gridgraph.state2obsv()
                q_values = self.sess.run(self.netWork.output, feed_dict={self.netWork.input: observation})
                action = self.greedy_policy(q_values)
                nextstate, reward, is_terminal, debug = self.gridgraph.step(action)
                state = nextstate
                episode_reward = episode_reward + reward
                cum_reward = cum_reward + reward
            reward_list.append(episode_reward)
        if stat:
            return cum_reward, reward_list
        else:
            return cum_reward

    def burn_in_memory_search(self, observationCombo, actionCombo, rewardCombo,
                              observation_nextCombo, is_terminalCombo):  # Burn-in with search
        print('Start burn in with A* search algorithm...')
        for i in range(len(observationCombo)):
            observation = observationCombo[i]
            action = actionCombo[i]
            reward = rewardCombo[i]
            observation_next = observation_nextCombo[i]
            is_terminal = is_terminalCombo[i]

            self.replay.append([observation, action, reward, observation_next, is_terminal])

        self.replay.is_burn_in = True
        print('Burn in with A* search algorithm finished.')



if __name__ == '__main__':

    # 初始化环境部分
    environment_name = 'grid'
    filename = 'test_benchmark_1.gr'
    grid_info = init.read(filename)
    gridParameters = init.gridParameters(grid_info)

    # Setting the session to allow growth, so it doesn't allocate all GPU memory.
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)

    # Setting this as the default tensorflow session.
    keras.backend.tensorflow_backend.set_session(sess)

    # You want to create an instance of the DQN_Agent class here, and then train / test it.
    model_path = './model/'
    data_path = './data/'
    solutionPicture_path='./solutionPicture/'
    solutionResult_path='./solutionResult/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if not os.path.exists(solutionPicture_path):
        os.makedirs(solutionPicture_path)
    if not os.path.exists(solutionResult_path):
        os.makedirs(solutionResult_path)

    ## 生成初始经验
    routeListMerged, routeListNotMerged = AStarSearch.getAStarRoute(gridParameters)
    observationCombo, actionCombo, rewardCombo, observation_nextCombo, is_terminalCombo=AStarSearch.getBrunInInform(routeListMerged,gridParameters)

    with tf.device('/cpu:0'):
        ## 初始全局目标网络
        targetNetwork = QNetwork(environment_name,'target_netWork')
        ## 初始化智能体
        agents=[]
        for i in range(N_workers):  # N—workers等于cpu数量
            i_name = 'agent_%i' % (i+1)  # worker name
            gridgraph = env.GridGraph(gridParameters,i_name)
            agent=DQN_Agent(workerName=i_name, environment_name='grid',sess=sess,gridgraph=gridgraph,render = False)
            ## 初始化重播缓冲区
            agent.burn_in_memory_search(observationCombo, actionCombo, rewardCombo, observation_nextCombo, is_terminalCombo)
            agents.append(agent)  # 创建独立的agent

        COORD = tf.train.Coordinator()  # 多线程
        agnet_threads = []
        for agent in agents:  # 并行过程
            job = lambda: agent.train(savepath=None, model_file=None)  # worker的工作目标,此处调用Worker类中的work
            t = threading.Thread(target=job)  # 每一个线程完成一个worker的工作目标
            t.start()  # 启动每一个worker
            agnet_threads.append(t)  # 每一个worker的工作都加入thread中
        COORD.join(agnet_threads)  # 合并几个worker,当每一个worker都运行完再继续后面步骤






    sess.close()

