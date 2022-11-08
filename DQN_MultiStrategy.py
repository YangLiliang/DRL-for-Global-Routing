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

np.random.seed(10701)
tf.set_random_seed(10701)
random.seed(10701)


class QNetwork():

    # This class essentially defines the network architecture.
    # The network should take in state of the world as an input,
    # and output Q values of the actions available to the agent as the output.

    def __init__(self, environment_name, networkname, trianable):
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
        with tf.variable_scope(networkname):
            layer1 = tf.layers.dense(self.input, self.architecture[0], tf.nn.relu, kernel_initializer=kernel_init,
                                     bias_initializer=bias_init, name='layer1', trainable=trianable)
            layer2 = tf.layers.dense(layer1, self.architecture[1], tf.nn.relu, kernel_initializer=kernel_init,
                                     bias_initializer=bias_init, name='layer2', trainable=trianable)
            layer3 = tf.layers.dense(layer2, self.architecture[2], tf.nn.relu, kernel_initializer=kernel_init,
                                     bias_initializer=bias_init, name='layer3', trainable=trianable)
            output1=tf.layers.dense(layer3, self.nAction, kernel_initializer=kernel_init,
                                          bias_initializer=bias_init, name='output1', trainable=trianable)
            output2=tf.layers.dense(layer3, 1, kernel_initializer=kernel_init,
                                          bias_initializer=bias_init, name='output2', trainable=trianable)
            self.output = output2+(output1-tf.reduce_mean(output1, axis=1, keep_dims=True))
            # self.output = tf.layers.dense(layer3, self.nAction, kernel_initializer=kernel_init,
            #                               bias_initializer=bias_init, name='output', trainable=trianable)

        self.targetQ = tf.placeholder(tf.float32, shape=[None, self.nAction], name='target')
        if trianable == True:
            self.loss = tf.losses.mean_squared_error(self.targetQ, self.output)
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        with tf.variable_scope(networkname, reuse=True):
            self.w1 = tf.get_variable('layer1/kernel')
            self.b1 = tf.get_variable('layer1/bias')
            self.w2 = tf.get_variable('layer2/kernel')
            self.b2 = tf.get_variable('layer2/bias')
            self.w3 = tf.get_variable('layer3/kernel')
            self.b3 = tf.get_variable('layer3/bias')
            self.w4 = tf.get_variable('output1/kernel')
            self.b4 = tf.get_variable('output1/bias')
            self.w5 = tf.get_variable('output2/kernel')
            self.b5 = tf.get_variable('output2/bias')


class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, memory_size=50000):
        self.capacity = memory_size  # for all priority values
        self.tree = np.zeros(2 * memory_size - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(memory_size, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity


    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame


        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0


    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change


    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root



class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    minimumNum = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 10.  # clipped abs error


    def __init__(self, memory_size=50000, burn_in=10000):
        self.tree = SumTree(memory_size)
        self.memory_size=memory_size
        self.burn_in=burn_in
        self.is_burn_in = False


    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p


    def sample(self, n):
        b_idx, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1))
        b_memory=[]
        # for i in range(n):
        #     b_memory.append(self.tree.data)
        # print(b_memory)
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1


        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        if min_prob == 0:
            min_prob = 0.00001
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i] = idx
            b_memory.append(data)

        return b_idx, b_memory, ISWeights


    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.minimumNum  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        ## 此处ps的转换过程需要修改
        # print(tree_idx)
        # print(ps)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

class DQN_Agent():

    # In this class, we will implement functions to do the following.
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
    #		(a) Epsilon Greedy Policy.
    # 		(b) Greedy Policy.
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.

    def __init__(self, environment_name, sess, gridgraph, render=False):

        # Create an instance of the network itself, as well as the memory.
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc.
        self.epsilon = 0.05
        self.modelReward=0
        self.modelRewardList=[]

        if environment_name == 'grid':
            self.gamma = 0.95
        self.max_episodes = 200  # 20000 #200
        self.batch_size = 32
        self.render = render

        self.qNetwork = QNetwork(environment_name, 'q', trianable=True)
        self.tNetwork = QNetwork(environment_name, 't', trianable=False)
        self.replay = Memory()

        self.gridgraph = gridgraph

        self.as_w1 = tf.assign(self.tNetwork.w1, self.qNetwork.w1)
        self.as_b1 = tf.assign(self.tNetwork.b1, self.qNetwork.b1)
        self.as_w2 = tf.assign(self.tNetwork.w2, self.qNetwork.w2)
        self.as_b2 = tf.assign(self.tNetwork.b2, self.qNetwork.b2)
        self.as_w3 = tf.assign(self.tNetwork.w3, self.qNetwork.w3)
        self.as_b3 = tf.assign(self.tNetwork.b3, self.qNetwork.b3)
        self.as_w4 = tf.assign(self.tNetwork.w4, self.qNetwork.w4)
        self.as_b4 = tf.assign(self.tNetwork.b4, self.qNetwork.b4)
        self.as_w5 = tf.assign(self.tNetwork.w5, self.qNetwork.w5)
        self.as_b5 = tf.assign(self.tNetwork.b5, self.qNetwork.b5)

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
        self.sess.run([self.as_w1, self.as_b1, self.as_w2, self.as_b2, self.as_w3, self.as_b3, self.as_w4, self.as_b4,self.as_w5,self.as_b5])

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
        for episode in np.arange(self.max_episodes * len(self.gridgraph.twopin_combo)):

            # n_node = len([n.name for n in tf.get_default_graph().as_graph_def().node])
            # print("No of nodes: ", n_node, "\n")

            # print('Route:',self.gridgraph.route)

            if(episode%(len(self.gridgraph.twopin_combo))==0 and episode!=0):
                modelReward=self.test(model_file=model_file)
                self.modelRewardList.append(modelReward)
                # 保存参数
                if modelReward>self.modelReward:
                    self.modelReward=modelReward
                    saver = tf.train.Saver()
                    saver.save(sess, model_path + filename + '.ckpt')


            solution_combo.append(self.gridgraph.route)

            state, reward_plot,is_best = self.gridgraph.reset()
            reward_plot_pure = reward_plot - self.gridgraph.posTwoPinNum * 100
            # print('reward_plot-self.gridgraph.posTwoPinNum*100',reward_plot-self.gridgraph.posTwoPinNum*100)

            if (episode) % twoPinNum == 0 and episode!=0:
                reward_plot_combo.append(reward_plot)
                reward_plot_combo_pure.append(reward_plot_pure)
            is_terminal = False
            rewardi = 0.0
            if episode % 100 == 0:
                self.network_assign()

            rewardfortwopin = 0
            while not is_terminal:
                observation = self.gridgraph.state2obsv()
                q_values = self.sess.run(self.qNetwork.output, feed_dict={self.qNetwork.input: observation})
                action = self.epsilon_greedy_policy(q_values)
                # print(action)
                nextstate, reward, is_terminal, debug = self.gridgraph.step(action)
                # print(nextstate)
                observation_next = self.gridgraph.state2obsv()
                self.replay.store([observation, action, reward, observation_next, is_terminal])
                state = nextstate
                rewardi = rewardi + reward
                rewardfortwopin = rewardfortwopin + reward

                batch_idx, batch, ISWeights=self.replay.sample(self.batch_size)
                batch_observation = np.squeeze(np.array([trans[0] for trans in batch]))
                batch_action = np.array([trans[1] for trans in batch])
                batch_reward = np.array([trans[2] for trans in batch])
                batch_observation_next = np.squeeze(np.array([trans[3] for trans in batch]))
                batch_is_terminal = np.array([trans[4] for trans in batch])
                q_batch = self.sess.run(self.qNetwork.output, feed_dict={self.qNetwork.input: batch_observation})
                q_batch_next = self.sess.run(self.qNetwork.output,
                                             feed_dict={self.qNetwork.input: batch_observation_next})
                t_batch_next = self.sess.run(self.tNetwork.output,
                                             feed_dict={self.tNetwork.input: batch_observation_next})
                max_action_next = np.argmax(q_batch_next, axis=1)

                target=np.zeros_like(np.max(t_batch_next, axis=1))
                for i in range(len(target)):
                    target[i]=t_batch_next[i][max_action_next[i]]

                y_batch = batch_reward + self.gamma * (1 - batch_is_terminal) * target

                targetQ = q_batch.copy()
                targetQ[np.arange(self.batch_size), batch_action] = y_batch
                _, train_error = self.sess.run([self.qNetwork.opt, self.qNetwork.loss],
                                               feed_dict={self.qNetwork.input: batch_observation,
                                                          self.qNetwork.targetQ: targetQ})
            reward_log.append(rewardi)  # comment in test; do not save model test

            self.gridgraph.instantrewardcombo.append(rewardfortwopin)
            if self.gridgraph.clearCapacityFlag==1:
                self.gridgraph.passby=np.zeros_like(self.gridgraph.capacity)
                self.gridgraph.clearCapacityFlag=0

        # print(episode, rewardi)

        # if is_best == 1:
        # 	print('self.gridgraph.route',self.gridgraph.route)
        # 		print('Save model')
        # # 		test_reward = self.test()
        # # 		test_reward_log.append(test_reward/20.0)
        # # 		test_episode.append(episode)
        # 		save_path = self.saver.save(self.sess, "{}/model_{}.ckpt".format(savepath,episode))
        # 		print("Model saved in path: %s" % savepath)
        ### Change made
        # if rewardi >= 0:
        # print(self.gridgraph.route)
        # solution_combo.append(self.gridgraph.route)

        # solution = solution_combo[-twoPinNum:]
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

        # print('best reward: ', score)
        # print('solutionDRL: ',solutionDRL,'\n')

        ## Generte solution

        # print ('solution_combo: ',solution_combo)

        #
        # print(test_reward_log)
        # train_episode = np.arange(self.max_episodes)
        # np.savez('../data/training_log.npz', test_episode=test_episode, test_reward_log=test_reward_log,
        # 		 reward_log=reward_log, train_episode=train_episode)

        # self.sess.close()
        tf.reset_default_graph()
        print(netSort)

        return solutionDRL, reward_plot_combo, reward_plot_combo_pure, solution, self.gridgraph.posTwoPinNum,score,capacity,self.modelRewardList

    def test(self, model_file=None, stat=False):
        # Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using a memory.

        # uncomment this line below for videos
        # self.env = gym.wrappers.Monitor(self.env, "recordings", video_callable=lambda episode_id: True)

        # filename = 'test_benchmark_1.gr.test'
        # filename = 'test_benchmark_2.gr'
        # grid_info = init.read(filename)
        gridParameters = init.gridParameters(grid_info)
        gridgraph = env.GridGraph(gridParameters)

        # gridgraph=self.gridgraph

        if model_file is not None:
            self.saver.restore(self.sess, model_file)

        reward_list = []
        cum_reward = 0.0
        cum_twopin=0
        for episode in np.arange(gridgraph.twopinNum):
            episode_reward = 0.0
            state = gridgraph.reset()
            is_terminal = False
            while not is_terminal:
                observation = gridgraph.state2obsv()
                q_values = self.sess.run(self.qNetwork.output, feed_dict={self.qNetwork.input: observation})
                action = self.greedy_policy(q_values)
                nextstate, reward, is_terminal, debug = gridgraph.step(action)
                if reward==100:
                    cum_twopin+=1
                state = nextstate
                episode_reward = episode_reward + reward
                cum_reward = cum_reward + reward
            reward_list.append(episode_reward)
        print('当前回报:',cum_reward)
        print('成功布线的引脚对:',cum_twopin)

        return cum_reward

    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of episodes / transitions.
        print('Start burn in...')
        state = self.gridgraph.reset()
        for i in np.arange(self.replay.burn_in):
            if i % 2000 == 0:
                print('burn in {} samples'.format(i))
            observation = self.gridgraph.state2obsv()
            action = self.gridgraph.sample()
            nextstate, reward, is_terminal, debug = self.gridgraph.step(action)
            observation_next = self.gridgraph.state2obsv()
            self.replay.store([observation, action, reward, observation_next, is_terminal])
            if is_terminal:
                # print(self.gridgraph.current_step)
                state = self.gridgraph.reset()
            else:
                state = nextstate
        self.replay.is_burn_in = True
        print('Burn in finished.')

    def burn_in_memory_search(self, observationCombo, actionCombo, rewardCombo,
                              observation_nextCombo, is_terminalCombo):  # Burn-in with search
        print('Start burn in with A* search algorithm...')
        for i in range(len(observationCombo)):
            observation = observationCombo[i]
            action = actionCombo[i]
            reward = rewardCombo[i]
            observation_next = observation_nextCombo[i]
            is_terminal = is_terminalCombo[i]

            self.replay.store([observation, action, reward, observation_next, is_terminal])

        self.replay.is_burn_in = True
        print('Burn in with A* search algorithm finished.')



if __name__ == '__main__':
    # 初始化环境部分
    environment_name = 'grid'
    filename = 'test_benchmark_1.gr'
    grid_info = init.read(filename)
    gridParameters = init.gridParameters(grid_info)
    gridgraph=env.GridGraph(gridParameters)

    # Setting the session to allow growth, so it doesn't allocate all GPU memory.
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)

    # Setting this as the default tensorflow session.
    keras.backend.tensorflow_backend.set_session(sess)

    # You want to create an instance of the DQN_Agent class here, and then train / test it.
    model_path = './MultiStrategyModel/'
    data_path = './MultiStrategyData/'
    solutionPicture_path='./MultiStrategySolutionPicture/'
    solutionResult_path='./MultiStrategySolutionResult/'

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if not os.path.exists(solutionPicture_path):
        os.makedirs(solutionPicture_path)
    if not os.path.exists(solutionResult_path):
        os.makedirs(solutionResult_path)
    ## 初始化智能体
    agent = DQN_Agent(environment_name, sess, gridgraph, render=False)

    ## 初始化重播缓冲区
    routeListMerged, routeListNotMerged = AStarSearch.getAStarRoute(gridParameters)
    observationCombo, actionCombo, rewardCombo, observation_nextCombo, is_terminalCombo=AStarSearch.getBrunInInform(routeListMerged,gridParameters)
    agent.burn_in_memory_search(observationCombo, actionCombo, rewardCombo, observation_nextCombo, is_terminalCombo)

    ## 开始训练
    solutionDRL, reward_plot_combo, reward_plot_combo_pure, solution, posTwoPinNum,score,bestCapacity,modelRewardList=agent.train(savepath=None, model_file=None)

    print('solutionDRL:',solutionDRL)
    print('reward:',score)
    # testGrid=env.GridGraph(gridParameters)
    # testGrid.capacity=bestCapacity
    # testGrid.showCapacity(filename)
    # test.printDRLSolution(solutionDRL,gridParameters)
    ## 输出测试结果
    overFlow,totalLength=test.testDRLSolution(solutionDRL,gridParameters,solutionPicture_path,filename)
    print('overflow:',overFlow)
    print('length:',totalLength)
    ## 输出reward图
    print('reward_plot_combo:',reward_plot_combo)
    test.printRewardPlot(reward_plot_combo,solutionPicture_path,filename)
    test.printRewardPlot(modelRewardList, solutionPicture_path, 'model.'+filename)
    test.printCompareRewardPlot(reward_plot_combo,modelRewardList,solutionPicture_path,'compare.'+filename)
    ## 输出布线结果
    test.printDRLSolution(solutionDRL,gridParameters,solutionResult_path,filename)

    ## 输出测试结果
    agent.test(model_path + filename + '.ckpt')


    sess.close()

