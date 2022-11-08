import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
#matplotlib.use('TkAgg')
import numpy as np

import Initializer as init
import GenerateTwoPin as GTP


observation_space = 12
action_space = 6
# Create grid graph based on parsed input info
class GridGraph(object):
    def __init__(self, gridParameters, agentname = None):
        self.gridParameters = gridParameters
        self.max_step = 100
        self.current_step = 0
        self.goal_state = None
        self.init_state = None
        self.current_state = None
        self.capacity = self.generate_capacity()
        self.route = []
        self.netOrder, self.net_pair, self.twopin_combo, self.twopin_combo_net = GTP.GenerateTwoPin(gridParameters)
        self.twopinNum = len(self.twopin_combo)
        self.twopin_pt = 0
        self.reward = 0.0
        self.instantreward = 0.0
        self.instantrewardcombo = []
        self.best_reward = 0.0
        self.best_route = []
        self.route_combo = []
        self.net_ind = 0
        self.pair_ind = 0
        self.passby = np.zeros_like(self.capacity)
        self.previous_action = -1
        self.posTwoPinNum = 0
        self.episode = 0
        self.positivePin=0
        self.bestCapacity=[]
        self.clearCapacityFlag=0
        self.agentname=agentname
        return


    def generate_grid(self):
        # Initialize grid coordinates
        # Input: grid size
        gridX, gridY, gridZ = np.meshgrid(np.arange(self.gridParameters['gridSize'][0]),
                                          np.arange(self.gridParameters['gridSize'][1]),
                                          np.arange(self.gridParameters['gridSize'][2]))

        return gridX, gridY, gridZ

    def generate_capacity(self):
        # Input: VerticalCapacity, HorizontalCapacity, ReducedCapacity, MinWidth, MinSpacing
        # Update Input: Routed Nets Path
        # Capacity description direction:
        # [0:+X, 1:-X, 2:+Y, 3:-Y, 4:+Z, 5:-Z]

        capacity = np.zeros((self.gridParameters['gridSize'][0], self.gridParameters['gridSize'][1],
                             self.gridParameters['gridSize'][2], 6))

        ## Apply initial condition to capacity
        # Calculate Available NumNet in each direction
        # Layer 0
        verticalNumNet = [self.gridParameters['verticalCapacity'][0] /
                          (self.gridParameters['minWidth'][0] + self.gridParameters['minSpacing'][0]),
                          self.gridParameters['verticalCapacity'][1] /
                          (self.gridParameters['minWidth'][1] + self.gridParameters['minSpacing'][1])]
        horizontalNumNet = [self.gridParameters['horizontalCapacity'][0] /
                            (self.gridParameters['minWidth'][0] + self.gridParameters['minSpacing'][0]),
                            self.gridParameters['horizontalCapacity'][1] /
                            (self.gridParameters['minWidth'][1] + self.gridParameters['minSpacing'][1])]
        # print(horizontalNumNet)
        # Apply available NumNet to grid capacity variables
        capacity[:, :, 0, 0] = capacity[:, :, 0, 1] = horizontalNumNet[0]
        capacity[:, :, 1, 0] = capacity[:, :, 1, 1] = horizontalNumNet[1]
        capacity[:, :, 0, 2] = capacity[:, :, 0, 3] = verticalNumNet[0]
        capacity[:, :, 1, 2] = capacity[:, :, 1, 3] = verticalNumNet[1]

        # Assume Via Ability to be very large
        capacity[:, :, 0, 4] = 10
        capacity[:, :, 1, 5] = 10

        # Remove edge capacity
        capacity[:, :, 1, 4] = 0
        capacity[:, :, 0, 5] = 0  # Z-direction edge capacity edge removal
        capacity[:, 0, :, 3] = 0
        capacity[:, self.gridParameters['gridSize'][1] - 1, :, 2] = 0  # Y-direction edge capacity edge removal
        capacity[0, :, :, 1] = 0
        capacity[self.gridParameters['gridSize'][0] - 1, :, :, 0] = 0  # X-direction edge capacity edge removal
        return capacity

    # def pin_density(self):
    #     # Input: pin location globally
    #     return

    def step(self, action):  # used for DRL
        state = self.current_state
        reward = -1.0
        # if action == 0 and (self.capacity[state[0], state[1], state[2] - 1, 0] > 0 ):
        #     nextState = (state[0] + 1, state[1], state[2], state[3] + self.gridParameters['tileWidth'], state[4])
        #     if self.passby[state[0], state[1], state[2] - 1, 0] == 0:
        #         self.passby[state[0], state[1], state[2] - 1, 0] = 1
        #         self.passby[state[0] + 1, state[1], state[2] - 1, 1] = 1
        #         self.updateCapacity(state, action)
        #     self.route.append((state[3], state[4], state[2], state[0], state[1]))
        # elif action == 1 and (self.capacity[state[0], state[1], state[2] - 1, 1] > 0 ):
        #     nextState = (state[0] - 1, state[1], state[2], state[3] - self.gridParameters['tileWidth'], state[4])
        #     if self.passby[state[0], state[1], state[2] - 1, 1] == 0:
        #         self.passby[state[0], state[1], state[2] - 1, 1] = 1
        #         self.passby[state[0] - 1, state[1], state[2] - 1, 0] = 1
        #         self.updateCapacity(state, action)
        #     self.route.append((state[3], state[4], state[2], state[0], state[1]))
        # elif action == 2 and (self.capacity[state[0], state[1], state[2] - 1, 2] > 0 ):
        #     nextState = (state[0], state[1] + 1, state[2], state[3], state[4] + self.gridParameters['tileHeight'])
        #     if self.passby[state[0], state[1], state[2] - 1, 2] == 0:
        #         self.passby[state[0], state[1], state[2] - 1, 2] = 1
        #         self.passby[state[0], state[1] + 1, state[2] - 1, 3] = 1
        #         self.updateCapacity(state, action)
        #     self.route.append((state[3], state[4], state[2], state[0], state[1]))
        # elif action == 3 and (self.capacity[state[0], state[1], state[2] - 1, 3] > 0 ):
        #     nextState = (state[0], state[1] - 1, state[2], state[3], state[4] - self.gridParameters['tileHeight'])
        #     if self.passby[state[0], state[1], state[2] - 1, 3] == 0:
        #         self.passby[state[0], state[1], state[2] - 1, 3] = 1
        #         self.passby[state[0], state[1] - 1, state[2] - 1, 2] = 1
        #         self.updateCapacity(state, action)
        #     self.route.append((state[3], state[4], state[2], state[0], state[1]))
        # elif action == 4 and (self.capacity[state[0], state[1], state[2] - 1, 4] > 0 ):
        #     nextState = (state[0], state[1], state[2] + 1, state[3], state[4])
        #     if self.passby[state[0], state[1], state[2] - 1, 4] == 0:
        #         self.passby[state[0], state[1], state[2] - 1, 4] = 1
        #         self.passby[state[0], state[1], state[2], 5] = 1
        #         self.updateCapacity(state, action)
        #     self.route.append((state[3], state[4], state[2], state[0], state[1]))
        # elif action == 5 and (self.capacity[state[0], state[1], state[2] - 1, 5] > 0 ):
        #     nextState = (state[0], state[1], state[2] - 1, state[3], state[4])
        #     if self.passby[state[0], state[1], state[2] - 1, 5] == 0:
        #         self.passby[state[0], state[1], state[2] - 1, 5] = 1
        #         self.passby[state[0], state[1], state[2] - 2, 4] = 1
        #         self.updateCapacity(state, action)
        #     self.route.append((state[3], state[4], state[2], state[0], state[1]))
        # else:
        #     nextState = state
        if action == 0 and (self.capacity[state[0], state[1], state[2] - 1, 0] > 0 or self.passby[state[0], state[1], state[2] - 1, 0] == 1):
            nextState = (state[0] + 1, state[1], state[2], state[3] + self.gridParameters['tileWidth'], state[4])
            if self.passby[state[0], state[1], state[2] - 1, 0] == 0:
                self.passby[state[0], state[1], state[2] - 1, 0] = 1
                self.passby[state[0] + 1, state[1], state[2] - 1, 1] = 1
                self.updateCapacity(state, action)
            self.route.append((state[3], state[4], state[2], state[0], state[1]))
        elif action == 1 and (self.capacity[state[0], state[1], state[2] - 1, 1] > 0 or self.passby[state[0], state[1], state[2] - 1, 1] == 1):
            nextState = (state[0] - 1, state[1], state[2], state[3] - self.gridParameters['tileWidth'], state[4])
            if self.passby[state[0], state[1], state[2] - 1, 1] == 0:
                self.passby[state[0], state[1], state[2] - 1, 1] = 1
                self.passby[state[0] - 1, state[1], state[2] - 1, 0] = 1
                self.updateCapacity(state, action)
            self.route.append((state[3], state[4], state[2], state[0], state[1]))
        elif action == 2 and (self.capacity[state[0], state[1], state[2] - 1, 2] > 0 or self.passby[state[0], state[1], state[2] - 1, 2] == 1):
            nextState = (state[0], state[1] + 1, state[2], state[3], state[4] + self.gridParameters['tileHeight'])
            if self.passby[state[0], state[1], state[2] - 1, 2] == 0:
                self.passby[state[0], state[1], state[2] - 1, 2] = 1
                self.passby[state[0], state[1] + 1, state[2] - 1, 3] = 1
                self.updateCapacity(state, action)
            self.route.append((state[3], state[4], state[2], state[0], state[1]))
        elif action == 3 and (self.capacity[state[0], state[1], state[2] - 1, 3] > 0 or self.passby[state[0], state[1], state[2] - 1, 3] == 1):
            nextState = (state[0], state[1] - 1, state[2], state[3], state[4] - self.gridParameters['tileHeight'])
            if self.passby[state[0], state[1], state[2] - 1, 3] == 0:
                self.passby[state[0], state[1], state[2] - 1, 3] = 1
                self.passby[state[0], state[1] - 1, state[2] - 1, 2] = 1
                self.updateCapacity(state, action)
            self.route.append((state[3], state[4], state[2], state[0], state[1]))
        elif action == 4 and (self.capacity[state[0], state[1], state[2] - 1, 4] > 0 or self.passby[state[0], state[1], state[2] - 1, 4] == 1):
            nextState = (state[0], state[1], state[2] + 1, state[3], state[4])
            if self.passby[state[0], state[1], state[2] - 1, 4] == 0:
                self.passby[state[0], state[1], state[2] - 1, 4] = 1
                self.passby[state[0], state[1], state[2], 5] = 1
                self.updateCapacity(state, action)
            self.route.append((state[3], state[4], state[2], state[0], state[1]))
        elif action == 5 and (self.capacity[state[0], state[1], state[2] - 1, 5] > 0 or self.passby[state[0], state[1], state[2] - 1, 5] == 1):
            nextState = (state[0], state[1], state[2] - 1, state[3], state[4])
            if self.passby[state[0], state[1], state[2] - 1, 5] == 0:
                self.passby[state[0], state[1], state[2] - 1, 5] = 1
                self.passby[state[0], state[1], state[2] - 2, 4] = 1
                self.updateCapacity(state, action)
            self.route.append((state[3], state[4], state[2], state[0], state[1]))
        else:
            nextState = state
            # reward=-2.0
            # reward = -2.0

        self.current_state = nextState
        self.current_step += 1
        # distance=abs(state[0]-nextState[0])+abs(state[1]-nextState[1])+abs(state[2]-nextState[2])
        # reward= -distance
        # reward = -1.0
        # reward_d = np.array(self.current_state[:3])-np.array(self.goal_state[:3])
        # reward_d = -np.sum(np.abs(reward_d))/20.0
        # reward += reward_d
        done = False
        if self.current_state[:3] == self.goal_state[:3]:
            done = True
            reward = 100
            self.positivePin+=1
            self.route.append((self.current_state[3], self.current_state[4], self.current_state[2],
                               self.current_state[0], self.current_state[1]))
        # elif np.sum(self.state2obsv()[3:9]) == 0:
        #     done = True
        #     self.twopin_pt = 100

        elif self.current_step >= self.max_step:
            done = True
            self.route.append((self.current_state[3], self.current_state[4], self.current_state[2],
                               self.current_state[0], self.current_state[1]))

        self.reward = self.reward + reward
        self.instantreward = reward


        # self.instantrewardcombo.append(reward)
        return self.state2obsv(), reward, done, []

    def reset(self):
        # if self.loop == 0:
        #     self.twopin_rdn =
        reward_plot = 0
        isBest = 0



        if self.twopin_pt >= len(self.twopin_combo):
            self.episode = self.episode + 1
            self.twopin_pt = 0
            self.net_ind = 0
            self.pair_ind = 0
            print("Reward: ", self.reward)
            reward_plot = self.reward
            self.route_combo.append(self.route)

            if self.positivePin > self.posTwoPinNum:
                self.best_reward = self.reward
                self.best_route = self.route_combo
                self.posTwoPinNum = self.positivePin
                self.bestCapacity=self.capacity
                isBest=1

            elif self.positivePin == self.posTwoPinNum:
                if self.reward > self.best_reward:
                    self.best_reward = self.reward
                    self.best_route = self.route_combo
                    self.bestCapacity = self.capacity
                    isBest=1

            # print(self.route_combo)
            self.route_combo = []
            self.reward = 0.0
            self.instantrewardcombo = []
            self.capacity = self.generate_capacity()
            print('Current agent:',self.agentname)
            print('Current best reward:',self.best_reward)
            print('Positive two pin num: {}/{}'.format(self.positivePin, len(self.twopin_combo)))
            print('\nNew loop!')
            print('Episode: {num}'.format(num=self.episode + 1))
            self.positivePin=0
        else:
            if len(self.route)>0:
                self.route_combo.append(self.route)

        # print(self.twopin_pt)
        # print(self.twopin_combo)
        self.init_state = self.twopin_combo[self.twopin_pt][0]
        self.goal_state = self.twopin_combo[self.twopin_pt][1]
        # print(self.init_state)
        self.current_state = self.init_state
        self.current_step = 0

        self.pair_ind += 1
        # print(self.pair_ind,'/',self.net_pair[self.net_ind],',',self.net_ind)
        if self.pair_ind >= self.net_pair[self.net_ind]:
            self.net_ind += 1
            self.pair_ind = 0
            self.clearCapacityFlag=1
            # print('clear')


        ### Change Made
        # print(self.route)
        self.route = []
        self.twopin_pt += 1

        return self.state2obsv(),reward_plot

    def state2obsv(self):
        state = np.array(self.current_state)
        capacity = np.squeeze(self.capacity[int(state[0]), int(state[1]), int(state[2]) - 1, :])
        distance = np.array(self.goal_state) - state
        observation = np.concatenate((state[:3], capacity, distance[:3]), axis=0).reshape(1, -1)
        # return observation.flatten()
        return observation

    def showCapacity(self, filename=None):
        # print(self.capacity)
        layer1_x=self.capacity[0:self.gridParameters['gridSize'][1]-1,:,0,0]
        l1_flat=layer1_x.flatten()

        layer2_y = self.capacity[:, 0:self.gridParameters['gridSize'][1] - 1, 1, 2]
        l2_flat=layer2_y.flatten()
        a=np.append(l1_flat,l2_flat)
        # print(a)
        print('标准差', np.std(a, ddof=1))
        # print(layer1_x)
        # print(layer1_y)
        # print(layer2_x)
        # print(layer2_y)

        print('layer1:', layer1_x)
        # ax=sns.heatmap(layer1_x, cmap='Reds',fmt='.1f')
        ax = sns.heatmap(layer1_x, cmap='Reds',annot=True,fmt=".1f")  # 将每个方格的数据显示出来


        if filename!=None:
            plt.savefig(filename + '.layer1.png')
        plt.show()
        plt.close()
        # sns.heatmap(layer1_y, cmap='Reds')
        # plt.show()
        # sns.heatmap(layer2_x, cmap='Reds')
        # plt.show()
        print('layer2:', layer2_y)
        sns.heatmap(layer2_y, cmap='Reds',annot=True,fmt=".1f")
        if filename != None:
            plt.savefig(filename+'.layer2.png')
        plt.show()
        plt.close()

    def updateCapacity(self,state,action):
        if action == 0:
            self.capacity[state[0], state[1], state[2] - 1, 0] -= 1
            self.capacity[state[0] + 1, state[1], state[2] - 1, 1] -= 1
        elif action == 1:
            self.capacity[state[0], state[1], state[2] - 1, 1] -= 1
            self.capacity[state[0] - 1, state[1], state[2] - 1, 0] -= 1
        elif action == 2:
            self.capacity[state[0], state[1], state[2] - 1, 2] -= 1
            self.capacity[state[0], state[1] + 1, state[2] - 1, 3] -= 1
        elif action == 3:
            self.capacity[state[0], state[1], state[2] - 1, 3] -= 1
            self.capacity[state[0], state[1] - 1, state[2] - 1, 2] -= 1
        elif action == 4:
            self.capacity[state[0], state[1], state[2] - 1, 4] -= 1
            self.capacity[state[0], state[1], state[2], 5] -= 1
        elif action == 5:
            self.capacity[state[0], state[1], state[2] - 1, 5] -= 1
            self.capacity[state[0], state[1], state[2] - 2, 4] -= 1



def updateCapacity(capacity, route):
    for i in range(len(route) - 1):
        diff = [route[i + 1][0] - route[i][0],
                route[i + 1][1] - route[i][1],
                route[i + 1][2] - route[i][2]]

        if diff[0] == 1:
            capacity[route[i][0], route[i][1], route[i][2] - 1, 0] -= 1
            capacity[route[i + 1][0], route[i + 1][1], route[i + 1][2] - 1, 1] -= 1
        elif diff[0] == -1:
            capacity[route[i][0], route[i][1], route[i][2] - 1, 1] -= 1
            capacity[route[i + 1][0], route[i + 1][1], route[i + 1][2] - 1, 0] -= 1
        elif diff[1] == 1:
            capacity[route[i][0], route[i][1], route[i][2] - 1, 2] -= 1
            capacity[route[i + 1][0], route[i + 1][1], route[i + 1][2] - 1, 3] -= 1
        elif diff[1] == -1:
            capacity[route[i][0], route[i][1], route[i][2] - 1, 3] -= 1
            capacity[route[i + 1][0], route[i + 1][1], route[i + 1][2] - 1, 2] -= 1
        elif diff[2] == 1:
            capacity[route[i][0], route[i][1], route[i][2] - 1, 4] -= 1
            capacity[route[i + 1][0], route[i + 1][1], route[i + 1][2] - 1, 5] -= 1
        elif diff[2] == -1:
            capacity[route[i][0], route[i][1], route[i][2] - 1, 5] -= 1
            capacity[route[i + 1][0], route[i + 1][1], route[i + 1][2] - 1, 4] -= 1
    return capacity


def get_action(position, nextposition):
    # position example (20,10,2,2,1)
    diff = (nextposition[3] - position[3], nextposition[4] - position[4], nextposition[2] - position[2])
    action = 0
    if diff[0] == 1:
        action = 0
    elif diff[0] == -1:
        action = 1
    elif diff[1] == 1:
        action = 2
    elif diff[1] == -1:
        action = 3
    elif diff[2] == 1:
        action = 4
    elif diff[2] == -1:
        action = 5
    return action

if __name__ == '__main__':
    # Filename corresponds to benchmark to route
    # filename = 'small.gr'
    filename = 'test_benchmark_1.gr'
    # filename = 'adaptec1.capo70.2d.35.50.90.gr'
    # filename = 'sampleBenchmark'

    # Getting Net Info
    grid_info = init.read(filename)
    gridParameters=init.gridParameters(grid_info)
    grid=GridGraph(gridParameters)
    # print(grid_info)
    grid.showCapacity()
    print(grid.twopin_combo)
    print(grid.net_pair)
    # # # print(init.gridParameters(grid_info)['netInfo'])
    #
    # for item in init.gridParameters(grid_info).items():
    #     print(item)

    # # for net in init.gridParameters(grid_info)['netInfo']:
    # #     print (net)
    # init.GridGraph(init.gridParameters(grid_info)).show_grid()
    # init.GridGraph(init.gridParameters(grid_info)).pin_density_plot()
    #
    # capacity = GridGraph(init.gridParameters(grid_info)).generate_capacity()
    # print(capacity[:,:,0,1])
    # gridX, gridY, gridZ= GridGraph(init.gridParameters(grid_info)).generate_grid()
    # print(gridX[1,1,0])
    # print(gridY[1,1,0])
    # print(gridZ[1,1,0])

    # print('capacity[1,0,0,:]',capacity[1,0,0,:])
    # print('capacity[2,0,0,:]',capacity[2,0,0,:])
    # print('capacity[1,1,1,:]',capacity[1,1,1,:])
    # print('capacity[0,1,1,:]',capacity[0,1,1,:])
    # print('capacity[2,2,1,:]',capacity[2,2,1,:])


    # # Check capacity update
    # print("Check capacity update")
    # print(capacity[1, 2, 0, 4])
    # RouteListMerged = [(1,2,1,12,23),(1,2,2,12,23)] # Coordinates rule follows (xGrid, yGrid,Layer(1,2),xLength,yLength)
    # capacity = updateCapacity(capacity,RouteListMerged)
    # print(capacity[1, 2, 0, 4])

    # # # Check capacity update
    # print("Check updateCapacityRL")
    # print(capacity[1,2,0,3])
    # state = (1,2,1,13,23); action = 3;
    # capacity = updateCapacityRL(capacity,state,action)
    # print(capacity[1,2,0,3])
    # print(capacity[1,1,0,2])

    # # # Check get action
    # position = (20, 60, 2, 2, 6)
    # nextposition = (20, 50, 2, 2, 5)
    # actiontest = get_action(position,nextposition)
    # print('Action',actiontest)