import GridGraph as gridGraph
import Initializer as init
import numpy as np
import matplotlib.pyplot as plt


def printDRLSolution(solutionDRL, gridParameters,path,filename):
    env=gridGraph.GridGraph(gridParameters)

    # dump solution of DRL
    f = open(path+filename+'.DRLsolution', 'w+')
    # for i in range(1):
    twoPinSolutionPointer = 0
    # print('solution_combo_filled',solution_combo_filled)

    for i in range(gridParameters['numNet']):
        singleNetRouteCache = []

        value = '{netName} {netID} {cost}\n'.format(netName=gridParameters['netInfo'][i]['netName'],
                                                    netID=gridParameters['netInfo'][i]['netID'],
                                                    cost=0)  # max(0,len(routeListMerged[indicator])-1))
        f.write(value)
        for j in range(len(solutionDRL[i])):
            for k in range(len(solutionDRL[i][j]) - 1):

                a = solutionDRL[i][j][k]
                b = solutionDRL[i][j][k + 1]

                if (a[3], a[4], a[2], b[3], b[4], b[2]) not in singleNetRouteCache:
                    # and (b[3],b[4],b[2]) not in singleNetRouteCacheSmall:
                    singleNetRouteCache.append((a[3], a[4], a[2], b[3], b[4], b[2]))
                    singleNetRouteCache.append((b[3], b[4], b[2], a[3], a[4], a[2]))
                    # singleNetRouteCacheSmall.append((a[3],a[4],a[2]))
                    # singleNetRouteCacheSmall.append((b[3],b[4],b[2]))

                    # diff = [abs(a[2] - b[2]), abs(a[3] - b[3]), abs(a[4] - b[4])]
                    # if diff[1] > 2 or diff[2] > 2:
                    #     continue
                    # elif diff[1] == 2 or diff[2] == 2:
                    #     continue
                    # elif diff[0] == 0 and diff[1] == 0 and diff[2] == 0:
                    #     continue
                    # elif diff[0] + diff[1] + diff[2] >= 2:
                    #     continue
                    # else:
                    value = '({},{},{})-({},{},{})\n'.format(int(a[0]), int(a[1]), a[2], int(b[0]), int(b[1]), b[2])
                    f.write(value)
            twoPinSolutionPointer = twoPinSolutionPointer + 1
        f.write('!\n')
    f.close()

def testDRLSolution(solutionDRL, gridParameters,path,filename):
    env = gridGraph.GridGraph(gridParameters)
    capacity = env.capacity
    # print(solutionDRL)
    # for i in range(len(solutionDRL)):
    #     solution=[]
    #     for j in range(len(solutionDRL[i])):
    #         for k in range(len(solutionDRL[i][j])):
    #             solution.append(solutionDRL[i][j][k])
    #     solutionTest.append(solution)
    #
    # print(solutionTest)

    overFlow=0
    totalLength=0

    for i in range(gridParameters['numNet']):
        singleNetRouteCache = []

        for j in range(len(solutionDRL[i])):
            for k in range(len(solutionDRL[i][j]) - 1):

                a = solutionDRL[i][j][k]
                b = solutionDRL[i][j][k + 1]
                action_1 = gridGraph.get_action(a, b)
                action_2 = gridGraph.get_action(b, a)
                # print(action_1,action_2)
                if (a[3], a[4], a[2], b[3], b[4], b[2]) not in singleNetRouteCache:
                    # and (b[3],b[4],b[2]) not in singleNetRouteCacheSmall:
                    singleNetRouteCache.append((a[3], a[4], a[2], b[3], b[4], b[2]))
                    singleNetRouteCache.append((b[3], b[4], b[2], a[3], a[4], a[2]))
                    # singleNetRouteCacheSmall.append((a[3],a[4],a[2]))
                    # singleNetRouteCacheSmall.append((b[3],b[4],b[2]))

                    # diff = [abs(a[2] - b[2]), abs(a[3] - b[3]), abs(a[4] - b[4])]
                    # if diff[1] > 2 or diff[2] > 2:
                    #     continue
                    # elif diff[1] == 2 or diff[2] == 2:
                    #     continue
                    # elif diff[0] == 0 and diff[1] == 0 and diff[2] == 0:
                    #     continue
                    # elif diff[0] + diff[1] + diff[2] >= 2:
                    #     continue
                    # else:
                    capacity[a[3], a[4], a[2] - 1, action_1] -= 1
                    capacity[b[3], b[4], b[2] - 1, action_2] -= 1
                    totalLength+=1
                    if capacity[a[3], a[4], a[2] - 1, action_1]<0:
                        overFlow+=1
    # print(capacity)
    env.capacity=capacity
    name=path+filename
    env.showCapacity(name)
    return overFlow,totalLength
    # env.showCapacity()

def printRewardPlot(rewardCombo,path,filename):
    n = np.linspace(1, len(rewardCombo), len(rewardCombo))
    plt.figure()
    plt.plot(n, rewardCombo)
    plt.xlabel('episodes')
    plt.ylabel('reward')
    plt.savefig(path+filename+'.reward.jpg')
    # plt.show()
    plt.close()

def printCompareRewardPlot(rewardCombo,modelRewardCombo,path,filename):
    n = np.linspace(1, len(rewardCombo), len(modelRewardCombo))
    plt.figure()
    plt.plot(n, rewardCombo,label="evaluated")
    plt.plot(n, modelRewardCombo,label="real",color="#DB7093")
    plt.xlabel('episodes')
    plt.ylabel('reward')
    plt.show()
    plt.savefig(path+filename+'.reward.jpg')
    # plt.show()
    plt.close()


if __name__ == '__main__':
    f=open("solutionDRL.txt", "r")
    solutionDRL = f.readline()
    print(solutionDRL)
    filename = 'test_benchmark_1.gr'
    grid_info = init.read(filename)
    gridParameters = init.gridParameters(grid_info)
    # testDRLSolution(solutionDRL, gridParameters)