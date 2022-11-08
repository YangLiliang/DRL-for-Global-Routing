import Initializer as init
import GenerateMST as MST
import operator

def GenerateTwoPin(gridParameters):
    # filename = 'test_benchmark_2.gr'
    # grid_info = init.read(filename)
    # gridParameters = init.gridParameters(grid_info)

    # 根据net边缘长度进行排序
    halfWireLength = init.VisualGraph(gridParameters).bounding_length()
    sortedHalfWireLength = sorted(halfWireLength.items(), key=operator.itemgetter(1), reverse=True)  # Large2Small
    netOrder = []
    for i in range(gridParameters['numNet']):
        order = int(sortedHalfWireLength[i][0])
        netOrder.append(order)

    # 生成引脚对
    twopin_combo = [] ## 存储引脚对
    net_pairNum = [] ## 存储每个net内有多少引脚对
    for numNet in range(gridParameters['numNet']):
        netPinList = []
        netPinCoord = []
        i = netOrder[numNet]
        # 存储引脚坐标
        for j in range(gridParameters['netInfo'][i]['numPins']):
            pin = tuple([int((gridParameters['netInfo'][i][str(j + 1)][0] - gridParameters['Origin'][0]) /
                             gridParameters['tileWidth']),
                         int((gridParameters['netInfo'][i][str(j + 1)][1] - gridParameters['Origin'][1]) /
                             gridParameters['tileHeight']),
                         int(gridParameters['netInfo'][i][str(j + 1)][2]),
                         int(gridParameters['netInfo'][i][str(j + 1)][0]),
                         int(gridParameters['netInfo'][i][str(j + 1)][1])])
            if pin[0:3] in netPinCoord:
                continue
            else:
                netPinList.append(pin)
                netPinCoord.append(pin[0:3])
        twoPinList = []
        # 存储引脚对
        for j in range(len(netPinList) - 1):
            pinStart = netPinList[j]
            pinEnd = netPinList[j + 1]
            twoPinList.append([pinStart, pinEnd])

        # Insert Tree method to decompose two pin problems here
        # 生成MST
        twoPinList_MST = MST.generateMST(twoPinList)

        twopin_combo.append(twoPinList_MST)
        net_pairNum.append(len(twoPinList_MST))

    # twopin_combo的格式如下：
    # [[[(0, 7, 1, 4, 78), (7, 0, 1, 78, 1)]],[[(2, 7, 1, 22, 78), (7, 7, 1, 76, 73)], [(7, 7, 1, 76, 73), (7, 0, 1, 77, 8)]]]
    twopin_combo_nonet = []
    for net in twopin_combo:
        # for net in twopinListComboCleared:
        for pinpair in net:
            twopin_combo_nonet.append(pinpair)
    # twopin_combo_nonet的格式如下：
    # [[(0, 7, 1, 4, 78), (7, 0, 1, 78, 1)],[(2, 7, 1, 22, 78), (7, 7, 1, 76, 73)], [(7, 7, 1, 76, 73), (7, 0, 1, 77, 8)]]
    return netOrder, net_pairNum,twopin_combo_nonet,twopin_combo

    # print('twopinListComboCleared',twopinListComboCleared)
if __name__ == '__main__':
    netOrder,net_pairNum, twopin_combo_nonet,twopin_combo=GenerateTwoPin()
    print(netOrder)
    print(net_pairNum)
    print(twopin_combo_nonet)
    print(twopin_combo)

    # 'gridSize': [8, 8, 2]
    # 'verticalCapacity': [0.0, 4.0]
    # 'horizontalCapacity': [4.0, 0.0]
    # 'minWidth': [1.0, 1.0]
    # 'minSpacing': [0.0, 0.0]
    # 'viaSpacing': [0.0, 0.0]
    # 'Origin': [0.0, 0.0]
    # 'tileWidth': 10.0
    # 'tileHeight': 10.0
    # 'numNet': 20