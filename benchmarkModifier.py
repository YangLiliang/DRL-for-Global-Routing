def read(grfile):
    file = open(grfile,'r')
    grid_info = {}
    i = 0
    for line in file:
        if not line.strip():
            continue
        else:
            grid_info[i]= line.split()
        i += 1
    file.close()
    return grid_info

# Parsing input data()
def gridParameters(grid_info):
    gridParameters = {}
    gridParameters['gridSize'] = [int(grid_info[0][1]),int(grid_info[0][2]),int(grid_info[0][3])]
    gridParameters['verticalCapacity'] = [float(grid_info[1][2]),float(grid_info[1][3])]
    gridParameters['horizontalCapacity'] = [float(grid_info[2][2]), float(grid_info[2][3])]
    gridParameters['minWidth'] = [float(grid_info[3][2]), float(grid_info[3][3])]
    gridParameters['minSpacing'] = [float(grid_info[4][2]), float(grid_info[4][3])]
    gridParameters['viaSpacing'] = [float(grid_info[5][2]), float(grid_info[5][3])]
    gridParameters['Origin'] = [float(grid_info[6][0]), float(grid_info[6][1])]
    gridParameters['tileWidth'] = float(grid_info[6][2]); gridParameters['tileHeight'] = float(grid_info[6][3])
    gridParameters['reducedCapacitySpecify'] = {}
    for lineNum in range(len(grid_info)):
        if 'num' in grid_info[lineNum]:
            gridParameters['numNet'] = int(grid_info[lineNum][2])
    netNum = 0
    pinEnumerator = 1; lineEnumerator = 8
    netParametersStore = []
    for lineNum in range(7,len(grid_info)):
        if 'A' in grid_info[lineNum][0]:
            netParameters = {}
            netParameters['netName'] = grid_info[lineNum][0]
            netParameters['netID'] = int(grid_info[lineNum][1])
            netParameters['numPins'] = int(grid_info[lineNum][2])
            netParameters['minWidth'] = float(grid_info[lineNum][3])
            pinNum = 1
            while ('A' not in grid_info[lineNum+pinNum][0]) and (len(grid_info[lineNum+pinNum])>1):
                netParameters[str(pinNum)] = [int(grid_info[lineNum+pinNum][0]),int(grid_info[lineNum+pinNum][1]),
                                             int(grid_info[lineNum+pinNum][2])]
                pinNum += 1

            gridParameters['reducedCapacity'] = grid_info[lineNum+pinNum]
            pinEnumerator = pinNum
            lineEnumerator = lineNum + pinNum + 1
            netParametersStore.append(netParameters)
        if ('n' in grid_info[lineNum][0])and (grid_info[lineNum][0] != 'num'):
            netParameters = {}
            netParameters['netName'] = grid_info[lineNum][0]
            netParameters['netID'] = int(grid_info[lineNum][1])
            netParameters['numPins'] = int(grid_info[lineNum][2])
            netParameters['minWidth'] = float(grid_info[lineNum][3])
            pinNum = 1
            while ('n' not in grid_info[lineNum+pinNum][0]) and (len(grid_info[lineNum+pinNum])>1):
                netParameters[str(pinNum)] = [int(grid_info[lineNum+pinNum][0]),int(grid_info[lineNum+pinNum][1]),
                                             int(grid_info[lineNum+pinNum][2])]
                pinNum += 1

            gridParameters['reducedCapacity'] = grid_info[lineNum+pinNum]
            pinEnumerator = pinNum
            lineEnumerator = lineNum + pinNum + 1
            netParametersStore.append(netParameters)
        gridParameters['netInfo'] = netParametersStore
    # Parsing adjustments depicting reduced capacity (override layer specification)
    i = 1
    for lineNum in range(lineEnumerator, len(grid_info)):
        reducedEdge = [int(grid_info[lineNum][0]),int(grid_info[lineNum][1]),int(grid_info[lineNum][2]),
                       int(grid_info[lineNum][3]),int(grid_info[lineNum][4]),int(grid_info[lineNum][5]),
                       int(grid_info[lineNum][6])]
        gridParameters['reducedCapacitySpecify'][str(i)] = reducedEdge
            # grid_info[lineNum]
        i += 1
    return gridParameters