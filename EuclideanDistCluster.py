import math
from sklearn import preprocessing
count1 = 1
count2 = 1
distance = 0
status = True

eventType = 0

clusterdRowID = []
tempDistances = []

f = open('occu_demo_data.csv', 'rU' ) #open train data


#preping the analysis file with the respective window data
for line in f:
    cells = line.split(",")
    print("line number:"  + str(count1))

    k = open('occu_demo_data.csv', 'rU')  # open train data

    del tempDistances [:]

    for tempLine in k:
        tempCells = tempLine.split(",")
        distance = pow(pow((float(cells[2]) - float(tempCells[2])), 2) + pow((float(cells[3]) - float(tempCells[3])), 2) + pow((float(cells[4]) - float(tempCells[4])), 2) + pow((float(cells[5]) - float(tempCells[5])), 2) + pow((float(cells[6]) - float(tempCells[6])), 2),0.5)
        tempDistances.append(distance)

    k.close()

    print(tempDistances)

    count2 = 1
    status = True
    for x in tempDistances:

        if count2 not in clusterdRowID:
            if x <= 600:
                clusterdRowID.append(count2)
                if status:
                    eventType = eventType + 1
                    print("Event Type" + str(eventType))
                status = False
                print("Row number: " +str(count2))

        count2 = count2 + 1

    count1 = count1 + 1

f.close()