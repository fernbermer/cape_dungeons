import csv
import numpy as np
from sklearn import linear_model
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dates = [1,2,3,4,5,6,7]
positions = []
bigPositions =[]

# Map row to dictionary (dictionary comprehension)
def shareholder(column_names, row):
    return {column_names[column]: data for column, data in enumerate(row) if column < len(column_names)}

# Map CSV file to list of dictionaries (list comprehension)
#position0 = Current Position and position8 = Public Position
shareholders = [shareholder(['Rank','institutionName','position0', 'position1', 'position2', 'position3','position4', 'position5', 'position6','position7','position8', 'Sharesout'], row) for row in csv.reader(open('keycorp.csv', 'r'))]


def predict_prices(dates, prices, x):
  dates = np.reshape(dates, (len(dates), 1))

  svr_lin = SVR(kernel='linear', C=1e3)
  svr_poly = SVR(kernel = 'poly', C=1e3, degree=2)
  svr_rbf = SVR(kernel = 'rbf', C=1e3, gamma=0.1)
  svr_rbf.fit(dates, prices)
  svr_lin.fit(dates, prices)
  svr_poly.fit(dates, prices)

  plt.scatter(dates, prices, color='black', label='Data')
  plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF model')
  plt.plot(dates, svr_lin.predict(dates), color='green', label='Linear model')
  plt.plot(dates, svr_poly.predict(dates), color='blue', label='Polynomial model')
  plt.xlabel('Date')
  plt.ylabel('Price')
  plt.title('Support Vector Regression')
  plt.legend()
  plt.show()

  return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]

#try to get positions into array
for i in range(20):
    for j in range(9):
        pos = shareholders[i+1]['position'+str(j)]
        if pos == "":
            pos =0.0
        bigPositions.append(float(pos))
    positions.append(bigPositions)
    bigPositions = []

for i in range(20):
    currentPos = positions[i][0]
    institutionName = shareholders[i+1]['institutionName']
    if currentPos == 0.0:
        if (positions[i][8] != 0.0):
            positions[i][0] = positions[i][8]
        elif(positions[i][8] == 0.0):
            '''print('We are here')
            print(positions[i][1:8])
            print(positions[i])'''
            
            #print(positions[i])
            predictedPositions = predict_prices(dates, positions[i][1:8], 8)
            print(institutionName + ' : '+str(predictedPositions))