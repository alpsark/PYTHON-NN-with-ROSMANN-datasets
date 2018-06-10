import csv
import io
import librosa as lr
import matplotlib.pyplot as plt
import numpy as np

plt.interactive(False)

#data_saved ,Sales_precision, Customers_precision ayarlanmalı
print('lets go')
#import datasets
keys = []
X = []  # feature vector array
Y_Sales = []  # feature target array
Y_Customers = []  # feature target array
target_sales_row= 0
target_customers_row = 0
feature_row = []
Sales_precision = 1000
Customers_precision = 100

def import_csv( filename ):
    # import datasets
    global keys
    global X  # feature vector array
    global Y_Sales  # feature target array
    global Y_Customers  # feature target array
    global target_sales_row
    global target_customers_row
    global feature_row
    with open(filename,'rU') as f:
        reader = csv.reader(f)
        Keyget = False
        first = True
        rownum = 1
        for row in reader:
            print(rownum)
            rownum +=1
            if Keyget:
                for i, value in enumerate(row[0].split(";")):
                    if "Store Type" in keys[i]:
                        #decider[keys[i]].append(int(value))
                        feature_row.append(int(value))
                    elif "Sales" in keys[i]:
                        target_sales_row=int(round(int(value)/Sales_precision))
                    elif "Customers" in keys[i]:
                        target_customers_row = int(round(int(value)/Customers_precision))
                    elif "Store" not in keys[i]:
                        # decider[keys[i]].append(int(value))
                        feature_row.append(int(value))
                    else:
                        for x in range(1,1116):
                            if(int(value) == x):
                                feature_row.append(1)
                            else:
                                feature_row.append(0)
                #feature_row = np.transpose(feature_row)
                #target_row = np.transpose(target_row)
                # if first:
                #     X =np.concatenate([X,np.array(feature_row).T])
                #     X = X.reshape((1,X.size))
                #     first = False
                # else:
                #     X =np.concatenate([X,np.array(feature_row).reshape((1,np.array(feature_row).size))])
                # X = np.concatenate([X, np.array(feature_row).T])
                # Y_Sales = np.append(Y_Sales,target_sales_row)
                # Y_Customers =np.append(Y_Customers,target_customers_row)
                X.append(np.array(feature_row).reshape((1,np.array(feature_row).size))[0])
                Y_Sales.append( target_sales_row)
                Y_Customers.append(target_customers_row)
                feature_row = []
            else:
                # first row
                Keyget = True
                for stri in row[0].split(";"):
                    if(stri == "Store"):
                        for x in range(1,1116):
                            keys.append(stri+str(x))
                    else:
                        keys.append(stri)
# you now have a column-major 2D array of your file.

import pickle
def save_imported_data( filename ):
    global X  # feature vector array
    global Y_Sales  # feature target array
    global Y_Customers  # feature target array
    # Saving the objects:
    with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([X, Y_Sales, Y_Customers], f)

def load_imported_data(filename):
    global X  # feature vector array
    global Y_Sales  # feature target array
    global Y_Customers  # feature target array
    # Getting back the objects:
    with open(filename,'rb') as f:  # Python 3: open(..., 'rb')
        X, Y_Sales, Y_Customers = pickle.load(f)

def save_trained_data( filename , X_Sales_train, X_Sales_test, Y_Sales_train, Y_Sales_test , nn):
    # Saving the objects:
    with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([X_Sales_train, X_Sales_test, Y_Sales_train, Y_Sales_test, nn], f)


def load_trained_data(filename):
    # Getting back the objects:
    with open(filename, 'rb') as f:  # Python 3: open(..., 'rb')
       return pickle.load(f)

data_saved = False
if  data_saved:
    load_imported_data('objs.pkl')
    print("Data load complete")
else:
    import_csv('traindata9.csv')
    print("Data pull complete")
    save_imported_data('objs .pkl')
    print("Data save complete")

# machine learning part
import Machine_learn
from sklearn import cross_validation
nn_Sales = Machine_learn.NN_1HL(hidden_layer_size=50)
nn_Customers = Machine_learn.NN_1HL(hidden_layer_size=50)

# import sklearn.datasets as datasets
# iris = datasets.load_iris()
# x = iris.data
# y = iris.target

#file = open("X.txt", "w")
#for item in decider.X:
#   file.write("%s\n" % item)
#file.close()

#file = open("Y.txt", "w")
#file.write(decider.Y)
#file.close()

# X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(x, y, test_size=0.40)
X_Sales_train, X_Sales_test, Y_Sales_train, Y_Sales_test = cross_validation.train_test_split(np.array(X), np.array(Y_Sales),stratify=Y_Sales, test_size=0.20)
X_customers_train, X_customers_test, Y_customers_train, Y_customers_test = cross_validation.train_test_split(np.array(X), np.array(Y_Customers),stratify=Y_Customers, test_size=0.20)

print("Test and Train data seperated")

num_labels = len(set(Y_Sales_train))
if(num_labels == max(Y_Sales_train)+1):
    print("Sales data ok")
else:
    print("Sales data error")

num_labels = len(set(Y_customers_train))
if(num_labels == max(Y_customers_train)+1):
    print("Customer data ok")
else:
    print("Customer data error")



nn_Sales.fit(X_Sales_train, Y_Sales_train)

from matplotlib import pyplot as plt
plt.figure(1)

print("Sales Trained")

data_saved = False
if  data_saved:
    X_Sales_train, X_Sales_test, Y_Sales_train, Y_Sales_test, nn_Sales = load_trained_data('objs_sales.pkl')
    print("Data load complete")
else:
    #save_trained_data('objs_sales.pkl',X_Sales_train, X_Sales_test, Y_Sales_train, Y_Sales_test,nn_Sales)
    print("Data save complete")



from sklearn.metrics import accuracy_score
predictions_Sales =  nn_Sales.predict(X_Sales_test)
score_Sales=accuracy_score(Y_Sales_test, nn_Sales.predict(X_Sales_test))
print("accuracy Sales: " , score_Sales )#



## The line / model
plt.subplot(221)
plt.scatter(Y_Sales_test, predictions_Sales)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Sales')

nn_Customers.fit(X_customers_train, Y_customers_train)

print("Customers Trained")

data_saved = False
if  data_saved:
    X_customers_train, X_customers_test, Y_customers_train, Y_customers_test, nn_Customers = load_trained_data('objs_customers.pkl')
    print("Data load complete")
else:
    #save_trained_data('objs_customers.pkl', X_customers_train, X_customers_test, Y_customers_train, Y_customers_test ,nn_Customers)
    print("Data save complete")

predictions_Customers =nn_Customers.predict(X_customers_test)
score_Customers=accuracy_score(Y_customers_test,predictions_Customers )

## The line / model
plt.subplot(222)
plt.scatter(Y_customers_test, predictions_Customers)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Customers')


print("accuracy Customers: " , score_Customers )#

plt.show()