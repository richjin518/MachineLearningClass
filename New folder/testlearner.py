""""""  		  	   		     		  		  		    	 		 		   		 		  
"""  		  	   		     		  		  		    	 		 		   		 		  
Test a learner.  (c) 2015 Tucker Balch  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		     		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		     		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		     		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		     		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		     		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		     		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		     		  		  		    	 		 		   		 		  
or edited.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		     		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		     		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		     		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		     		  		  		    	 		 		   		 		  
"""  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
import math  		  	   		     		  		  		    	 		 		   		 		  
import sys  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
import numpy as np  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as it
import matplotlib.pyplot as plt
  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		     		  		  		    	 		 		   		 		  
    if len(sys.argv) != 2:  		  	   		     		  		  		    	 		 		   		 		  
        print("Usage: python testlearner.py <filename>")  		  	   		     		  		  		    	 		 		   		 		  
        sys.exit(1)  	
    #print(sys.argv[1])
    inf = open(sys.argv[1])  
    if sys.argv[1] == "Data/Istanbul.csv":
        next(inf)
        data = np.array([list(map(float, s.strip().split(",")[1:])) for s in inf.readlines()] )  	    
    else:
        data = np.array( [list(map(float, s.strip().split(","))) for s in inf.readlines()] )  	
  	   		     		  		  		    	 		 		   		 		  
    # compute how much of the data is training and testing  		  	   		     		  		  		    	 		 		   		 		  
    train_rows = int(0.6 * data.shape[0])  		  	   		     		  		  		    	 		 		   		 		  
    test_rows = data.shape[0] - train_rows  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    # separate out training and testing data  		  	   		     		  		  		    	 		 		   		 		  
    train_x = data[:train_rows, 0:-1]  	
    #print(train_x[:10])	  	   		     		  		  		    	 		 		   		 		  
    train_y = data[:train_rows, -1]
    #print(train_y[:10])	  	   		     		  		  		    	 		 		   		 		  
    test_x = data[train_rows:, 0:-1]  	
    #print(test_x[:10])	  	   		     		  		  		    	 		 		   		 		  
    test_y = data[train_rows:, -1]  	
    #print(test_y[:10])	  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    print(f"{test_x.shape}")  		  	   		     		  		  		    	 		 		   		 		  
    print(f"{test_y.shape}")	  	   		     		  		  		    	 		 		   		 		  

    """
    # Experiment 1  
    num = 50
    leafsize = [i for i in range(1,num+1)]
    rsme_train = []
    rsme_test = []
    for i in leafsize:
        learner = dt.DTLearner(leaf_size = i, verbose = False)
        learner.add_evidence(train_x, train_y)
        pred_train = learner.query(train_x)
        rsme_train.append(math.sqrt(((train_y - pred_train) ** 2).sum() / train_y.shape[0]))
        pred_test = learner.query(test_x)
        rsme_test.append(math.sqrt(((test_y - pred_test) ** 2).sum() / test_y.shape[0]))
    plt.figure()
    plt.plot(leafsize, rsme_train, marker='o', markersize=3, label = 'Train dataset' )  
    plt.plot(leafsize, rsme_test, marker='o', markersize=3, label = 'Test dataset' ) 
    plt.title('Relationship between RSME and leafsize for Decision Tree Learner')
    plt.xlabel('Leaf Size')
    plt.ylabel('RSME')
    plt.grid(True, 'both')
    plt.legend(loc = 0)
    plt.savefig('Experiment1.png')
    #plt.show()
    """

    # Experiment 2
    num = 50
    leafsize = [i for i in range(1,num+1)]
    rsme_train = []
    rsme_test = []
    num_bags = 20
    for i in leafsize:
        learner = bl.BagLearner(learner = dt.DTLearner, kwargs = {'leaf_size':i}, bags = num_bags, boost = False, verbose = False)
        print(i)
        learner.add_evidence(train_x, train_y)
        pred_train = learner.query(train_x)
        rsme_train.append(math.sqrt(((train_y - pred_train) ** 2).sum() / train_y.shape[0]))
        pred_test = learner.query(test_x)
        rsme_test.append(math.sqrt(((test_y - pred_test) ** 2).sum() / test_y.shape[0]))
    plt.figure()
    plt.plot(leafsize, rsme_train, marker='o', markersize=3, label = 'Train dataset' )  
    plt.plot(leafsize, rsme_test, marker='o', markersize=3, label = 'Test dataset' ) 
    plt.title('Relationship between RSME and leafsize for Bagging Learner')
    plt.xlabel('Leaf Size')
    plt.ylabel('RSME')
    plt.grid(True, 'both')
    plt.legend(loc = 0)
    plt.savefig('Experiment2.png')
    plt.show()    
"""		  	   		     		  		  		    	 		 		   		 		  
    # create a learner and train it  		  	   		     		  		  		    	 		 		   		 		  
    learner = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner  		  	   		     		  		  		    	 		 		   		 		  
    learner.add_evidence(train_x, train_y)  # train it  		  	   		     		  		  		    	 		 		   		 		  
    print(learner.author())  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    # evaluate in sample  		  	   		     		  		  		    	 		 		   		 		  
    pred_y = learner.query(train_x)  # get the predictions  		  	   		     		  		  		    	 		 		   		 		  
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])  		  	   		     		  		  		    	 		 		   		 		  
    print()  		  	   		     		  		  		    	 		 		   		 		  
    print("In sample results")  		  	   		     		  		  		    	 		 		   		 		  
    print(f"RMSE: {rmse}")  		  	   		     		  		  		    	 		 		   		 		  
    c = np.corrcoef(pred_y, y=train_y)  		  	   		     		  		  		    	 		 		   		 		  
    print(f"corr: {c[0,1]}")  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    # evaluate out of sample  		  	   		     		  		  		    	 		 		   		 		  
    pred_y = learner.query(test_x)  # get the predictions  		  	   		     		  		  		    	 		 		   		 		  
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])  		  	   		     		  		  		    	 		 		   		 		  
    print()  		  	   		     		  		  		    	 		 		   		 		  
    print("Out of sample results")  		  	   		     		  		  		    	 		 		   		 		  
    print(f"RMSE: {rmse}")  		  	   		     		  		  		    	 		 		   		 		  
    c = np.corrcoef(pred_y, y=test_y)  		  	   		     		  		  		    	 		 		   		 		  
    print(f"corr: {c[0,1]}")  	
"""	  	   		     		  		  		    	 		 		   		 		  
