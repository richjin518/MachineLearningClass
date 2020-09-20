import numpy as np  
import time	

class DTLearner(object):   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    This is a Linear Regression Learner. It is implemented correctly.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		     		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		     		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    def __init__(self, leaf_size = 1, verbose=False):  		  	   		     		  		  		    	 		 		   		 		  
        """  		  	   		     		  		  		    	 		 		   		 		  
        Constructor method  		  	   		     		  		  		    	 		 		   		 		  
        """ 
        self.leafsize = leaf_size
        self.verbose = verbose
        self.DT = [] 		  	   		     		  		  		    	 		 		   		 		  
        # move along, these aren't the drones you're looking for  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    def author(self):  		  	   		     		  		  		    	 		 		   		 		  
        return "yzhang3407"  # replace tb34 with your Georgia Tech username  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    def add_evidence(self, data_x, data_y):  		  	   		     		  		  		    	 		 		   		 		  
        """  		  	   		     		  		  		    	 		 		   		 		  
        Add training data to learner  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
        :param data_x: A set of feature values used to train the learner  		  	   		     		  		  		    	 		 		   		 		  
        :type data_x: numpy.ndarray  		  	   		     		  		  		    	 		 		   		 		  
        :param data_y: The value we are attempting to predict given the X data  		  	   		     		  		  		    	 		 		   		 		  
        :type data_y: numpy.ndarray  		  	   		     		  		  		    	 		 		   		 		  
        """  

        data = np.column_stack((data_x, data_y))
        self.DT = self.build_tree(data)        
        
        if self.verbose == True:
            print ("Tree built:")
            print (self.DT)
        

    def build_tree(self, data):
        maxcorr = 0
        num = len(data[0,:-1])
        index = 0
        
        #print(np.sort(data[:,0].T))
        #time.sleep(10)
        #print(data[:10])
        #print(len(data))
        #if len(data) <= 4:
        #print("my", data)
        #time.sleep(1)
        
        #time.sleep(2)
        if data.shape[0] <= self.leafsize:
            return np.array([[-1, data[:, -1].mean(), "NA", "NA"]], dtype = object)
        #print("now: ", np.unique(data[:,-1]).shape[0])
        if np.unique(data[:,-1]).shape[0] == 1:
            return np.array([[-1, data[:, -1].mean(), "NA", "NA"]], dtype = object)
        else:
            #print(data[0,:-1], num)
            for i in range(num):
                curr_corr = np.corrcoef(data[:,i], data[:,-1], rowvar = False)
                #print(data[:,i], data[:,-1])
                #print(i, data[:, i], data[:,-1], curr_corr)
                if abs(curr_corr[-1,0]) > maxcorr: #or abs(abs(curr_corr[1,0]) - maxcorr) < 1e-09:
                    #print("in if 1 ", index, i, num)
                    #print(curr_corr, maxcorr)
                    maxcorr = abs(curr_corr[-1,0])
                    #print(curr_corr)
                    index = i
                   # print("in if 2 ", index, i)
                
                #print("inner", i, maxcorr, level)
            
          
            SplitVal = np.median(data[:,index])
            #print("final", index, maxcorr, SplitVal)

            #print("left")
            newdata = data[data[:,index]<=SplitVal]
            #print("unique: ", data.shape[0], len(np.unique(data[:,0])), data[:,0])
            if newdata.shape[0] == 0 or newdata.shape[0] == data.shape[0]:
                return np.array([[-1, data[:, -1].mean(), "NA", "NA"]], dtype = object) 

            lefttree = self.build_tree(data[data[:,index]<=SplitVal])
            #print("right")
            righttree = self.build_tree(data[data[:,index]>SplitVal])
            root = np.array([[index, SplitVal, 1, lefttree.shape[0] + 1]], dtype = object)
            return np.concatenate((root, lefttree, righttree))


    def query(self, points): 
        result = []
        for point in points:
            i = 0
            while True:
                key = int(self.DT[i][0])
                if key == -1:
                    result.append(self.DT[i][1])
                    break
                elif point[key] <= self.DT[i][1]:
                    i+=int(self.DT[i][2])
                elif point[key] > self.DT[i][1]:
                    i+=int(self.DT[i][3])
        #print (result)
        return np.array(result)
	  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		     		  		  		    	 		 		   		 		  
    print("the secret clue is 'zzyzx'")
   
    
    data_x = np.array([(0.885,0.330,9.1),(0.725,0.390,10.9),(0.560,0.500,9.4),(0.735,0.57,9.8),(0.61,0.63,8.4),(0.26,0.63,11.8),(0.5,0.68,10.5),(0.32,0.78,10)])
    data_y = np.array([4,5,6,5,3,8,7,6])
    learner = DTLearner(1, True)
    tree = learner.add_evidence(data_x, data_y) 	
    points = np.array([(0.885,0.330,9.1),(0.725,0.390,10.9),(0.560,0.500,9.4)])
    query = learner.query(points)
    
    
    
    

class DTLearner_2(object):
    
    def __init__(self, leaf_size = 1, tree = None, verbose = False):
        self.leaf_size = leaf_size
        self.tree = tree
        self.verbose = verbose
    
    def author(self):
        return 'yguan35'
    
    def add_evidence(self, Xtrain, Ytrain):
        self.tree = self.fit(Xtrain, Ytrain)
    
    def fit(self, Xtrain, Ytrain):
        if Xtrain.shape[0] <= self.leaf_size:
            return np.array([['leaf', Ytrain.mean(axis=0), None, None]])
        elif Ytrain.shape == Ytrain[Ytrain == Ytrain[0]].shape:
            return np.array([['leaf', Ytrain[0], None, None]])
        else:
            m, n = Xtrain.shape
            corr_max = 0
            var_index = None
            for i in range(n):
                corr = abs(np.corrcoef(Xtrain[:,i],Ytrain,rowvar = False)[-1,0])
                if corr > corr_max:
                    corr_max = corr
                    var_index = i
            sorted_index = np.argsort(Xtrain[:,var_index])
            Xtrain = Xtrain[sorted_index]
            Ytrain = Ytrain[sorted_index]
            splitval = np.median(Xtrain[:, var_index], axis = 0)
            
            if Xtrain[Xtrain[:,var_index]>splitval].shape[0] >0:
                lefttree = self.fit(Xtrain[Xtrain[:,var_index]<=splitval], Ytrain[Xtrain[:,var_index]<=splitval])
                righttree = self.fit(Xtrain[Xtrain[:,var_index]>splitval], Ytrain[Xtrain[:,var_index]>splitval])
                root = [[var_index, splitval, 1, lefttree.shape[0]+1]]
                return np.append(root, np.concatenate((lefttree,righttree), axis=0), axis=0)
            else:
                return np.array([['leaf', Ytrain.mean(axis=0), None, None]])
                        
    
    def query(self, Xtest):
        m, n = self.tree.shape
        j, k = Xtest.shape
        result = []
        for j in range(j):
            i = 0
            while i < m:
                if self.tree[i, 0] == 'leaf' and self.tree[i,2] is None and self.tree[i,3] is None:
                    result.append(self.tree[i,1])
                    break
                else:
                    if Xtest[j, int(self.tree[i,0])] <= self.tree[i,1]:
                        i += int(self.tree[i,2])
                    else:
                        i += int(self.tree[i,3])
        return np.array(result)
