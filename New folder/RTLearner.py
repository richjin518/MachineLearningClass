import numpy as np  

class RTLearner(object):   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    This is a Linear Regression Learner. It is implemented correctly.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		     		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		     		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    def __init__(self, leaf_size=1, verbose=False):  		  	   		     		  		  		    	 		 		   		 		  
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
        #print("Input Data:")
        #print(data)
        self.DT = self.build_tree(data)        
        
        if self.verbose == True:
            print ("Tree built:")
            print (self.DT)
        

    def build_tree(self, data):
        num = len(data[0,:-1])
        
        #print("Test:")
        #print(data,data[:,-1],np.unique(data[:,-1]),np.unique(data[:,-1]).shape[0])
        #time.sleep(2)
        if data.shape[0] <= self.leafsize:
            return np.array([[-1, data[:,-1].mean(), "NA", "NA"]], dtype = object)
        if np.unique(data[:,-1]).shape[0] == 1:
            #print ("current data:", data[:,-1][0])
            return np.array([[-1, np.unique(data[0,-1])[0], "NA", "NA"]], dtype = object)
        else:
            
            index = np.random.randint(num) 
            SplitVal = np.median(data[:,index])
            #print (index, SplitVal)
            lefttree = self.build_tree(data[data[:,index]<=SplitVal])
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
                elif point[key] >= self.DT[i][1]:
                    i+=int(self.DT[i][3])
        print (result)
        return np.array(result)
	  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		     		  		  		    	 		 		   		 		  
    #print("the secret clue is 'zzyzx'")
    """
    data_x = np.array([(0.885,0.330,9.1),(0.725,0.390,10.9),(0.560,0.500,9.4),(0.735,0.57,9.8),(0.61,0.63,8.4),(0.26,0.63,11.8),(0.5,0.68,10.5),(0.32,0.78,10)])
    data_y = np.array([4,5,6,5,3,8,7,6])
    learner = RTLearner(1, False)
    tree = learner.add_evidence(data_x, data_y) 	
    points = np.array([(0.885,0.330,9.1),(0.725,0.390,10.9),(0.560,0.500,9.4)])
    query = learner.query(points)
    """
    