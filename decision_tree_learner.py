import numpy as np  
import time		  	   		     		  		  		    	 		 		   		 		  

from sklearn.preprocessing import normalize	   		 		  

class DTLearner(object):   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    This is a Linear Regression Learner. It is implemented correctly.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		     		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		     		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    def __init__(self, leaf_size, verbose):  		  	   		     		  		  		    	 		 		   		 		  
        """  		  	   		     		  		  		    	 		 		   		 		  
        Constructor method  		  	   		     		  		  		    	 		 		   		 		  
        """ 
        self.leafsize = leaf_size
        self.verbose = verbose
        self.DT = [] 		  	   		     		  		  		    	 		 		   		 		  
        # move along, these aren't the drones you're looking for  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    def author(self):  		  	   		     		  		  		    	 		 		   		 		  
        return "yzhang3407"  # replace tb34 with your Georgia Tech username  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    def add_evidence(self, data_x, data_y, level):  		  	   		     		  		  		    	 		 		   		 		  
        """  		  	   		     		  		  		    	 		 		   		 		  
        Add training data to learner  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
        :param data_x: A set of feature values used to train the learner  		  	   		     		  		  		    	 		 		   		 		  
        :type data_x: numpy.ndarray  		  	   		     		  		  		    	 		 		   		 		  
        :param data_y: The value we are attempting to predict given the X data  		  	   		     		  		  		    	 		 		   		 		  
        :type data_y: numpy.ndarray  		  	   		     		  		  		    	 		 		   		 		  
        """  
        data = np.column_stack((data_x, data_y))

        print(data)
        self.DT = self.build_tree(data, level)        
        
        if self.verbose == True:
            print ("Tree built successfully")
            print (self.DT)
        

    def build_tree(self, data, level):
        #print(level)
        maxcorr = 0
        num = len(data[0,:-1])
        index = 0
        
        #print(data)
        #time.sleep(2)
        if data.shape[0] == 1:
            return np.array([["leaf", data[:,-1][0], "NA", "NA"]], dtype = object)
        if np.unique(data[:,-1]).shape[0] == 1:
            return np.array([["leaf", data[0,-1][0], "NA", "NA"]], dtype = object)
        else:
            #print(data[0,:-1], num)

            for i in range(num):
                curr_corr = np.corrcoef(data[:,i], data[:,-1])
                #print(data[:,i], data[:,-1])
                #print(i, data[:, i], data[:,-1], curr_corr)
                if abs(curr_corr[1,0]) - maxcorr > 1e-09 or abs(abs(curr_corr[1,0]) - maxcorr) < 1e-09:
                    print("in if 1 ", index, i, num)
                    #print(curr_corr, maxcorr)
                    maxcorr = abs(curr_corr[1,0])
                    index = i
                    print("in if 2 ", index, i)
                
                print("inner", i, maxcorr, level)
            print("final", index, maxcorr, level)
                    
            SplitVal = np.median(data[:,index])
            lefttree = self.build_tree(data[data[:,index]<SplitVal], level + 1)
            righttree = self.build_tree(data[data[:,index]>=SplitVal], level + 1)
            root = np.array([[index, SplitVal, 1, lefttree.shape[0] + 1]], dtype = object)
            return np.concatenate((root, lefttree, righttree))


    def query(self, points): 
        result = []
        for point in points:
            i = 0
            while True:
                key = int(self.DT[i][0])
                if key == 'Leaf':
                   result.append(self.DT[i][1])
                elif point[key] <= self.DT[i][1]:
                    i+=int(self.DT[i][2])
                elif point[key] >= self.DT[i][1]:
                    i+=int(self.DT[i][3])
        return np.array(result)
	  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		     		  		  		    	 		 		   		 		  
    print("the secret clue is 'zzyzx'")
    data_x = np.array([(0.885,0.330,9.1),(0.725,0.390,10.9),(0.560,0.500,9.4),(0.735,0.57,9.8),(0.61,0.63,8.4),(0.26,0.63,11.8),(0.5,0.68,10.5),(0.32,0.78,10)])
    data_y = np.array([4,5,6,5,3,8,7,6])
    learner = DTLearner(1, True)
    level = 0
    tree = learner.add_evidence(data_x, data_y, level) 	
    