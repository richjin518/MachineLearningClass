import LinRegLearner as lrl
import BagLearner as bl
import numpy as np
class InsaneLearner(object):	     		  		  		    	 		 		   		 		  
    def __init__(self, verbose = False):  		  	   		     		  		  		    	 		 		   		 		  
        self.learners = []
        self.verbose = verbose
        self.num = 20
        for i in range(self.num):
            self.learners.append(bl.BagLearner(lrl.LinReglearner, {}, 20, False, False))
    def author(self):  		  	   		     		  		  		    	 		 		   		 		  
        return "yzhang3407"  
    def add_evidence(self, data_x, data_y):
        for i in range(self.num):
            self.learners[i].add_evidence(data_x, data_y)
    def query(self, points):
        result = []
        for i in range(self.num):
            result.append(self.learners[i].query(points))
        return np.mean(result, axis = 0)	     		  		  		    	 		 		   		 		  

	  	   		     		  		  		    	 		 		   		 		  