import numpy as np  

class BagLearner(object):   		     		  		  		    	 		 		   		 		  
	  	   		     		  		  		    	 		 		   		 		  
    def __init__(self, learner, kwargs, bags, boost, verbose):  		  	   		     		  		  		    	 		 		   		 		  
        
        self.learners = []
        self.kwargs = kwargs
        for i in range(0, bags):
            self.learners.append(learner(**kwargs))
        self.bags = bags
        self.boost = boost
        self.verbose = verbose		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    def author(self):  		  	   		     		  		  		    	 		 		   		 		  
        return "yzhang3407"  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    def add_evidence(self, data_x, data_y):
        for learner in self.learners:
            index = np.random.randint(data_x.shape[0], size = data_x.shape[0])
            learner.add_evidence(data_x[index], data_y[index])


    def query(self, points): 
        result = 0.0
        for learner in self.learners:
            result += learner.query(points)
        return result / self.bags

	  	   		     		  		  		    	 		 		   		 		  