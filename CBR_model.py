'''
CBR implementation
'''

from pandas import read_excel
import numpy as np
from sklearn.model_selection import LeaveOneOut

class CBR_model:
    def __init__(self, sim_fns, file_path, cols, w, k=3):
        self.sim_fns = sim_fns
        self.file_path = file_path
        self.cols = cols
        self.k = k

        #excel = read_excel(file_path, header=0, usecols=range(4,100))
        excel = read_excel(file_path, header=0)
        #data = excel[cols].sample(frac=1).values
        data = excel[cols].values
        
        # X is all features, y is classified value
        self.X = data[:, 0:-1]
        self.y = data[:, -1]

        import numpy as np
        print("Unique CLASS values found:", np.unique(self.y))

        self.no_of_instances = self.X.shape[0]
        self.no_of_features = self.X.shape[1]
        # Initialize weights to 0
        self.attrib_weights = [0] * self.no_of_features
        self.w = w

    def predict(self, train_index, test_index):
        sim_val_arr = []

        for ti in train_index:
            sim_val = 0
            #print(X[ti], y[ti], X[test_index[0]])

            for idx, Xi in enumerate(self.X[ti]):
                Ti = self.X[test_index[0]][idx]
                attrib = self.cols[idx]
                # Similarity value from the sim functions multiplied by the weight
                sim_val += self.sim_fns[attrib](Xi,Ti) * self.w[idx]

            # For each instance append similarity score
            sim_val_arr.append(sim_val)
        #print(sim_val_arr)

        sim_val_arr = np.array(sim_val_arr)

        nearest_idx = sim_val_arr.argsort()[-self.k:][::-1]

        #print(nearest_idx)

        k_nearest_solutions = [self.y[l] for l in nearest_idx]
        #print(k_nearest_match_cases, k_nearest_solutions)
        return k_nearest_solutions
    

    def test_accuracy(self):
        correct_total = 0

        # Leave one out cross validation
        for train_index, test_index in LeaveOneOut().split(self.X):
            predicted = self.predict(train_index,test_index)
            actual = self.y[test_index[0]]
            is_correct = 1 if actual in predicted else 0
            print(actual,predicted, is_correct)
            correct_total+=is_correct

        accuracy = correct_total / self.y.shape[0]

        ## Added this for debugging purposes
        print("Unique CLASS values found in dataset:", np.unique(self.y))
        
        print("Accuracy for {} neighbors is {}%: ".format(self.k,accuracy*100))


