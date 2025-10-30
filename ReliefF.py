'''
ReliefF implementation from: https://doi.org/10.1023/A:1025667309714
'''

from pandas import read_excel
import numpy as np
from random import randrange
from math import exp

class ReliefF:

    def __init__(self, diff_fns, file_path, cols, m=None, k=3, range_num=100):
        self.diff_fns = diff_fns
        self.file_path = file_path
        self.cols = cols
        self.m = m if m is not None else 10  # Ensure m has a default value
        self.k = k

        # Validate range_num against the actual number of columns
        #excel = read_excel(file_path, header=0, usecols=range(4,100))
        excel = read_excel(file_path, header=0)
        range_num = min(range_num, excel.shape[1])  # Adjust range_num if necessary

        # Ensure cols are valid
        if not all(col in excel.columns for col in cols):
            raise ValueError("Some columns in 'cols' are not present in the Excel file.")

        data = excel[cols].sample(frac=1).values

        # Ensure data is not empty
        if data.size == 0:
            raise ValueError("No data found for the specified columns.")

        # X is all features, y is classified value
        self.X = data[:, 0:-1]
        self.y = data[:, -1]

        if self.X.shape[0] < self.k:
            raise ValueError("Not enough instances in the dataset for the specified 'k'.")

        print(self.X)
        print(self.y)

        self.no_of_instances = self.X.shape[0]
        self.no_of_features = self.X.shape[1]
        # Initialize weights to 0
        self.attrib_weights = [0] * self.no_of_features
        

    def find_nearest_hits(self,X, y, ins, k):
        """
        Gets the nearest k hit indexes
        X: feature matrix
        y: value vector
        ins: the instance index
        k: number of k nearest hits
        """
        hit_idx = []
        target = y[ins]

        for i in range(X.shape[0]):
            if y[i] == target and i != ins:
                hit_idx.append(i)
        
        diff_idx = [abs(x - ins) for x in hit_idx]
        nearest_hit_idx = np.argsort(diff_idx)[:k]
        
        nearest_hits = [hit_idx[l] for l in nearest_hit_idx]
        return nearest_hits

    def find_nearest_misses(self, X, y, ins, k, C):
        """
        Gets the nearest k miss indexes
        X: feature matrix
        y: value vector
        ins: the instance index
        k: number of k nearest hits
        C: Class which is not class of Ri
        """
        
        miss_idx = []
        target = y[ins]
        for i in range(X.shape[0]):
            if y[i] == C:
                miss_idx.append(i)
        #hit_idx.sort()
        
        diff_idx = [abs(x - ins) for x in miss_idx]
        
        nearest_miss_idx = np.argsort(diff_idx)[:k]
        
        #print(nearest_hit_idx)
        
        nearest_miss = [miss_idx[l] for l in nearest_miss_idx]
        return nearest_miss
    
    def prior(self, C, Ri_class):
        """
        Gets the prior probability
        C: class 1
        Ri_class: class 2
        """
        y_list = self.y.tolist()
        p_C = y_list.count(C) / self.no_of_instances
        p_Ri_class = y_list.count(Ri_class) / self.no_of_instances
        prior = p_C / (1-p_Ri_class)
        return prior
    
    def diff(self, A, I1, I2):
        """
        Gets the diff value based on lambdas defined in diff_fns
        """
        return self.diff_fns[A](I1,I2)

    def normalize_weights(self, w_arr):
        """
        Normalizes the weights calculated
        """
        # Add a check for numerical stability
        if max(w_arr) - min(w_arr) == 0:
            return [1 / len(w_arr)] * len(w_arr)  # Avoid division by zero
        norm = [(x - min(w_arr)) / (max(w_arr) - min(w_arr)) for x in w_arr]
        return norm

    
    def calculate_weights(self):
        """
        Calculates the weights for each attributes from ReliefF method
        """
        # Validate diff_fns contains all required keys
        for A in range(self.no_of_features):
            if self.cols[A] not in self.diff_fns:
                raise ValueError(f"Missing diff function for attribute: {self.cols[A]}")

        # Weight calculation
        for i in range(self.m):
            nearest_misses_arr = []
            # ALl avaialble classes
            classes = set(self.y)
            classes = list(classes)
            
            ins = randrange(self.no_of_instances)
            Ri = self.X[ins]
            #print(Ri)
            
            Ri_class = self.y[ins]
            # Find k nearest hits
            nearest_hits = self.find_nearest_hits(self.X,self.y,ins,self.k)
            
            # Remove the current class, need unmatched classes for misses
            classes.remove(self.y[ins])
            
            # Iterating through classes to find the misses
            for c in classes:
                # Find k nearest misses
                nearest_misses = self.find_nearest_misses(self.X,self.y,ins,self.k,c )
                
                nearest_misses_arr.append(nearest_misses)
            
            # Calculate nearest hits calculation
            for A in range(self.no_of_features):
                hit_calc = 0
                attrib = self.cols[A]        
                for j in range(self.k):
                    #print('nearest_hits',nearest_hits)
                    Hj = self.X[nearest_hits[j]]
                    hit_calc += self.diff(attrib,Ri[A],Hj[A]) / (self.m*self.k)
                    #print(attrib,Ri[A],Hj[A],hit_calc)
                    
            # Calculate nearest misses calculation
                miss_calc = 0
                                
                for idx, nearest_miss in enumerate(nearest_misses_arr):
                    miss_class = classes[idx]
                    #print(miss_class, nearest_miss)
                    attrib = self.cols[A]        
                    for j in range(self.k):
                        #print('nearest_miss',nearest_miss)
                        Mj = self.X[nearest_miss[j]]
                        miss_calc += self.diff(attrib,Ri[A],Mj[A]) / (self.m*self.k)
                    
                    #print("miss_class,Ri_class",miss_class,Ri_class)
                    miss_calc = miss_calc * self.prior(miss_class,Ri_class)
                    
                # Update weight of the atttribute A
                
                self.attrib_weights[A] = self.attrib_weights[A] - hit_calc + miss_calc
            
            print("attrib_weights", self.attrib_weights)

        normalized_weights = self.normalize_weights(self.attrib_weights)
        #print(normalized_weights)
        return normalized_weights