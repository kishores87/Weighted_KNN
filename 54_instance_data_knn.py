from ReliefF import ReliefF
import numpy as np
from CBR_model import CBR_model
from pandas import read_excel   # Added new to compute max value for normalization



file_path = '54_instance_raw_data_credit_score.xlsx'
cols = [
    'X1'
    ,'X2'
    ,'X3'
    ,'CLASS'
    # ,'saliency_matrix'
]


# Read from input to learn the true max of X3 for scaling
_max_df = read_excel(file_path, header=0)
true_max_x3 = float(_max_df['X3'].max()) if 'X3' in _max_df.columns else 200.0
if not true_max_x3 or true_max_x3 <= 0:
    true_max_x3 = 200.0  # safe fallback


diff_fns = {
        'X1':
            lambda x, y: 0 if x == y else 1,
        'X2':
            lambda x, y: 0 if x == y else 1,
        'X3':
            #lambda x, y: 0 if x == y else 1
            #lambda x, y: abs(float(x) - float(y)) / 200.0
            lambda x, y: abs(float(x) - float(y)) / true_max_x3   # Normalize based on true max value
    }

sim_fns = {
        'X1':
            lambda x, y: 1 if x == y else 0,
        'X2':
            lambda x, y: 1 if x == y else 0,
        'X3':
            #lambda x, y: 1 if x == y else 0
            lambda x, y: 1.0 - (abs(float(x) - float(y)) / true_max_x3)  # Normalize based on true max value
    }

# Times Run
m = 10000

# Use k misses and hits
k = 1 

relief = ReliefF(diff_fns,file_path, cols, m, k)
weights = relief.calculate_weights()
print(weights)

# adding for traiggering purpose
print("ReliefF feature weights (for X1, X2, X3):", weights)


# Weight learned from ReliefF_function : 10000 iterations normalized

w = weights

#> 77.77%
#w = [0.3450572418958393, 0.3437714298686596, 0.3111713282355012]  
# Use k misses and hits
k = 5 #changed from 1 to 5 for better accuracy

cbr_model = CBR_model(sim_fns, file_path, cols,w,k)

cbr_model.test_accuracy()
print(w)