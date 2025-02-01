import utils
import sklearn
import numpy as np

# INPUT PARAMETERS
# - weight
# - bike_wgt
# - speed
# - slope
# - altitude
# - cadence
# - enhanced_speed
# - current_slope
# - heart_rate
# - estimated_power


def predict(tree_model, altitude, cadence, enhanced_speed, current_slope, heart_rate, distance, weight, bike_wgt):
    
    if not isinstance(tree_model, sklearn.tree._tree.Tree):
        tree_model = tree_model.tree_
    
    instance = [altitude,
                cadence,
                enhanced_speed,
                current_slope,
                heart_rate,
                distance,
                utils.compute_power(weight, bike_wgt, enhanced_speed, current_slope)]
    
    node = 0  # start from the root
    
    while tree_model.children_left[node] != -1:  # while not a leaf
        feature_index = tree_model.feature[node]
        threshold = tree_model.threshold[node]
        
        if instance[feature_index] <= threshold:
            node = tree_model.children_left[node]
        else:
            node = tree_model.children_right[node]
    
    # When a leaf is reached
    return np.float32(tree_model.value[node][0][0])