import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error


def slice_parquet_activity(zwift_data, parquet_data, activity_name, col_subset = True):
    flnm = zwift_data[zwift_data["Activity Name"] == activity_name]["Filename"].values
    if len(flnm) > 0:
        out = parquet_data[parquet_data["FileName"] == flnm[0]].sort_values("timestamp", ignore_index = True)
        if col_subset:
            return out[['FileName', 'lap', 'distance',
                        'speed', 'enhanced_speed', 'heart_rate', 'power', 'cadence',
                        'grade', 'slope', 'current_slope',
                        'altitude',  'enhanced_altitude', 'filtered_altitude',
                        'latitude', 'position_lat', 'longitude', 'position_long',
                        'temperature', 'time_from_course', 'timestamp',
                        'Activity Name', 'Athlete Weight', 'Bike Weight',
                        'Weighted Average Power']]
        else:
            return out
    else:
        print("Activity name not found")


def ReLU(x):
    return x * (x > 0)


def compute_power(weight, bike_wgt, speed, slope):
    
    # 1) rolling resistance
    C_rr = np.random.uniform(0.0025, 0.004) # coefficient of rolling resistance
    m = weight + bike_wgt # total mass
    g = 9.81

    P_rr = C_rr * m * speed * g

    # 2) Aerodynamic drag
    rho = 1.2 # air density
    C_d = 0.8 # drag coefficient
    A = 0.4 # frontal area

    P_aero = 0.5 * rho * C_d * A * np.power(speed, 3)

    # 3) Gradient resistance
    slope_rad = slope * (np.pi/180)
    P_gr = m * g * speed * np.sin(slope_rad)

    eta = 0.95 # drivetrain efficiency

    P = (P_rr + P_aero + P_gr) / eta

    return max(np.divide(P, weight), 0)


def data_cleaning(df, weight, bike_wgt, scaling = True):
    Xy = df[["timestamp", "altitude", "cadence", "enhanced_speed", "current_slope", "heart_rate", "distance", "power"]].copy()
    # heart_rate w.r.t. rest HR and max HR
    # Xy["heart_rate"] = 100 * (Xy['heart_rate']/ (max_hr-rest_hr))
    Xy["heart_rate"] = Xy["heart_rate"].bfill()
    Xy["current_slope"] = Xy["current_slope"].fillna(0)
    Xy.loc[(Xy["cadence"].isnull()) & (Xy["enhanced_speed"] == 0), "cadence"] = 0
    Xy["distance"] = Xy["distance"] / 1000
    Xy["power"] = Xy["power"].fillna(0)
    Xy["power"] = Xy["power"] / weight

    Xy = Xy.dropna(how = "any").reset_index(drop = True)

    Xy["estimated_power"] = Xy.apply(lambda row: compute_power(weight = weight, bike_wgt = bike_wgt,
                                                               speed = row["enhanced_speed"],
                                                               slope = row["current_slope"]),
                                     axis = 1)

    Xy = Xy.sort_values("timestamp", ignore_index = True)
    
    Xy["heart_rate_5s"] = Xy["heart_rate"].shift(5).bfill()
    Xy["heart_rate_10s"] = Xy["heart_rate"].shift(10).bfill()
    Xy["heart_rate_15s"] = Xy["heart_rate"].shift(15).bfill()
    Xy["heart_rate_20s"] = Xy["heart_rate"].shift(20).bfill()

    Xy["cadence"] = Xy["cadence"].fillna(0)
    Xy["cadence_5s"] = Xy["cadence"].shift(5).bfill()
    Xy["cadence_10s"] = Xy["cadence"].shift(10).bfill()
    Xy["cadence_15s"] = Xy["cadence"].shift(15).bfill()
    Xy["cadence_20s"] = Xy["cadence"].shift(20).bfill()

    Xy["enhanced_speed_5s"] = Xy["enhanced_speed"].shift(5).bfill()
    Xy["enhanced_speed_10s"] = Xy["enhanced_speed"].shift(10).bfill()
    Xy["enhanced_speed_15s"] = Xy["enhanced_speed"].shift(15).bfill()
    Xy["enhanced_speed_20s"] = Xy["enhanced_speed"].shift(20).bfill()

    Xy = Xy.fillna(0)

    Xy = Xy.drop("timestamp", axis = 1)
    
    X_train, X_test, y_train, y_test = train_test_split(Xy.drop("power", axis = 1), Xy["power"],
                                                        test_size = 0.3, random_state = 42, shuffle = True)
    
    dict_out = {}
    dict_out["X_train"] = X_train
    dict_out["X_test"] = X_test
    dict_out["y_train"] = y_train
    dict_out["y_test"] = y_test
    
    if scaling:
        std_scaler = StandardScaler()
        std_scaler.fit(X_train)
        X_train_scaled = std_scaler.transform(X_train)
        X_test_scaled = std_scaler.transform(X_test)
        
        dict_out["scaler_mean"] = std_scaler.mean_
        dict_out["scaler_var"] = std_scaler.var_
        
        
    return dict_out


def random_forest_train(X_train, y_train):

    # Initialize a Decision Tree Classifier
    d3 = DecisionTreeRegressor(random_state = 101, criterion = "absolute_error")
    d3.fit(X_train, y_train)

    # Compute impurities to look for an adequate pruning term
    path = d3.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    # Plot the impurity level given ccp_alpha
#     plt.plot(ccp_alphas[:-1], impurities[:-1], marker = "o", drawstyle = "steps-post")
#     plt.xlabel("effective alpha")
#     plt.ylabel("total impurity of leaves")
#     plt.title("Total Impurity vs effective alpha for training set")
#     plt.show()

    # Tree Pruning (we are overfitting - Accuracy: 100% in training)
    tree_params = {"ccp_alpha":[ccp_alphas[impurities < impty].max()
                                for impty in np.linspace(0.1, np.round(impurities.max()/2, 3),
                                                        num = 5)]}

    tree_grid = GridSearchCV(DecisionTreeRegressor(random_state = 101, criterion = "absolute_error"),
                            tree_params, cv = 3, n_jobs = 6, verbose = 10,
                            return_train_score = True)
    tree_grid.fit(X_train, y_train)

    return tree_grid.best_estimator_
          
          
def scoring(model, name, X_test, y_test, weight, plot = True, ax = None):
    # Show the results
    print("[{}]".format(name.upper()))
    print("\tWith ReLU MAE: {:.2f} W/kg ({:.2f} W)".format(mean_absolute_error(y_test, ReLU(model.predict(X_test))),
                                               mean_absolute_error(y_test, ReLU(model.predict(X_test))) * weight))

    print("\tWithout ReLU MAE: {:.2f} W/kg ({:.2f} W)".format(mean_absolute_error(y_test, model.predict(X_test)),
                                               mean_absolute_error(y_test, model.predict(X_test)) * weight))
    print()
    if plot:
        if ax is not None:
            ax.plot(sorted(y_test)[::-1], label = "REAL POWER", lw = 5)
            ax.plot(sorted(ReLU(model.predict(X_test)))[::-1], label = "PREDICTED + RELU", lw = 5)
            ax.plot(sorted(model.predict(X_test))[::-1], label = "PREDICTED", alpha = .8, ls = "--")
            ax.set_title("[{}] Test power curve".format(name.upper()))
            ax.legend()
        else:
            plt.figure(figsize = (16, 5))
            plt.plot(sorted(y_test)[::-1], label = "REAL POWER", lw = 5)
            plt.plot(sorted(ReLU(model.predict(X_test)))[::-1], label = "PREDICTED + RELU", lw = 5)
            plt.plot(sorted(model.predict(X_test))[::-1], label = "PREDICTED", alpha = .8, ls = "--")
            plt.title("[{}] Test power curve".format(name.upper()))
            plt.legend()
            plt.show()
            
            
def power_curve_comparison(models_dict, data_dict, weight):
    
    names = [n.upper() for n in models_dict.keys()]
    models = list(models_dict.values())
    
    fig, ax = plt.subplots(2, 2, figsize = (16, 9))
    for g in range(len(names)):
        scoring(model = models[g],
                name = names[g],
                X_test = data_dict["X_test"],
                y_test = data_dict["y_test"],
                weight = weight,
                plot = True,
                ax = ax[g // 2, g % 2])

    ax[1, 1].plot(weight * np.array(sorted(data_dict["y_test"])[::-1]), label = "data")
    for mm in range(len(names)):
        ax[1, 1].plot(85.5 * np.array(sorted(models[mm].predict(data_dict["X_test"]))[::-1]),
                              label = names[mm])
        ax[1, 1].legend()
        ax[1, 1].set_title("COMPARISON")