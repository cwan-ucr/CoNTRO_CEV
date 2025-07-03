import numpy as np
import pickle

# Load model and scaler based on EV type
EV = 'off'  # or 'on'
if EV == 'off':
    MinMaxScaler = pickle.load(open("controller/Eco2.0Accel-1TO1AccelStep025ETEDScaler_20.p", "rb"))
    nextDesiredVelocityRegressor = pickle.load(open("controller/Eco2.0Accel-1TO1AccelStep025ETEDRegressor_20.p", "rb"))
else:
    MinMaxScaler = pickle.load(open("controller/ELECTRIC20ThinPlate-1TO1AccelStep025ExtraDistanceExtraTimeScaler.p", "rb"))
    nextDesiredVelocityRegressor = pickle.load(open("controller/ELECTRIC20ThinPlate-1TO1AccelStep025ExtraDistanceExtraTimeRegressor.p", "rb"))


# Vectorized prediction
def predict_velocity_vectorized(min_time_arr, max_time_arr, init_vel_arr, dist_arr):
    features = np.column_stack((min_time_arr, max_time_arr, init_vel_arr, dist_arr))
    scaled = MinMaxScaler.transform(features)
    predictions = nextDesiredVelocityRegressor.predict(scaled)
    return predictions