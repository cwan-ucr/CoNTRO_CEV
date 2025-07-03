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

def predict_velocity(min_time, max_time, init_vel, dist):
    features = np.array([[min_time, max_time, init_vel, dist]])
    scaled = MinMaxScaler.transform(features)
    prediction = nextDesiredVelocityRegressor.predict(scaled)[0]
    return prediction