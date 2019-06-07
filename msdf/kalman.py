import numpy as np
from numpy.linalg import inv, pinv


class KalmanFilter:
    def __init__(self):
        self.prev_estimate = np.zeros((4, 1))
        self.prev_ignorance = 500 * np.eye(4)
        self.prev_time = 0
        # list of tuples (x, P)
        self.predictions = []

    def get_delta(self, t):
        return t - self.prev_time

    def prediction(self, F, D):
        transition = F
        process_error = D
        estimate = transition @ self.prev_estimate
        ignorance = transition @ self.prev_ignorance @ transition.T + process_error
        result = (estimate, ignorance)
        self.predictions.append(result)
        return result

    def filtering(self, measurement, prediction, H, R, t):
        estimate, ignorance = prediction
        innovation = measurement - H @ estimate
        innovation_error = H @ ignorance @ H.T + R
        gain = ignorance @ H.T @ inv(innovation_error)

        self.prev_estimate = estimate + gain @ innovation
        self.prev_ignorance = ignorance - gain @ innovation_error @ gain.T
        self.prev_time = t
        result = (self.prev_estimate, self.prev_ignorance)
        self.predictions[-1] = result
        return result

    def retrodiction(self, F, steps=2):
        if len(self.predictions) < steps + 1:
            return
        # prev => l-1|k
        # last => l+1|k
        # pred => l+1|l
        prev_x = self.predictions[-1][0]
        prev_P = self.predictions[-1][1]
        for step in range(1, steps+1):
            last_x = self.predictions[-step][0]
            last_P = self.predictions[-step][1]
            pred_x = prev_x
            pred_P = prev_P
            prev_x = self.predictions[-step-1][0]
            prev_P = self.predictions[-step-1][1]

            innovation = last_x - pred_x
            covariance = last_P - pred_P

            gain = last_P @ F.T @ pinv(covariance)
            new_x = pred_x + gain @ innovation
            new_P = pred_P + gain @ covariance @ gain.T
            self.predictions[-step-1] = (new_x, new_P)
