import numpy as np
from numpy.linalg import inv


class KalmanFilter:
    def __init__(self):
        self.prev_estimate = np.zeros((4, 1))
        self.prev_ignorance = 500 * np.eye(4)
        self.prev_time = 0

    def get_delta(self, t):
        return t - self.prev_time

    def prediction(self, F, D):
        transition = F
        process_error = D
        estimate = transition @ self.prev_estimate
        ignorance = transition @ self.prev_ignorance @ transition.T + process_error
        result = (estimate, ignorance)
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
        return result