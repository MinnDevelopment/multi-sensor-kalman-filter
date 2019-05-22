import numpy as np
from numpy.linalg import inv

from msdf.sensors import RadarSensor


class KalmanFilter:
    def __init__(self):
        self.prev_estimate = np.zeros((4, 1))
        self.prev_ignorance = np.eye(4)
        self.prev_t = 0
        self.t = 0

    def prediction(self, F, D):
        transition = F
        process_error = D
        estimate = transition @ self.prev_estimate
        ignorance = transition @ self.prev_ignorance @ transition.T + process_error
        self.t += 1
        return estimate, ignorance

    def filtering(self, measurement, F, D, H, R):
        estimate, ignorance = self.prediction(F, D)
        v = measurement - H @ estimate
        gain = H @ ignorance @ H.T + R
        W = ignorance @ H.T @ inv(gain)

        self.prev_estimate = estimate + W @ v
        self.prev_ignorance = ignorance - W @ gain @ W.T
        self.prev_t = self.t
        return self.prev_estimate, self.prev_ignorance


def main():
    sensor = RadarSensor((0, 0))
    filter = KalmanFilter()

    for t in sensor.truth.space:
        F = sensor.F(filter.t, filter.prev_t)
        D = sensor.D(filter.t, filter.prev_t)
        if filter.t % 5 == 0:
            print("Filtering...")
            H = sensor.H
            R, z = sensor.measure(t)
            pred, _ = filter.filtering(z, F, D, H, R)
        else:
            pred, _ = filter.prediction(F, D)
        print("Real: ", sensor.truth.trajectory(t).flatten())
        print("Pred: ", pred.flatten()[0:2])
        print()
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
