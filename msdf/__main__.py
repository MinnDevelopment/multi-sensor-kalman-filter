import numpy as np
from numpy.linalg import inv

from msdf.sensors import RadarSensor


class KalmanFilter:
    def __init__(self):
        self.prev_estimate = np.zeros((4, 1))
        self.prev_ignorance = 1e3 * np.eye(4)
        self.prev_t = 0
        self.t = 1

    def prediction(self, F, D):
        transition = F
        process_error = D
        estimate = transition @ self.prev_estimate
        ignorance = transition @ self.prev_ignorance @ transition.T + process_error
        self.t += 1
        return estimate, ignorance

    def filtering(self, measurement, prediction, H, R):
        estimate, ignorance = prediction
        innovation = measurement - H @ estimate
        error = H @ ignorance @ H.T + R
        gain = ignorance @ H.T @ inv(error)

        self.prev_estimate = estimate + gain @ innovation
        self.prev_ignorance = ignorance - gain @ error @ gain.T
        self.prev_t = self.t
        self.t += 1
        return self.prev_estimate, self.prev_ignorance


def main():
    sensor = RadarSensor((0, 0))
    filter = KalmanFilter()
    space = sensor.truth.space

    predictions = []
    for t in space:
        delta = filter.t - filter.prev_t
        F = sensor.F(filter.t, filter.prev_t)
        D = sensor.D(filter.t, filter.prev_t)
        H = sensor.H
        real = sensor.truth.trajectory(t).flatten()
        pred, P = filter.prediction(F, D)
        estimate = pred.flatten()[0:2]
        R, z = sensor.measure(t)

        if delta >= 5:
            print("Filtering...")
            pred, _ = filter.filtering(z, (pred, P), H, R)
            estimate = pred.flatten()[0:2]

        predictions.append(estimate)
        print("Measured: ", z.flatten())
        print("Real: ", real)
        print("Pred: ", estimate)
        print()

    import matplotlib.pyplot as plt

    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    track = [sensor.truth.trajectory(t).flatten() for t in space]
    x = [v[0] for v in track]
    y = [v[1] for v in track]
    plt.plot(x, y, color="k", alpha=0.5)
    x = [v[0] for v in predictions]
    y = [v[1] for v in predictions]
    plt.plot(x, y, "ro")
    plt.show()
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
