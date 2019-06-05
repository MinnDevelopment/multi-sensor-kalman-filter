from time import time, sleep

from numpy.linalg import inv

from msdf.sensors import *


def millis():
    return int(time() * 1000)


class KalmanFilter:
    def __init__(self):
        self.prev_estimate = np.zeros((4, 1))
        self.prev_ignorance = 1e3 * np.eye(4)
        self.predictions = []
        self.prev_time = millis()

    def get_delta(self):
        return millis() - self.prev_time

    def prediction(self, F, D):
        transition = F
        process_error = D
        estimate = transition @ self.prev_estimate
        ignorance = transition @ self.prev_ignorance @ transition.T + process_error
        result = (estimate, ignorance)
        self.predictions.append(result)
        return result

    def filtering(self, measurement, prediction, H, R):
        estimate, ignorance = prediction
        innovation = measurement - H @ estimate
        error = H @ ignorance @ H.T + R
        gain = ignorance @ H.T @ inv(error)

        self.prev_estimate = estimate + gain @ innovation
        self.prev_ignorance = ignorance - gain @ error @ gain.T
        self.prev_time = millis()
        result = (self.prev_estimate, self.prev_ignorance)
        self.predictions.pop()
        self.predictions.append(result)
        return result


def main():
    sensor = RadarSensor((0, 0))
#    sensor = GridSensor()
    kalman = KalmanFilter()
    space = sensor.truth.space

    predictions = []
    for t in space:
        delta = kalman.get_delta()
        F = sensor.F(delta)
        D = sensor.D(delta)
        H = sensor.H
        real = sensor.truth.trajectory(t).flatten()
        pred, P = kalman.prediction(F, D)
        estimate = pred.flatten()[0:2]
        R, z = sensor.measure(t)

        if delta >= 5:
            filtered = True
            print("Filtering...", delta)
            pred, _ = kalman.filtering(z, (pred, P), H, R)
            estimate = pred.flatten()[0:2]
        else:
            filtered = False

        predictions.append((estimate, filtered))
        print("Measured: ", z.flatten())
        print("Real: ", real)
        print("Pred: ", estimate)
        print()
        sleep(0.001)

    import matplotlib.pyplot as plt

    track = [sensor.truth.trajectory(t).flatten() for t in space]
    x = [v[0] for v in track]
    y = [v[1] for v in track]
    plt.plot(x, y, color="k", alpha=0.5)
    preds = [v[0] for v in predictions if not v[1]]
    x = [v[0] for v in preds]
    y = [v[1] for v in preds]
    plt.plot(x, y, "ro")
    preds = [v[0] for v in predictions if v[1]]
    x = [v[0] for v in preds]
    y = [v[1] for v in preds]
    plt.plot(x, y, "go")
    plt.show()
    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main())
