from matplotlib import pyplot as plt, animation as anim

from msdf.utils import dist


def check(key, kwargs):
    try:
        return kwargs[key] is not False
    except KeyError:
        return True


class Plotter:
    def animate(self, **kwargs):
        raise NotImplementedError()


class WorldPlotter(Plotter):
    def __init__(self, truth, sensor, kalman, logging=False):
        self.truth = truth
        self.sensor = sensor
        self.kalman = kalman
        self.logging = logging
        self.pred_ptr = None
        self.measure_ptr = None
        self.ptr = None
        self.count = 0

    def animate(self, **kwargs):
        fig, ax = plt.subplots()

        measurements_x = []
        measurements_y = []
        predictions_x = []
        predictions_y = []

        def init_frame():
            ax.set_xlim(-12000, 12000)
            ax.set_ylim(-12000, 12000)

            if check("track", kwargs):
                traj = [self.truth.trajectory(t) for t in self.truth.space]
                x = [v[0] for v in traj]
                y = [v[1] for v in traj]
                ax.plot(x, y, 'k', alpha=0.5)
            self.count = 0

        def update_frame(t):
            delta = self.kalman.get_delta(self.count)
            true_position = self.truth.trajectory(t).flatten()
            F = self.sensor.F(delta)
            D = self.sensor.D(delta)
            z, R = self.sensor.measure(t)
            x, P = self.kalman.prediction(F, D)
            if delta >= 5:
                if self.logging:
                    print("Filtering...", z.flatten(), R.flatten())
                H = self.sensor.H
                x, P = self.kalman.filtering(z, (x, P), H, R, self.count)
            elif delta == 1 and check("retrodiction", kwargs):
                self.kalman.retrodiction(self.sensor.F(1))

            if self.logging:
                print("Measure", z.flatten())
                print("Truth", true_position)
                print("Prediction", x.flatten()[0:2])
                print("Missed by ", dist(true_position, x))
                print("Covariance", R.flatten())

            measurements_x.append(*z[0])
            measurements_y.append(*z[1])
            predictions_x.append(*x[0])
            predictions_y.append(*x[1])

            # plot sensor positions and visualizations
            self.sensor.draw(ax, z)

            # print line of predictions
            if check("prediction", kwargs):
                x = predictions_x
                y = predictions_y
                if self.pred_ptr:
                    self.pred_ptr.set_data(x, y)
                else:
                    self.pred_ptr, = ax.plot(x, y, 'b')

            # print line of measurements (scattered)
            if check("measures", kwargs):
                x = measurements_x
                y = measurements_y
                if self.measure_ptr:
                    self.measure_ptr.set_data(x, y)
                else:
                    self.measure_ptr, = ax.plot(x, y, 'r', alpha=0.5)

            # print true position
            if self.ptr:
                self.ptr.set_data(*true_position)
            else:
                self.ptr, = ax.plot(*true_position, 'ko')
            self.count += 1
            if self.logging:
                print()

        return anim.FuncAnimation(fig, update_frame, self.truth.space, init_frame, interval=50, repeat=False)
