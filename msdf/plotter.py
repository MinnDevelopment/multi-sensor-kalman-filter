from matplotlib import pyplot as plt, animation as anim

from msdf.utils import dist


class Plotter:
    def animate(self):
        raise NotImplementedError()


class WorldPlotter(Plotter):
    def __init__(self, truth, sensor, kalman):
        self.truth = truth
        self.sensor = sensor
        self.kalman = kalman
        self.pred_ptr = None
        self.measure_ptr = None
        self.ptr = None
        self.count = 0

    def animate(self):
        fig, ax = plt.subplots()

        predictions = []
        measurements = []

        def init_frame():
            ax.set_xlim(-12000, 12000)
            ax.set_ylim(-12000, 12000)
            for p in self.sensor.get_positions() or []:
                if p is not None:
                    ax.plot(*p.flatten(), 'ko')

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
                print("Filtering...", z.flatten(), R.flatten())
                H = self.sensor.H
                x, P = self.kalman.filtering(z, (x, P), H, R, self.count)

            print("Measure", z.flatten())
            print("Truth", true_position)
            print("Prediction", x.flatten()[0:2])
            print("Missed by ", dist(true_position, x))

            measurements.append(z.flatten())
            predictions.append(x.flatten())

            if self.pred_ptr:
                self.pred_ptr.remove()
            x = [v[0] for v in predictions]
            y = [v[1] for v in predictions]
            self.pred_ptr, = ax.plot(x, y, 'b')

            if self.measure_ptr:
                self.measure_ptr.remove()
            x = [v[0] for v in measurements]
            y = [v[1] for v in measurements]
            self.measure_ptr, = ax.plot(x, y, 'r', alpha=0.5)

            if self.ptr:
                self.ptr.remove()
            self.ptr, = ax.plot(*true_position, 'ko')
            self.count += 1
            print()

        return anim.FuncAnimation(fig, update_frame, self.truth.space, init_frame, interval=50, repeat=False)
