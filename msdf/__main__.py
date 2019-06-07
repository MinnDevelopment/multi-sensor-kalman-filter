from msdf import *


def main():
    truth = GroundTruth()
    sensors = []

    # position = (0, 0)
    # sensor = RadarSensor(position)
    # sensor.truth = truth
    # sensors.append(sensor)

    # sensor = GridSensor()
    # sensor.truth = truth
    # sensors.append(sensor)

    position = (-10000, 10000)
    sensor = RadarSensor(position)
    sensor.truth = truth
    sensors.append(sensor)
    position = (-10000, -10000)
    sensor = RadarSensor(position)
    sensor.truth = truth
    sensors.append(sensor)

    position = (10000, 10000)
    sensor = RadarSensor(position)
    sensor.truth = truth
    sensors.append(sensor)
    position = (10000, -10000)
    sensor = RadarSensor(position)
    sensor.truth = truth
    sensors.append(sensor)

    position = (6000, 0)
    sensor = RadarSensor(position)
    sensor.truth = truth
    sensors.append(sensor)
    position = (-6000, 0)
    sensor = RadarSensor(position)
    sensor.truth = truth
    sensors.append(sensor)

    sensor = MergedSensor(sensors)
    kalman = KalmanFilter()

    plotter = WorldPlotter(truth, sensor, kalman)
    a = plotter.animate()
    plt.show()

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main())
