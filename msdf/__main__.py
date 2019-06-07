from msdf import *

preset = [(-10000, -10000), (10000, 10000), (-10000, 10000), (10000, -10000),
          (6000, 0), (-6000, 0), (0, 5000), (0, -5000)]


def add_sensor(pos, sensors, truth):
    sensor = RadarSensor(pos)
    sensor.truth = truth
    sensors.append(sensor)


def radar(truth):
    sensors = []

    answer = input("Use preset? [y/n] ")
    if answer.lower() == "y":
        for pos in preset:
            add_sensor(pos, sensors, truth)
    else:
        answer = "y"
        while answer.lower() == "y":
            print("Adding new sensor...")
            x = int(input("x = "))
            y = int(input("y = "))
            add_sensor((x, y), sensors, truth)
            answer = input("Continue? [y/n] ")

    return MergedSensor(sensors)


def grid(truth):
    sensor = GridSensor()
    sensor.truth = truth
    return sensor


def main():
    truth = GroundTruth()
    sensor = None

    print("Choose Sensor Type")
    print("[1] Radar")
    print("[2] Grid")
    answer = input("\nType: ")

    if answer == "1":
        sensor = radar(truth)
    elif answer == "2":
        sensor = grid(truth)
    else:
        print("Try again!")
        exit(1)

    kalman = KalmanFilter()
    plotter = WorldPlotter(truth, sensor, kalman)
    a = plotter.animate(prediction=True)
    plt.show()

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main())
