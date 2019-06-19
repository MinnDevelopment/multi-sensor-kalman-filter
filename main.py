from msdf import *

preset = [(6000, 0), (-6000, 0), (0, 5000), (0, -5000)]


def add_sensor(pos, sensors, truth):
    sensor = RadarSensor(pos)
    sensor.truth = truth
    sensors.append(sensor)


def radar(truth):
    sensors = []

    # preset = one sensor in each loop and one above and below the intersection respectively
    answer = input("Use preset? [y/n] ")
    if answer in "yY":
        for pos in preset:
            add_sensor(pos, sensors, truth)
    else:
        answer = "y"
        while answer in "yY":
            print("Adding new sensor...")
            x = int(input("x = "))
            y = int(input("y = "))
            add_sensor((x, y), sensors, truth)
            print("Current sensors:", sensors)
            answer = input("Continue? [y/n] ")

    return MergedSensor(sensors)


def grid(truth):
    sensor = GridSensor()
    sensor.truth = truth
    return sensor


def main():
    # open interactive menu for configuration
    truth = GroundTruth()

    print("Choose Sensor Type")
    print("[0] Exit Program")
    print("[1] Radar")
    print("[2] Grid")
    answer = input("\nType: ")

    if answer == "1":
        sensor = radar(truth)
    elif answer == "2":
        sensor = grid(truth)
    elif answer == "0":
        return 0
    else:
        print("Try again!")
        return 1

    answer = input("Use retrodiction? [y/n] ")
    retro = answer in "yY"

    kalman = KalmanFilter()
    plotter = WorldPlotter(truth, sensor, kalman)
    a = plotter.animate(retrodiction=retro)
    plt.show()

    return 0


if __name__ == '__main__':
    import sys
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
