# coding: utf-8
from OpenposeLogParser import OpenposeJsonParser
from OpenposeLauncher import OpenposeLauncher
import time
import _thread
import matplotlib.pyplot as plt
from matplotlib import animation


def run_openpose():
    openpose_launcher = OpenposeLauncher(dir_contains_models="G:/openpose",
                                         openpose_binary_path="G:/openpose/bin/OpenPoseDemo.exe")
    openpose_launcher.openpose_camera("G:\openpose\output", 0)

def run_parser_print_out():
    while 1:
        for i in  OpenposeJsonParser().stream_update_point_change_data_in_the_dir("G:\openpose\output",sum=True):
            print(i)

def run_parser_plot_out():
    fig, ax = plt.subplots()
    ln, = ax.plot([], [], lw=2)
    ax.set_ylim(0, 5)  # 设置scale
    ax.set_xlim(0, 50)
    ax.grid()
    xdata, ydata = [], []



    def update(data):
        x,y =data
        xdata.append(x)
        ydata.append(y)
        ax.set_xlim(x - 50, x)
        ln.set_data(xdata, ydata)

        return ln,

    def data_gen():
        x = 0
        while 1:
            for i in OpenposeJsonParser().stream_update_point_change_data_in_the_dir("G:\openpose\output",
                                                                    sum=True):
                x += 1
                yield  x,i
    print(12321)
    ani = animation.FuncAnimation(fig, update, data_gen, blit=True, interval=.1,
                                  repeat=False)
    plt.show()



_thread.start_new_thread(run_openpose,())
_thread.start_new_thread(run_parser_plot_out(),())
# do not close main thread
while 1:
    pass

