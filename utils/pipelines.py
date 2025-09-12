from generators import PinkNoiseGenerator, SignalGenerator, LogSweepGenerator
from analasers import RecordingAnalyzer, Analyzer
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
import numpy as np
from threading import Thread
from queue import Empty, Full
from time import sleep, time

# pass

if __name__ == "__main__":
    RATE = 96000
    CHUNKSIZE = 1024 * 4
    LENGTH = 30  # seconds

    # Create a pink noise generator
    generator = PinkNoiseGenerator(
        rate=RATE, chunksize=CHUNKSIZE, length=LENGTH, band=(100, 1000)
    )

#     # Create a logarithmic sweep generator
    # generator = LogSweepGenerator(
    #     rate=RATE, chunksize=CHUNKSIZE, length=LENGTH, band=(100, 1000)
    # )

#     # Create a recording analyzer
    analyzer = RecordingAnalyzer(RATE)
    analyzer.ref = "None"  # "Chanel B" or "None"
    analyzer.weighting = "A"  # "A", "C" or None
    analyzer.window_width=1/10

#     # Start the threads
    generator.start()
    analyzer.start()
#     pass
    def linkgentoanalayzer(gen: SignalGenerator, anal: Analyzer):
        while gen.is_alive() or not gen.output_queue.empty():
            chunk = gen.get()  # Get mono chunk from generator
            chunk = np.column_stack((chunk, chunk))  # Make it stereo
            try:
                # Put stereo chunk to analyzer input queue
                anal.input_queue.put_nowait(chunk)
            except Full:
                # print("Analyzer input queue full, dropping chunk.")
                pass
            sleep(CHUNKSIZE / RATE)
        print(f"Linker stoped.")

    linker = Thread(target=linkgentoanalayzer, args=(generator, analyzer))
    linker.start()

    plt.ion()
    fig, ax = plt.subplots()
    ax: Axes = ax
    (line,) = ax.semilogx([], [])
    ax.set_xlim(20, 20e3)
    line: Line2D = line
    while analyzer.is_alive() or not analyzer.freq_queue.empty():
        if not generator.is_alive() and generator.output_queue.empty():
            analyzer.stop()
        freq_data = analyzer.getresults()
        # analyzer.levels_queue.get_nowait()
        # freq_data[:, 1] = freq_data[:, 1] / freq_data[:, 0]  # A-weighting approximation
        line.set_data(freq_data)
        ax.relim()
        ax.autoscale_view(scalex=False)
        plt.draw()
        plt.pause(0.05)
    plt.ioff()
    plt.show()
