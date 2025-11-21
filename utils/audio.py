"""Audio I/O module for real-time audio processing and device management.

This module provides comprehensive audio input/output capabilities including:
- Real-time audio streaming with configurable devices
- Audio level monitoring and measurement
- Device enumeration and management
- Signal generation and recording
- Cross-platform support with Windows-specific optimizations
"""

import sys
import sounddevice as sd
import numpy as np
from threading import Thread, Lock, Event
from time import sleep, time, monotonic, monotonic_ns
from typing import Literal, NamedTuple

from .generators import log_sweep, pink_noise
from .classes import GenMode, RefMode

# Platform detection for Windows-specific functionality
WIN32 = sys.platform == "win32"

if WIN32:
    import pythoncom


class PaStreamCallbackTimeInfo(NamedTuple):
    """Timing information for PortAudio stream callbacks.

    Contains precise timing information for audio stream callbacks,
    useful for synchronization and latency measurement.

    Attributes:
        inputBufferAdcTime (float): Time when the first sample of the input
            buffer was captured at the ADC input
        currentTime (float): Current time in seconds
        outputBufferDacTime (float): Time when the first sample of the output
            buffer will be played by the DAC
    """

    inputBufferAdcTime: float
    currentTime: float
    outputBufferDacTime: float


class io_list_updater(Thread):
    """Background thread for continuously updating audio device lists.

    Maintains up-to-date lists of available input and output audio devices.
    Runs as a daemon thread and periodically refreshes device information
    to handle device connections/disconnections during runtime.

    Attributes:
        inputs_list (list[str]): List of available input device descriptions
        outputs_list (list[str]): List of available output device descriptions
        list_lock (Lock): Thread lock for safe access to device lists
        enable (Event): Control event to enable/disable device scanning
        paused (Event): Status event indicating if scanning is paused
    """

    def __init__(self) -> None:
        """Initialize the audio device list updater."""
        # Device lists
        self.inputs_list: list[str] = []
        self.outputs_list: list[str] = []

        # Thread synchronization
        self.list_lock = Lock()
        self.enable = Event()
        self.paused = Event()

        super().__init__(daemon=True)

    def run(self) -> None:
        """Main thread loop for updating device lists.

        Continuously monitors and updates audio device lists when enabled.
        Reinitializes the PortAudio system after each update to ensure
        fresh device information.
        """
        while True:
            if self.enable.is_set():
                self.paused.clear()
                self.upd_inputs()
                self.upd_outputs()
                # Reinitialize PortAudio to refresh device list
                sd._terminate()
                sd._initialize()
            else:
                self.paused.set()
            sleep(1)  # Update interval

    def upd_inputs(self) -> None:
        """Update the list of available input devices.

        Thread-safe method to refresh the input device list.
        """
        device_names = self.list_devices("input")
        with self.list_lock:
            self.inputs_list = device_names[:]

    def upd_outputs(self) -> None:
        """Update the list of available output devices.

        Thread-safe method to refresh the output device list.
        """
        device_names = self.list_devices("output")
        with self.list_lock:
            self.outputs_list = device_names[:]

    @staticmethod
    def list_devices(io: Literal["input", "output"] | None = None) -> list[str]:
        """Get a formatted list of available audio devices.

        Args:
            io (Literal["input", "output"] | None, optional): Filter devices by type.
                - "input": Only input-capable devices
                - "output": Only output-capable devices
                - None: All devices
                Defaults to None.

        Returns:
            list[str]: List of formatted device descriptions in the format:
                "index: name, host_api, input_channels>>output_channels, sample_rate kHz"

        Note:
            Excludes "Windows WDM-KS" devices as they are typically not suitable
            for general audio applications.
        """
        # Get host API names for reference
        hostapi_names = []
        for hostapi in sd.query_hostapis():
            if isinstance(hostapi, dict):
                hostapi_names.append(hostapi["name"])

        devices = []
        for device_info in sd.query_devices():
            if isinstance(device_info, dict):
                device_index = device_info["index"]
                device_name = device_info["name"]
                host_api = hostapi_names[device_info["hostapi"]]

                input_channels = device_info["max_input_channels"]
                output_channels = device_info["max_output_channels"]
                sample_rate = device_info["default_samplerate"]

                # Format device description
                device_text = (
                    f"{device_index}: {device_name}, {host_api}, "
                    f"{input_channels}>>{output_channels}, {sample_rate/1000:.1f} kHz"
                )

                # Filter out WDM-KS devices (typically problematic)
                if host_api != "Windows WDM-KS":
                    if io is None:
                        devices.append(device_text)
                    elif io == "input" and input_channels > 0:
                        devices.append(device_text)
                    elif io == "output" and output_channels > 0:
                        devices.append(device_text)

        return devices

    @staticmethod
    def get_device_indx(name: str) -> int:
        """Extract device index from formatted device name.

        Args:
            name (str): Formatted device name ("index: name, ...")

        Returns:
            int: Device index for use with sounddevice
        """
        return int(name.split(":")[0])

    @property
    def inputs(self) -> list[str]:
        """Get thread-safe copy of current input device list.

        Returns:
            list[str]: List of available input devices
        """
        with self.list_lock:
            return self.inputs_list[:]

    @property
    def outputs(self) -> list[str]:
        """Get thread-safe copy of current output device list.

        Returns:
            list[str]: List of available output devices
        """
        with self.list_lock:
            return self.outputs_list[:]


class InputMeter(Thread):
    """Real-time audio level meter for input monitoring.

    Continuously monitors audio input levels from a specified device,
    providing peak level measurements for stereo channels. Useful for
    setting proper input gains and monitoring signal presence.

    Attributes:
        level (np.ndarray): Current peak levels for both channels [L, R]
        level_lock (Lock): Thread lock for safe access to level data
        enable (Event): Control event to start/stop monitoring
        device (int | None): Audio input device index, None for default
    """

    def __init__(self) -> None:
        """Initialize the input level meter."""
        # Level measurement data
        self.level = np.zeros(2)  # [Left, Right] channel levels
        self.level_lock = Lock()

        # Control and configuration
        self.enable = Event()
        self.device: int | None = None

        super().__init__(daemon=True)

    def run(self) -> None:
        """Main thread loop for continuous level monitoring.

        Creates an input stream and continuously reads audio data to
        calculate peak levels. Handles PortAudio errors gracefully by
        disabling monitoring and logging the error.
        """
        while True:
            self.enable.wait()  # Wait for enable signal

            try:
                # Create stereo input stream
                stream = sd.InputStream(device=self.device, channels=2)
                stream.start()

                # Continuous level monitoring loop
                while self.enable.is_set():
                    # Read audio chunk (1024 samples)
                    audio_chunk = stream.read(1024)[0]

                    # Calculate peak levels for each channel
                    peak_levels: np.ndarray = np.max(np.abs(audio_chunk), axis=0)

                    # Thread-safe update of level data
                    with self.level_lock:
                        self.level = peak_levels.copy()

                stream.close()

            except sd.PortAudioError as e:
                # Handle audio system errors
                self.enable.clear()
                print(f"InputMeter PortAudio error: {e}")

    def get_levels(self) -> np.ndarray:
        """Get current peak levels in a thread-safe manner.

        Returns:
            np.ndarray: Peak levels for [Left, Right] channels
                Values range from 0.0 (silence) to ~1.0 (full scale)
        """
        with self.level_lock:
            return self.level.copy()


class AudioIO(Thread):
    """High-level audio I/O manager for measurement and analysis.

    Provides synchronized audio input/output capabilities for acoustic measurements.
    Supports signal generation (log sweeps, pink noise) with simultaneous recording
    for transfer function analysis, frequency response measurement, and other
    audio analysis tasks.

    Key features:
    - Synchronized input/output streaming
    - Multiple signal generation modes
    - Real-time level monitoring
    - Configurable devices and parameters
    - Thread-safe operation with event synchronization

    Attributes:
        length (float): Recording/playback duration in seconds
        device (tuple): Input and output device indices (input_dev, output_dev)
        gen_mode (GenMode): Signal generation mode (LOG_SWEEP, PINK_NOISE)
        band (tuple): Frequency range for generated signals (min_freq, max_freq)
        ref (RefMode): Reference mode for channel comparison
        in_fs (float): Input sampling rate (Hz)
        out_fs (float): Output sampling rate (Hz)
        record (np.ndarray): Recorded audio data buffer
        signal (np.ndarray): Generated output signal
        Various Event objects for thread synchronization
    """

    def __init__(
        self,
        length: float = 10.0,
        device: tuple[int, int] | tuple[None, None] = (None, None),
        gen_mode: GenMode = GenMode.LOG_SWEEP,
        band: tuple[float, float] = (20, 20000),
        ref: RefMode = RefMode.CHANNEL_B,
        daemon: bool = False,
    ) -> None:
        """Initialize the AudioIO system.

        Args:
            length (float, optional): Recording duration in seconds. Defaults to 10.0.
            device (tuple[int, int] | tuple[None, None], optional): Audio device indices
                as (input_device, output_device). None uses system default.
                Defaults to (None, None).
            gen_mode (GenMode, optional): Signal generation mode.
                Defaults to GenMode.LOG_SWEEP.
            band (tuple[float, float], optional): Frequency range for generated signals
                in Hz as (min_freq, max_freq). Defaults to (20, 20000).
            ref (RefMode, optional): Reference mode for analysis.
                Defaults to RefMode.CHANNEL_B.
            daemon (bool, optional): Run as daemon thread. Defaults to False.
        """
        # Recording/playback configuration
        self.length = length
        self.device: tuple[int | None, int | None] = device
        self.gen_mode: GenMode = gen_mode
        self.band = band
        self.ref: RefMode = ref
        self.padding_time = 0.2  # Padding time in seconds for sync

        # Audio parameters (set during stream creation)
        self.in_fs = 0  # Input sampling rate
        self.in_n = 0  # Input buffer size
        self.out_fs = 0.0  # Output sampling rate
        self.out_n = 0  # Output buffer size

        # Audio data buffers
        self.record = np.empty((0, 2), np.float32)  # Recorded stereo data
        self.signal = np.empty((0, 2), np.float32)  # Generated output signal

        # Stream position tracking
        self.out_position = 0  # Current output position
        self.in_position = 0  # Current input position
        self.start_time = 0  # Recording start timestamp

        # Thread synchronization objects
        self.record_lock = Lock()  # Protects record buffer access
        self.running = Event()  # Main control: start/stop processing
        self.record_updated = Event()  # Signals new recorded data
        self.levels_updated = Event()  # Signals level data update
        self.record_completed = Event()  # Signals recording completion
        self.exit = Event()  # Signals thread shutdown
        self.out_stop = Event()  # Signals output stream completion
        self.in_stop = Event()  # Signals input stream completion

        super().__init__(daemon=daemon)

    def run(self) -> None:
        """Main thread loop for audio I/O operations.

        Manages the complete audio measurement cycle:
        1. Initialize COM for Windows compatibility
        2. Wait for run signal
        3. Query device capabilities and set up streams
        4. Generate test signals based on selected mode
        5. Start synchronized input/output streaming
        6. Wait for completion and cleanup

        The method handles Windows COM initialization for WASAPI compatibility
        and provides comprehensive error handling.
        """
        # Initialize COM for Windows audio compatibility
        if WIN32:
            pythoncom.CoInitializeEx(0)  # type: ignore

        try:
            while not self.exit.is_set():
                # Wait for signal to start measurement
                self.running.wait()

                # Reset completion flags
                self.in_stop.clear()
                self.out_stop.clear()

                if not self.exit.is_set():
                    # Query device capabilities and set sample rates
                    self._setup_device_parameters()

                    # Create and configure input stream
                    input_stream = self._create_input_stream()

                    # Create and configure output stream
                    output_stream = self._create_output_stream()

                    # Generate test signal based on selected mode
                    self._generate_test_signal()

                    # Start synchronized streaming
                    input_stream.start()
                    output_stream.start()

                    # Wait for both streams to complete
                    self.out_stop.wait()
                    self.in_stop.wait()

                    # Clean up streams
                    input_stream.stop()
                    output_stream.stop()
                    input_stream.close()
                    output_stream.close()

                # Signal completion and reset state
                self.running.clear()
                self.record_completed.set()

        except Exception as e:
            print(f"AudioIO exception: {e}")
        finally:
            # Clean up COM initialization
            if WIN32:
                pythoncom.CoUninitialize()  # type: ignore
            self.running.clear()

    def _setup_device_parameters(self) -> None:
        """Query device capabilities and set sampling rates."""
        # Get input device sample rate
        input_info = sd.query_devices(self.device[0])
        if isinstance(input_info, dict):
            self.in_fs = input_info["default_samplerate"]

        # Get output device sample rate
        output_info = sd.query_devices(self.device[1])
        if isinstance(output_info, dict):
            self.out_fs = output_info["default_samplerate"]

    def _create_input_stream(self) -> sd.InputStream:
        """Create and configure the input audio stream.

        Returns:
            sd.InputStream: Configured input stream
        """
        input_stream = sd.InputStream(
            device=self.device[0],
            callback=self.input_callback,
            blocksize=int(self.in_fs * 0.1),  # 100ms blocks
        )

        # Update actual sample rate from stream
        self.in_fs:int = input_stream.samplerate

        # Calculate buffer size (add extra time for padding and sync)
        self.in_n = int((self.length + self.padding_time + 1) * self.in_fs)

        # Initialize recording buffer
        with self.record_lock:
            self.record_completed.clear()
            self.record = np.zeros((self.in_n, 2), np.float32)
            self.in_position = 0

        return input_stream

    def _create_output_stream(self) -> sd.OutputStream:
        """Create and configure the output audio stream.

        Returns:
            sd.OutputStream: Configured output stream
        """
        output_stream = sd.OutputStream(
            device=self.device[1],
            callback=self.output_callback,
            blocksize=int(self.out_fs * 0.1),  # 100ms blocks
        )

        # Update actual sample rate from stream
        self.out_fs = output_stream.samplerate
        self.out_position = 0

        return output_stream

    def _generate_test_signal(self) -> None:
        """Generate the test signal based on the selected generation mode."""
        # Calculate signal length in samples
        self.out_n = int(self.length * self.out_fs)

        # Generate signal based on mode
        sample_rate_int = int(self.out_fs)
        if self.gen_mode == GenMode.LOG_SWEEP:
            self.signal = log_sweep(self.out_n, sample_rate_int, self.band)
        elif self.gen_mode == GenMode.PINK_NOISE:
            self.signal = pink_noise(self.out_n, sample_rate_int, self.band)
        else:
            raise ValueError(f"Unknown generation mode: {self.gen_mode}")

        # Add padding for synchronization
        padding_samples = int(self.out_fs * self.padding_time)
        self.signal = np.pad(self.signal, ((padding_samples, 0), (0, 0)))
        self.out_n += padding_samples

    def output_callback(
        self,
        outdata: np.ndarray,
        n: int,
        time: PaStreamCallbackTimeInfo,
        status: sd.CallbackFlags,
    ) -> None:
        """Audio output callback for streaming generated signals.

        Called by PortAudio when output buffer needs new audio data.
        Streams the generated test signal and signals completion when done.

        Args:
            outdata (np.ndarray): Output buffer to fill with audio data
            n (int): Number of frames requested
            time (PaStreamCallbackTimeInfo): Timing information
            status (sd.CallbackFlags): Stream status flags
        """
        # Calculate data range to copy
        start_pos = self.out_position
        end_pos = min(start_pos + n, self.out_n)

        # Fill output buffer with zeros first
        outdata.fill(0)

        # Copy signal data to output buffer
        samples_to_copy = end_pos - start_pos
        outdata[:samples_to_copy, :2] = self.signal[start_pos:end_pos]

        # Update position
        self.out_position = end_pos

        # Signal completion when all data has been sent
        if end_pos >= self.out_n:
            self.out_stop.set()

    def input_callback(
        self,
        indata: np.ndarray,
        n: int,
        time: PaStreamCallbackTimeInfo,
        status: sd.CallbackFlags,
    ) -> None:
        """Audio input callback for recording incoming audio data.

        Called by PortAudio when new audio input data is available.
        Records the data to the internal buffer and signals updates.
        Handles mono to stereo conversion if needed.

        Args:
            indata (np.ndarray): Input audio data from device
            n (int): Number of frames available
            time (PaStreamCallbackTimeInfo): Timing information
            status (sd.CallbackFlags): Stream status flags
        """
        with self.record_lock:
            # Calculate data range to store
            start_pos = self.in_position
            end_pos = min(start_pos + n, self.in_n)

            # Handle mono to stereo conversion if needed
            input_data = indata
            if input_data.shape[1] < 2:
                input_data = np.repeat(input_data, 2, axis=1)

            # Store data in recording buffer
            samples_to_store = end_pos - start_pos
            self.record[start_pos:end_pos] = input_data[:samples_to_store, :2]

            # Update position
            self.in_position = end_pos

        # Signal that new data is available
        self.levels_updated.set()
        self.record_updated.set()

        # Signal completion when buffer is full
        if end_pos >= self.in_n:
            self.in_stop.set()

    def get_record(self) -> np.ndarray:
        """Get the current recorded audio data.

        Returns a copy of the recorded data up to the current position.
        Clears the record_updated event flag after retrieval.
        Optionally replaces channel 2 with the generator signal if using
        GENERATOR reference mode.

        Returns:
            np.ndarray: Recorded stereo audio data with shape (n_samples, 2)
                Channel 0: Always the recorded input
                Channel 1: Recorded input or generator signal (if ref=GENERATOR)
        """
        # Get thread-safe copy of current recording
        with self.record_lock:
            current_data = self.record[: self.in_position].copy()

        # Clear the update flag
        self.record_updated.clear()

        # Replace channel 2 with generator signal if using generator reference
        if self.ref == RefMode.GENERATOR:
            signal_length = min(len(current_data), self.out_position)
            current_data[:, 1].fill(0)
            current_data[:signal_length, 1] = self.signal[:signal_length, 0]

        return current_data

    def get_levels(self, time_step: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """Get time-varying peak levels from recorded audio data.

        Analyzes the recorded audio in time chunks and returns peak levels
        for each channel over time. Useful for creating level meters and
        monitoring signal presence during recording.

        Args:
            time_step (float, optional): Time resolution in seconds for level analysis.
                Smaller values provide higher time resolution but more data points.
                Defaults to 0.1 (100ms chunks).

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing:
                - timestamps: 1D array of time values (shape: (N,)) in seconds
                - levels: 2D array of peak levels (shape: (2, N)) where:
                  levels[0, :] = peak levels for channel 1 (left)
                  levels[1, :] = peak levels for channel 2 (right)

        Note:
            Clears the levels_updated event flag after retrieval.
        """
        # Get thread-safe copy of current recording
        with self.record_lock:
            current_position = self.in_position
            audio_data = self.record[:current_position].copy()

        # Calculate chunk parameters
        chunk_size = int(time_step * self.in_fs)

        # Generate chunk start/end indices
        chunk_starts = np.arange(0, current_position, chunk_size)
        chunk_ends = chunk_starts + chunk_size

        # Generate timestamps for each chunk
        timestamps = np.arange(len(chunk_starts)) * time_step

        # Calculate peak levels for each chunk
        peak_levels = np.zeros((len(chunk_starts), 2))
        for i, (start_idx, end_idx) in enumerate(zip(chunk_starts, chunk_ends)):
            chunk_data = audio_data[start_idx:end_idx]
            peak_levels[i] = np.max(np.abs(chunk_data), axis=0)

        # Clear the update flag
        self.levels_updated.clear()

        return timestamps, peak_levels.T

    def kill(self) -> None:
        """Terminate the AudioIO thread completely.

        Sets all necessary events to gracefully shut down the thread.
        Should be called when the AudioIO object is no longer needed.
        """
        self.exit.set()
        self.running.set()
        self.in_stop.set()
        self.out_stop.set()

    def run_once(self) -> None:
        """Execute a single measurement cycle.

        Starts the audio streaming and waits for both input and output
        to complete. Blocks until the measurement is finished.

        This is a convenience method for synchronous operation.
        """
        self.running.set()
        self.in_stop.wait()
        self.out_stop.wait()

    def stop_audio(self) -> None:
        """Stop the current audio streaming operation.

        Interrupts any ongoing measurement by stopping both input
        and output streams. The thread remains alive and can be
        restarted with another run_once() or running.set() call.
        """
        self.running.clear()
        self.in_stop.set()
        self.out_stop.set()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from scipy.signal import welch, periodogram
    from scipy.signal.windows import gaussian

    pass
    # # updater = io_list_updater()
    # # updater.start()
    # # updater.join()
    # meter = InputMeter()
    # meter.start()
    # meter.enable.set()

    # while True:
    #     print(meter.get_levels())

    for dev in io_list_updater.list_devices():
        print(dev)

    # stream = sd.Stream()
    io = AudioIO(
        10, gen_mode=GenMode.LOG_SWEEP, band=(100, 20000), device=(14, 12), daemon=True
    )
    io.start()
    io.run_once()
    # while True:
    #     io.run_once()
    figure, plots = plt.subplots(2, 1, figsize=(8, 6))
    plots: list[Axes]
    ts, levels = io.get_levels(0.1)
    plots[0].plot(ts, 20 * np.log10(levels[0].clip(1e-12)), label="Channel 1")
    plots[0].plot(ts, 20 * np.log10(levels[1].clip(1e-12)), label="Channel 2")

    rec = io.get_record()
    fs, Pxx = periodogram(rec, fs=io.in_fs, axis=0)
    plots[1].semilogx(
        fs, 10 * np.log10((fs * Pxx[:, 0]).clip(1e-20)), label="Channel 1"
    )
    plots[1].semilogx(
        fs, 10 * np.log10((fs * Pxx[:, 1]).clip(1e-20)), label="Channel 2"
    )
    # a, b = io.get_record().T
    # ts = np.arange(len(a)) / io.fs
    # plt.plot(ts, a, label="Channel 1")
    # plt.plot(ts, b, label="Channel 2")
    # ts = np.diff(io.cb_ts[: io.cb_i])
    # ns = np.cumsum(io.cb_ns[: len(ts)]) / io.fs * 1e9 - np.cumsum(ts)
    # plt.plot(np.cumsum(ts) / 1e9, ts)
    # plt.plot(np.cumsum(ts) / 1e9, ns)

    # # plt.plot(x)
    # # print()
    # f, p = periodogram(x, io.fs)
    # plt.semilogx(f, 10 * np.log10((f * p).clip(1e-20)), linewidth=0.2)

    # log_f = np.geomspace(20, 20e3, 10000)
    # log_p = np.interp(log_f, f, p)
    # ww = 1000
    # w = gaussian(ww, ww / 8)
    # log_p = np.convolve(log_p, w / np.sum(w), "same")
    # plt.semilogx(log_f, 10 * np.log10((log_f * log_p).clip(1e-20)))

    # # plt.plot(10 * np.log10(w))
    plt.legend()
    plots[0].grid(True, "both")
    plots[1].grid(True, "both")
    plt.show()
