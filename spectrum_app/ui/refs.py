from dataclasses import dataclass


REC_NUMBER = 5


@dataclass
class FileDialogs:
    fft: int | str
    wav: int | str
    wav_import: int | str
    project_save: int | str
    project_open: int | str


@dataclass
class PlotRefs:
    levels_xaxis: int | str


@dataclass
class ControlRefs:
    run_btn: int | str
    band_input: int | str
    rec_len: int | str
    mon_cb: int | str
    left_level: int | str
    right_level: int | str
    inputs_combo: int | str
    outputs_combo: int | str
    ref_combo: int | str
    welch_n_input: int | str
    window_width_input: int | str
    freq_length_input: int | str


@dataclass
class UiRefs:
    run_btn: int | str
    band_input: int | str
    rec_len: int | str
    mon_cb: int | str
    left_level: int | str
    right_level: int | str
    inputs_combo: int | str
    outputs_combo: int | str
    ref_combo: int | str
    welch_n_input: int | str
    window_width_input: int | str
    freq_length_input: int | str
    levels_xaxis: int | str
    last_io_update: float = 0.0


def merge_refs(controls: ControlRefs, plots: PlotRefs) -> UiRefs:
    return UiRefs(
        run_btn=controls.run_btn,
        band_input=controls.band_input,
        rec_len=controls.rec_len,
        mon_cb=controls.mon_cb,
        left_level=controls.left_level,
        right_level=controls.right_level,
        inputs_combo=controls.inputs_combo,
        outputs_combo=controls.outputs_combo,
        ref_combo=controls.ref_combo,
        welch_n_input=controls.welch_n_input,
        window_width_input=controls.window_width_input,
        freq_length_input=controls.freq_length_input,
        levels_xaxis=plots.levels_xaxis,
    )
