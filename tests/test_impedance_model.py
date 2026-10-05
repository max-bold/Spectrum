import unittest

import numpy as np
from scipy.signal import find_peaks

from audioanalysis.impedance_model import (
    fit_impedance,
    fit_impedance_auto,
    format_spice_table,
    rlc_from_rf0q,
    speaker_impedance,
)


class ImpedanceModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frequency = np.geomspace(20.0, 20_000.0, 1024)

    def magnitude(self, params):
        return np.abs(speaker_impedance(
            self.frequency, np.asarray(params), (len(params) - 2) // 3
        ))

    def test_fixed_stage_keeps_one_millihenry_and_measured_peak(self) -> None:
        measured = self.magnitude([6.8, 0.0004, 30.0, 60.0, 3.0])
        fit = fit_impedance(
            self.frequency, measured, 1, refine_inductances=False
        )
        params = fit.physical_params[2:].reshape(-1, 3)
        inductance, capacitance = rlc_from_rf0q(*params.T)
        np.testing.assert_allclose(inductance, 0.001, rtol=1e-12)
        peaks, _ = find_peaks(measured, prominence=0.5)
        self.assertEqual(params[0, 1], self.frequency[peaks[0]])
        np.testing.assert_allclose(
            capacitance, 1 / ((2 * np.pi * params[:, 1])**2 * 0.001)
        )
        self.assertFalse(fit.inductances_refined)
        self.assertEqual(format_spice_table(fit).sections[0][0], "1")

    def test_refinement_fits_width_without_moving_resonance(self) -> None:
        measured = self.magnitude([6.8, 0.0004, 30.0, 60.0, 3.0])
        fit, stages = fit_impedance_auto(self.frequency, measured)
        self.assertEqual(len(stages), 2)
        self.assertTrue(fit.inductances_refined)
        fixed = fit_impedance(self.frequency, measured, 1, refine_inductances=False)
        np.testing.assert_array_equal(
            fixed.physical_params[3::3], fit.physical_params[3::3]
        )
        self.assertLess(fit.rms_log_error, fixed.rms_log_error * 0.05)
        self.assertLess(fit.rms_log_error, 0.01)
        np.testing.assert_allclose(fit.physical_params[:2], [6.8, 0.0004], rtol=0.01)
        inductance, _ = rlc_from_rf0q(*fit.physical_params[2:].reshape(-1, 3).T)
        self.assertGreater(inductance[0], 0.02)

    def test_compatible_peak_keeps_fixed_stage(self) -> None:
        f0 = self.frequency[300]
        resistance = 2 * np.pi * f0 * 0.001 * 10
        measured = self.magnitude([6.8, 1e-7, resistance, f0, 10.0])
        fit, stages = fit_impedance_auto(self.frequency, measured)
        self.assertEqual(fit.sections, 1)
        self.assertEqual(len(stages), 2)
        self.assertFalse(fit.inductances_refined)
        self.assertLess(fit.max_abs_log_error, 0.05)

    def test_two_peaks_do_not_create_extra_sections(self) -> None:
        measured = self.magnitude([6.8, 0.0004, 30., 45., 4., 20., 100., 5.])
        fit, stages = fit_impedance_auto(self.frequency, measured, max_sections=10)
        self.assertEqual(fit.sections, 2)
        peaks, _ = find_peaks(measured, prominence=max(0.5, np.ptp(measured)*0.04), distance=3)
        np.testing.assert_array_equal(fit.physical_params[3::3], self.frequency[peaks])
        self.assertLess(fit.rms_log_error, 0.02)
        self.assertEqual([r.sections for r in stages], [0, 1, 2])

    def test_monotonic_curve_uses_only_series_rl(self) -> None:
        measured = self.magnitude([6.8, 0.0004])
        fit, stages = fit_impedance_auto(self.frequency, measured)
        self.assertEqual(fit.sections, 0)
        self.assertEqual(len(stages), 1)
        np.testing.assert_allclose(fit.physical_params, [6.8, 0.0004], rtol=1e-5)
        with self.assertRaisesRegex(ValueError, "more sections"):
            fit_impedance(self.frequency, measured, 1)

    def test_invalid_points_are_removed_before_peak_detection(self) -> None:
        measured = self.magnitude([6.8, 0.0004])
        frequency = np.r_[self.frequency[::-1], np.nan, -1]
        magnitude = np.r_[measured[::-1], 3, 0]
        fit, _ = fit_impedance_auto(frequency, magnitude)
        np.testing.assert_allclose(fit.physical_params, [6.8, 0.0004], rtol=1e-5)

    def test_evaluation_limit_is_not_reported_as_success(self) -> None:
        measured = self.magnitude([6.8, 0.0004, 30., 60., 3.])
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            fit_impedance_auto(self.frequency, measured, max_evaluations=1)

    def test_invalid_shapes_and_section_ranges(self) -> None:
        with self.assertRaisesRegex(ValueError, "matching 1-D"):
            fit_impedance_auto(self.frequency, np.ones((1024, 1)))
        with self.assertRaisesRegex(ValueError, "min_sections"):
            fit_impedance_auto(self.frequency, np.ones(1024), min_sections=2, max_sections=1)
        with self.assertRaisesRegex(ValueError, "Fewer resonance"):
            fit_impedance_auto(self.frequency, np.ones(1024), min_sections=1)
        for target in (0, -0.01, np.nan, np.inf):
            with self.assertRaisesRegex(ValueError, "target_log_error"):
                fit_impedance_auto(self.frequency, np.ones(1024), target_log_error=target)

    def test_looser_accuracy_stops_before_second_resonance(self) -> None:
        measured = self.magnitude([6.8, 0.0004, 30., 45., 4., 20., 100., 5.])
        loose, loose_candidates = fit_impedance_auto(
            self.frequency, measured, target_log_error=0.3
        )
        strict, strict_candidates = fit_impedance_auto(
            self.frequency, measured, target_log_error=0.02
        )
        self.assertEqual(loose.sections, 1)
        self.assertEqual(strict.sections, 2)
        self.assertEqual(loose.stop_reason, "target_reached")
        self.assertEqual(strict.stop_reason, "target_reached")
        self.assertEqual(len(loose_candidates), 2)
        self.assertEqual(len(strict_candidates), 3)
        self.assertLessEqual(strict.rms_log_error, 0.02)

    def test_section_limit_reports_unreached_accuracy(self) -> None:
        measured = self.magnitude([6.8, 0.0004, 30., 45., 4., 20., 100., 5.])
        fit, candidates = fit_impedance_auto(
            self.frequency, measured, target_log_error=0.001, max_sections=1
        )
        self.assertEqual([r.sections for r in candidates], [0, 1])
        self.assertEqual(fit.stop_reason, "section_limit")
        self.assertGreater(fit.rms_log_error, fit.target_log_error)

    def test_residual_can_add_sections_beyond_visible_peaks(self) -> None:
        measured = self.magnitude([6.8, 0.0004, 30., 45., 4., 20., 100., 5.])
        fit, candidates = fit_impedance_auto(
            self.frequency, measured, target_log_error=0.001, max_sections=4
        )
        self.assertGreater(len(candidates), 3)
        self.assertLessEqual(fit.rms_log_error, candidates[2].rms_log_error)


if __name__ == "__main__":
    unittest.main()
