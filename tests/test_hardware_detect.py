import unittest
from unittest.mock import patch

from neural_cost.hardware import HardwareSpec
from neural_cost.hardware_detect import (
    DetectionResult,
    _measure_bandwidth_gb_s,
    _probe_apple_silicon,
    _probe_cpu_flops,
    detect_hardware,
)


class HardwareDetectTests(unittest.TestCase):
    def test_measure_bandwidth_gb_s(self):
        # Can run real since it's fast
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.assertIsNone(_measure_bandwidth_gb_s(size_mb=1, repeats=2))
            return
        bw = _measure_bandwidth_gb_s(size_mb=1, repeats=2)
        self.assertGreater(bw, 0.0)
        self.assertIsInstance(bw, float)

    @patch("shutil.which")
    @patch("subprocess.check_output")
    def test_probe_apple_silicon_m3(self, mock_check_output, mock_which):
        mock_which.return_value = "/usr/sbin/system_profiler"
        mock_check_output.return_value = "Chip: Apple M3\n"
        chip_name, flops, bw, caches = _probe_apple_silicon()
        self.assertEqual(chip_name, "Apple M3")
        self.assertEqual(flops, 3.6 * 1e12)
        self.assertEqual(bw, 100.0 * 1e9)
        self.assertEqual(len(caches), 1)
        self.assertEqual(caches[0].name, "SLC")

    @patch("shutil.which")
    @patch("subprocess.check_output")
    def test_probe_apple_silicon_m3_pro(self, mock_check_output, mock_which):
        mock_which.return_value = "/usr/sbin/system_profiler"
        mock_check_output.return_value = "Chip: Apple M3 Pro\n"
        chip_name, flops, bw, caches = _probe_apple_silicon()
        self.assertEqual(chip_name, "Apple M3 Pro")
        self.assertEqual(flops, 7.4 * 1e12)
        self.assertEqual(bw, 150.0 * 1e9)
        self.assertEqual(len(caches), 1)

    @patch("shutil.which")
    @patch("subprocess.check_output")
    def test_probe_apple_silicon_unknown(self, mock_check_output, mock_which):
        mock_which.return_value = "/usr/sbin/system_profiler"
        mock_check_output.return_value = "Chip: Apple M5 Ultra\n"
        chip_name, flops, bw, caches = _probe_apple_silicon()
        self.assertEqual(chip_name, "Apple M5 Ultra")
        self.assertIsNone(flops)
        self.assertIsNone(bw)
        self.assertEqual(caches, ())

    @patch("shutil.which")
    def test_probe_apple_silicon_no_profiler(self, mock_which):
        mock_which.return_value = None
        chip_name, flops, bw, caches = _probe_apple_silicon()
        self.assertIsNone(chip_name)
        self.assertIsNone(flops)
        self.assertIsNone(bw)
        self.assertEqual(caches, ())

    @patch("os.cpu_count")
    @patch("shutil.which")
    @patch("subprocess.check_output")
    def test_probe_cpu_flops(self, mock_check_output, mock_which, mock_cpu_count):
        mock_cpu_count.return_value = 4
        mock_which.return_value = "/usr/sbin/sysctl"
        mock_check_output.return_value = "2400000000\n"
        cores, clock_hz = _probe_cpu_flops()
        self.assertEqual(cores, 4)
        self.assertEqual(clock_hz, 2400000000.0)

    @patch("neural_cost.hardware_detect._measure_bandwidth_gb_s")
    @patch("neural_cost.hardware_detect._probe_apple_silicon")
    @patch("neural_cost.hardware_detect._probe_cpu_flops")
    def test_detect_hardware(self, mock_cpu, mock_apple, mock_measure):
        from neural_cost.hardware import CacheSpec

        mock_measure.return_value = 100.0
        mock_apple.return_value = (
            "Apple M3",
            3.6e12,
            100e9,
            (CacheSpec("SLC", 250e9, 8 * 1024 * 1024),),
        )
        mock_cpu.return_value = (8, 3e9)

        spec, result = detect_hardware(bandwidth_benchmark_mb=1)
        self.assertIsInstance(spec, HardwareSpec)
        self.assertIsInstance(result, DetectionResult)
        self.assertEqual(spec.name, "Apple M3")
        self.assertEqual(spec.peak_flops, 3.6e12)
        self.assertEqual(spec.memory_bandwidth, 100e9)
        self.assertEqual(len(spec.caches), 1)
        self.assertEqual(result.chip_name, "Apple M3")


if __name__ == "__main__":
    unittest.main()
