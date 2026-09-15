"""Tests for metrics.py — Levenshtein distance, char accuracy, sharpness, PSNR/SSIM."""
import numpy as np
import pytest
import metrics


class TestLevenshteinDistance:
    def test_identical_strings(self):
        assert metrics.levenshtein_distance("ABC", "ABC") == 0

    def test_empty_strings(self):
        assert metrics.levenshtein_distance("", "") == 0

    def test_one_empty(self):
        assert metrics.levenshtein_distance("ABC", "") == 3
        assert metrics.levenshtein_distance("", "XYZ") == 3

    def test_single_substitution(self):
        assert metrics.levenshtein_distance("ABC", "ABD") == 1

    def test_single_insertion(self):
        assert metrics.levenshtein_distance("ABC", "ABCD") == 1

    def test_single_deletion(self):
        assert metrics.levenshtein_distance("ABCD", "ABC") == 1

    def test_completely_different(self):
        assert metrics.levenshtein_distance("ABC", "XYZ") == 3

    def test_real_plate_off_by_one(self):
        # UP50AS4535 vs UPS0AS4535 — one character difference
        dist = metrics.levenshtein_distance("UP50AS4535", "UPS0AS4535")
        assert dist == 1


class TestCharAccuracy:
    def test_exact_match(self):
        result = metrics.char_accuracy("MH12AB1234", "MH12AB1234")
        assert result["exact_match"] is True
        assert result["char_similarity"] == 1.0
        assert result["edit_distance"] == 0

    def test_case_insensitive(self):
        result = metrics.char_accuracy("mh12ab1234", "MH12AB1234")
        assert result["exact_match"] is True

    def test_strips_whitespace_and_dashes(self):
        result = metrics.char_accuracy("MH-12-AB-1234", "MH12AB1234")
        assert result["exact_match"] is True

    def test_partial_match(self):
        result = metrics.char_accuracy("MH12AB1234", "MH12AB1235")
        assert result["exact_match"] is False
        assert result["edit_distance"] == 1
        assert result["char_similarity"] == 0.9

    def test_empty_prediction(self):
        result = metrics.char_accuracy("MH12AB1234", "")
        assert result["exact_match"] is False
        assert result["char_similarity"] == 0.0

    def test_empty_ground_truth(self):
        result = metrics.char_accuracy("", "ANYTHING")
        assert result["exact_match"] is False


class TestComputeSharpness:
    def test_sharp_image(self, blank_plate_image):
        s = metrics.compute_sharpness(blank_plate_image)
        assert s > 0

    def test_blurred_image_less_sharp(self, blank_plate_image):
        import cv2
        blurred = cv2.GaussianBlur(blank_plate_image, (21, 21), 0)
        s_sharp = metrics.compute_sharpness(blank_plate_image)
        s_blurred = metrics.compute_sharpness(blurred)
        assert s_sharp > s_blurred

    def test_empty_image(self):
        assert metrics.compute_sharpness(np.array([])) == 0.0

    def test_none_image(self):
        assert metrics.compute_sharpness(None) == 0.0

    def test_grayscale_input(self, small_gray_image):
        s = metrics.compute_sharpness(small_gray_image)
        assert s >= 0


class TestComputePsnrSsim:
    def test_identical_images(self, blank_plate_image):
        p, s = metrics.compute_psnr_ssim(blank_plate_image, blank_plate_image)
        # B8: PSNR should be clamped to finite value, not inf
        assert np.isfinite(p)
        assert p == 100.0  # _MAX_PSNR ceiling
        assert s == pytest.approx(1.0, abs=0.001)

    def test_different_images(self, blank_plate_image):
        noisy = blank_plate_image.copy()
        noisy = (noisy.astype(np.float32) + np.random.normal(0, 30, noisy.shape)).clip(0, 255).astype(np.uint8)
        p, s = metrics.compute_psnr_ssim(blank_plate_image, noisy)
        assert p > 0
        assert p < 100.0
        assert 0.0 < s < 1.0

    def test_empty_images(self):
        p, s = metrics.compute_psnr_ssim(np.array([]), np.array([]))
        assert p == 0.0
        assert s == 0.0

    def test_different_sizes(self, blank_plate_image):
        import cv2
        small = cv2.resize(blank_plate_image, (200, 60))
        p, s = metrics.compute_psnr_ssim(blank_plate_image, small)
        assert p > 0  # Should auto-resize and compute


class TestTimer:
    def test_timer_measures_elapsed(self):
        import time
        with metrics.timer("test") as t:
            time.sleep(0.05)
        assert t.elapsed >= 0.04
        assert t.elapsed < 1.0
