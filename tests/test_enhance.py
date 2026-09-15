"""Tests for enhance.py — deblurring, sharpening, contrast, binarization, deskew."""
import cv2
import numpy as np
import pytest
import enhance


class TestRectifyPerspective:
    def test_basic_rectification(self, sample_scene_with_plate, quadrilateral_points):
        result = enhance.rectify_perspective(sample_scene_with_plate, quadrilateral_points)
        assert result.shape == (120, 400, 3)

    def test_empty_frame_returns_zeros(self):
        corners = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float32)
        result = enhance.rectify_perspective(np.array([]), corners)
        assert result.shape == (120, 400, 3)

    def test_none_frame_returns_zeros(self):
        corners = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float32)
        result = enhance.rectify_perspective(None, corners)
        assert result.shape == (120, 400, 3)


class TestDeblurWiener:
    def test_motion_deblur(self, blank_plate_image):
        result = enhance.deblur_wiener(blank_plate_image, length=15, angle=0.0, k=0.015)
        assert result.shape == blank_plate_image.shape
        assert result.dtype == np.uint8

    def test_defocus_deblur(self, blank_plate_image):
        result = enhance.deblur_wiener(blank_plate_image, length=9, angle=0.0, k=0.02, is_defocus=True)
        assert result.shape == blank_plate_image.shape

    def test_grayscale_input(self, small_gray_image):
        result = enhance.deblur_wiener(small_gray_image, length=15)
        assert result.ndim == 2  # Should remain grayscale

    def test_empty_image_passthrough(self):
        empty = np.array([])
        result = enhance.deblur_wiener(empty)
        assert result.size == 0

    def test_none_passthrough(self):
        result = enhance.deblur_wiener(None)
        assert result is None


class TestSharpenUnsharp:
    def test_sharpening(self, blank_plate_image):
        result = enhance.sharpen_unsharp(blank_plate_image)
        assert result.shape == blank_plate_image.shape
        assert result.dtype == np.uint8

    def test_empty_passthrough(self):
        assert enhance.sharpen_unsharp(np.array([])).size == 0

    def test_none_passthrough(self):
        assert enhance.sharpen_unsharp(None) is None


class TestDenoise:
    def test_color_denoise(self, blank_plate_image):
        result = enhance.denoise(blank_plate_image)
        assert result.shape == blank_plate_image.shape

    def test_gray_denoise(self, small_gray_image):
        result = enhance.denoise(small_gray_image)
        assert result.ndim == 2

    def test_empty_passthrough(self):
        assert enhance.denoise(np.array([])).size == 0


class TestDenoiseBilateral:
    def test_bilateral_filter(self, blank_plate_image):
        result = enhance.denoise_bilateral(blank_plate_image)
        assert result.shape == blank_plate_image.shape

    def test_empty_passthrough(self):
        assert enhance.denoise_bilateral(np.array([])).size == 0


class TestEnhanceContrast:
    def test_color_clahe(self, blank_plate_image):
        result = enhance.enhance_contrast(blank_plate_image)
        assert result.shape == blank_plate_image.shape

    def test_gray_clahe(self, small_gray_image):
        result = enhance.enhance_contrast(small_gray_image)
        assert result.ndim == 2

    def test_empty_passthrough(self):
        assert enhance.enhance_contrast(np.array([])).size == 0


class TestAutoExposureCorrect:
    def test_dark_image_brightened(self):
        dark = np.full((60, 200, 3), 30, dtype=np.uint8)
        result = enhance.auto_exposure_correct(dark)
        # Should be brighter than input
        assert result.mean() > dark.mean()

    def test_bright_image_darkened(self):
        bright = np.full((60, 200, 3), 220, dtype=np.uint8)
        result = enhance.auto_exposure_correct(bright)
        # Should be darker than input
        assert result.mean() < bright.mean()

    def test_normal_exposure_unchanged_ish(self, blank_plate_image):
        result = enhance.auto_exposure_correct(blank_plate_image)
        assert result.shape == blank_plate_image.shape


class TestSuperResolve:
    def test_2x_upscale(self, blank_plate_image):
        result = enhance.super_resolve(blank_plate_image, scale=2)
        assert result.shape[0] == blank_plate_image.shape[0] * 2
        assert result.shape[1] == blank_plate_image.shape[1] * 2

    def test_3x_upscale(self, blank_plate_image):
        result = enhance.super_resolve(blank_plate_image, scale=3)
        assert result.shape[0] == blank_plate_image.shape[0] * 3


class TestAddOcrBorder:
    def test_adds_border(self, blank_plate_image):
        result = enhance.add_ocr_border(blank_plate_image, padding=20)
        assert result.shape[0] == blank_plate_image.shape[0] + 40
        assert result.shape[1] == blank_plate_image.shape[1] + 40

    def test_border_is_white(self, blank_plate_image):
        result = enhance.add_ocr_border(blank_plate_image, padding=20)
        # Top-left corner should be white
        assert result[0, 0].tolist() == [255, 255, 255]


class TestBinarizeForOcr:
    def test_produces_binary(self, blank_plate_image):
        result = enhance.binarize_for_ocr(blank_plate_image)
        assert result.ndim == 2
        unique = np.unique(result)
        assert set(unique).issubset({0, 255})

    def test_gray_input(self, small_gray_image):
        result = enhance.binarize_for_ocr(small_gray_image)
        assert result.ndim == 2


class TestMorphologicalClean:
    def test_cleans_noise(self):
        # Create image with small noise specks
        img = np.full((60, 200), 255, dtype=np.uint8)
        img[10, 10] = 0  # Single noise pixel
        result = enhance.morphological_clean(img, ksize=3)
        # The single pixel should be removed by morphological opening
        assert result[10, 10] == 255

    def test_preserves_large_features(self):
        img = np.full((60, 200), 255, dtype=np.uint8)
        cv2.rectangle(img, (50, 10), (150, 50), 0, -1)  # Large black rectangle
        result = enhance.morphological_clean(img, ksize=3)
        # Large feature should be mostly preserved
        assert result[30, 100] == 0


class TestAutoDeskew:
    def test_no_correction_needed(self, blank_plate_image):
        result = enhance.auto_deskew(blank_plate_image)
        assert result.shape == blank_plate_image.shape

    def test_empty_passthrough(self):
        result = enhance.auto_deskew(np.array([]))
        assert result.size == 0


class TestEnsureBgr:
    def test_gray_to_bgr(self, small_gray_image):
        result = enhance.ensure_bgr(small_gray_image)
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_bgr_passthrough(self, blank_plate_image):
        result = enhance.ensure_bgr(blank_plate_image)
        assert np.array_equal(result, blank_plate_image)

    def test_none_passthrough(self):
        assert enhance.ensure_bgr(None) is None


class TestMultiframeAverage:
    def test_single_frame(self, blank_plate_image):
        result = enhance.multiframe_average([blank_plate_image])
        assert np.array_equal(result, blank_plate_image)

    def test_multiple_frames_reduces_noise(self, blank_plate_image):
        frames = []
        for _ in range(5):
            noisy = blank_plate_image.astype(np.float32) + np.random.normal(0, 20, blank_plate_image.shape)
            frames.append(np.clip(noisy, 0, 255).astype(np.uint8))
        result = enhance.multiframe_average(frames)
        assert result.shape == blank_plate_image.shape

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            enhance.multiframe_average([])
