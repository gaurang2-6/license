"""
Shared pytest fixtures for the forensic license plate pipeline test suite.
"""
import os
import sys

import cv2
import numpy as np
import pytest

# Ensure src/ is importable
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


@pytest.fixture
def blank_plate_image():
    """A simple 400x120 white image with black text — simulates a clean plate."""
    img = np.full((120, 400, 3), 240, dtype=np.uint8)
    cv2.putText(img, "MH12AB1234", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (15, 15, 15), 4, cv2.LINE_AA)
    return img


@pytest.fixture
def small_gray_image():
    """A small 50x150 grayscale image for testing edge cases."""
    return np.random.randint(80, 200, (50, 150), dtype=np.uint8)


@pytest.fixture
def dark_frame():
    """An underexposed dark BGR frame."""
    return np.full((540, 960, 3), 20, dtype=np.uint8)


@pytest.fixture
def bright_frame():
    """A normal-exposure BGR frame."""
    return np.full((540, 960, 3), 140, dtype=np.uint8)


@pytest.fixture
def sample_scene_with_plate():
    """A synthetic scene with a plate-like region for detection testing."""
    scene = np.full((540, 960, 3), 100, dtype=np.uint8)
    # Draw a plate-like rectangle
    cv2.rectangle(scene, (375, 300), (615, 400), (240, 240, 240), -1)
    cv2.rectangle(scene, (380, 305), (610, 395), (20, 20, 20), 3)
    cv2.putText(scene, "TEST1234", (395, 370), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (15, 15, 15), 3, cv2.LINE_AA)
    return scene


@pytest.fixture
def quadrilateral_points():
    """Standard 4-point quadrilateral corners for perspective testing."""
    return np.array([[375, 300], [615, 300], [615, 400], [375, 400]], dtype=np.float32)
