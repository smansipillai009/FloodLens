import numpy as np

from app.services.s2_confirmation import classify_optical_agreement, cloud_and_clear_masks


def test_four_state_logic():
    sar = np.array([[1, 1, 1, 2, 2, 0]], np.uint8)
    clear = np.array([[1, 1, 0, 1, 1, 1]], bool)
    mndwi = np.array([[.4, -.2, .5, 0, 0, 0]], np.float32)
    ndvi = np.array([[0, 0, 0, .1, .6, 0]], np.float32)
    out = classify_optical_agreement(sar, clear, mndwi, ndvi, None, 0.0, 0.25)
    assert out.tolist() == [[2, 3, 1, 2, 3, 0]]   # confirmed, contradicted, SAR-only, confirmed, contradicted, none


def test_snow_shadow_cloud_are_not_clear():
    scl = np.array([[4, 9, 11, 2, 6, 3]], np.uint8)
    _, clear = cloud_and_clear_masks(scl, np.ones((1, 6), bool), 0)
    assert clear.tolist() == [[True, False, False, False, True, False]]
