import numpy as np

from app.services.flood_detection import adaptive_thresholds, robust_sigma

BASE = {"water_change_db_vv": -3.0, "water_grow_change_db_vv": -1.5, "debris_change_db_vh": 3.0, "debris_grow_change_db_vh": 1.5,
        "adaptive_k_seed": 3.0, "adaptive_k_grow": 1.75}


def test_robust_sigma_ignores_a_real_flood_patch():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1.3, (300, 300)); truth = np.zeros_like(x, bool); truth[100:130, 100:200] = True
    x[truth] -= 8.0                                                        # 3% of the pixels carry a strong real change
    s = robust_sigma(x, np.ones_like(x, bool))
    assert 1.2 < s < 1.45


def test_noisy_scenes_get_stricter_thresholds_but_never_weaker_than_config():
    quiet = adaptive_thresholds(BASE, 0.6, 0.6)
    assert quiet["water_change_db_vv"] == -3.0 and quiet["debris_change_db_vh"] == 3.0       # floor = configured values
    noisy = adaptive_thresholds(BASE, 1.3, 1.4)
    assert noisy["water_change_db_vv"] < -3.8 and noisy["debris_change_db_vh"] > 4.1
    assert noisy["water_grow_change_db_vv"] < -2.2 and noisy["debris_grow_change_db_vh"] > 2.4


def test_false_alarm_rate_is_roughly_constant_across_noise_levels():
    rng = np.random.default_rng(1)
    rates = []
    for sigma in (0.7, 1.3, 2.0):
        x = rng.normal(0, sigma, (400, 400))
        s = robust_sigma(x, np.ones_like(x, bool))
        thr = adaptive_thresholds(BASE, s, s)["debris_change_db_vh"]
        rates.append((x > thr).mean())
    assert max(rates[1:]) < 0.005 and abs(rates[1] - rates[2]) < 0.003                       # ~3 sigma seeds => ~0.1-0.2 % false alarms
