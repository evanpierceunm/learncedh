"""Frozen, versioned adoption calibration fitted on earlier chronological forecasts.
A predictive association with breadth of adoption, not causal pilot-skill adjustment.
"""
import json
from pathlib import Path
import numpy as np
from scipy.special import expit, logit

CALIBRATION = json.loads(Path(__file__).with_name('adoption-calibration.json').read_text())

def adoption_shift(pilots):
    if type(pilots) is not int or pilots < 0:
        raise ValueError('Pilot count must be a nonnegative integer')
    return (CALIBRATION['intercept'] + CALIBRATION['logPilotCoefficient'] *
            (np.log1p(pilots) - CALIBRATION['logPilotCenter']))

def calibrate(probability, shift):
    if not np.isfinite(probability) or not 0 < probability < 1 or not np.isfinite(shift):
        raise ValueError('Calibration requires an interior probability and finite shift')
    return float(expit(logit(probability) + shift))
