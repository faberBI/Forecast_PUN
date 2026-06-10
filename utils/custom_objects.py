
# custom_objects.py
import numpy as np
import pandas as pd

# ====== STESSE COSTANTI USATE NEL TRAINING ======
OFFPEAK_WEIGHT_BASE = 1.0
PEAK_WEIGHT_BASE = 1.5
MIDDAY_WEIGHT = 2.5
EVENING_WEIGHT = 3.0

# 12:00 - 17:00  -> quarter of day da 48 a 67
MIDDAY_QOD = range(48, 68)

# 19:00 - 21:30 -> quarter of day da 76 a 86
EVENING_QOD = range(76, 87)

def peak_weight_func(index):
    """
    Peso più alto nei regimi problematici:
    - 12:00 - 17:00
    - 19:00 - 21:30
    fallback peak normale sulle altre ore diurne.
    """
    index = pd.DatetimeIndex(index)
    qod = index.hour * 4 + (index.minute // 15)

    weights = np.full(len(index), OFFPEAK_WEIGHT_BASE, dtype=float)

    # regime problematici
    weights[np.isin(qod, list(MIDDAY_QOD))] = MIDDAY_WEIGHT
    weights[np.isin(qod, list(EVENING_QOD))] = EVENING_WEIGHT

    # peak "normali" ma non critiche
    daytime_mask = (index.hour >= 8) & (index.hour < 21)
    weights[(weights == OFFPEAK_WEIGHT_BASE) & daytime_mask] = PEAK_WEIGHT_BASE

    return weights
