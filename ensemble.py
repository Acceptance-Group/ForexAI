import os
import pickle
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

ENSEMBLE_DIR = "models"
XGB_MODEL_PATH = os.path.join(ENSEMBLE_DIR, "dir_xgb.json")
LGBM_MODEL_PATH = os.path.join(ENSEMBLE_DIR, "dir_lgbm.txt")
CB_MODEL_PATH = os.path.join(ENSEMBLE_DIR, "dir_cb.cbm")
SCALER_PATH = "models/scaler.pkl"

_dir_xgb = None
_dir_lgbm = None
_dir_cb = None
_dir_scaler = None


def _load_ensemble():
    global _dir_xgb, _dir_lgbm, _dir_cb, _dir_scaler
    if _dir_xgb is not None:
        return _dir_xgb, _dir_lgbm, _dir_cb, _dir_scaler

    _dir_xgb = xgb.XGBClassifier()
    _dir_xgb.load_model(XGB_MODEL_PATH)

    _dir_lgbm = lgb.Booster(model_file=LGBM_MODEL_PATH)

    _dir_cb = CatBoostClassifier()
    _dir_cb.load_model(CB_MODEL_PATH)

    with open(SCALER_PATH, "rb") as f:
        _dir_scaler = pickle.load(f)

    return _dir_xgb, _dir_lgbm, _dir_cb, _dir_scaler


def predict_direction_proba(features):
    m_xgb, m_lgbm, m_cb, scaler = _load_ensemble()

    if features.ndim == 1:
        features = features.reshape(1, -1)

    scaled = scaler.transform(features)

    p_xgb = m_xgb.predict_proba(scaled)[:, 1]

    p_lgbm = m_lgbm.predict(scaled)
    if p_lgbm.ndim > 1:
        p_lgbm = p_lgbm[:, 1]
    else:
        p_lgbm = 1.0 / (1.0 + np.exp(-p_lgbm))

    p_cb = m_cb.predict_proba(scaled)[:, 1]

    p_ens = (p_xgb + p_lgbm + p_cb) / 3.0

    return p_ens, {"xgb": p_xgb, "lgbm": p_lgbm, "cb": p_cb}


def predict_direction_proba_all(features):
    m_xgb, m_lgbm, m_cb, scaler = _load_ensemble()

    if features.ndim == 1:
        features = features.reshape(1, -1)

    scaled = scaler.transform(features)

    p_xgb = m_xgb.predict_proba(scaled)[:, 1]

    p_lgbm = m_lgbm.predict(scaled)
    if p_lgbm.ndim > 1:
        p_lgbm = p_lgbm[:, 1]
    else:
        p_lgbm = 1.0 / (1.0 + np.exp(-p_lgbm))

    p_cb = m_cb.predict_proba(scaled)[:, 1]

    p_ens = (p_xgb + p_lgbm + p_cb) / 3.0

    return p_ens, {"xgb": p_xgb, "lgbm": p_lgbm, "cb": p_cb}
