"""
XGBoost Configuration for Window Size 1024
"""

from pathlib import Path

# 디렉토리 경로
BASE_DIR = Path(__file__).parent
DATA_DIR = Path("/Users/seohyeon/AT_freq_tuning/data_v3")
OUTPUT_DIR = BASE_DIR / "preprocessed_data"
MODEL_DIR = BASE_DIR / "model_results"

# 데이터 설정
WINDOW_SIZE = 1024
STRIDE = WINDOW_SIZE // 2  # 512
RESONANCE_FREQS = [40, 50, 60]  # Hz
TEST_SIZE = 0.2
RANDOM_STATE = 42

# 증강 설정
ENABLE_AUGMENTATION = True
AUGMENTATION_SNR_DB = [15]  # SNR 값 리스트

# XGBoost 모델 설정
XGBOOST_CONFIG = {
    'max_depth': 5,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'objective': 'multi:softmax',
    'num_class': 3,
    'random_state': RANDOM_STATE,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 1,
}
