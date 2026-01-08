# 경로 설정
DATA_DIR = "/Users/seohyeon/AT_freq_tuning/data_v3"
OUTPUT_DIR = "/Users/seohyeon/AT_freq_tuning/freq_tuning_svm/preprocessed_data"
MODEL_DIR = "/Users/seohyeon/AT_freq_tuning/freq_tuning_svm/model_results"

# 공진 주파수 (고정)
RESONANCE_FREQS = [40, 50, 60]

# 데이터 증강 설정
WINDOW_SIZE = 8192
STRIDE = 4096
NUM_AUGMENTATIONS = 2

# SVM 하이퍼파라미터
SVM_CONFIG = {
    'kernel': 'rbf',
    'C': 10,
    'gamma': 'scale',
    'random_state': 42
}