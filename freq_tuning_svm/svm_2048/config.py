# 경로 설정
DATA_DIR = "/Users/seohyeon/AT_freq_tuning/data_v3"
OUTPUT_DIR = "/Users/seohyeon/AT_freq_tuning/freq_tuning_svm/svm_2048/preprocessed_data"
MODEL_DIR = "/Users/seohyeon/AT_freq_tuning/freq_tuning_svm/svm_2048/model_results"

# 공진 주파수 (고정)
RESONANCE_FREQS = [40, 50, 60]

# 데이터 증강 설정
WINDOW_SIZE = 2048
STRIDE = 1024
AUGMENTATION_SNR_DB = [-10, -5, 5, 10]  # 증강할 SNR 레벨 (dB)
ENABLE_AUGMENTATION = True  # 증강 활성화 여부

# 데이터 분할 설정
TEST_SIZE = 0.2
RANDOM_STATE = 42

# SVM 하이퍼파라미터
SVM_CONFIG = {
    'kernel': 'rbf',
    'C': 10,
    'gamma': 'scale',
    'random_state': 42
}