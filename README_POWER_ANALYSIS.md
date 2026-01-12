# VEH Power Improvement Analysis - Quick Start Guide

## 📊 실행 방법

```bash
cd /Users/seohyeon/AT_freq_tuning
.venv/bin/python calculate_power_improvement.py
```

## 📁 생성되는 파일

1. **veh_power_improvement_comparison_1024.png** - 종합 비교 그래프
2. **veh_power_improvement_results_1024.csv** - 상세 수치 데이터
3. **VEH_Power_Improvement_Analysis_Summary.md** - 분석 결과 요약 문서

## 🎯 주요 결과 요약

| 모델 | 개선율 | 순위 |
|------|--------|------|
| RF | **+35.18%** | 🥇 |
| kNN | **+35.12%** | 🥈 |
| XGBoost | **+33.12%** | 🥉 |
| SVM | **+32.21%** | 4위 |

## 📈 분석 개요

- **매핑 전 평균 전력**: 253.7 mW
- **매핑 후 평균 전력**: 337.7 mW (RF)
- **평균 개선율**: ~34%

## 🔍 핵심 인사이트

✅ **모든 모델이 30% 이상의 전력 개선 달성**  
✅ **RF와 kNN이 최고 성능 (35%+ 개선)**  
✅ **주파수 매핑의 효과가 명확히 입증됨**

## 📝 전력 계산 공식

입력 주파수를 최적 공진 주파수로 매핑하여 VEH 전력 출력을 계산:

- **매핑 전**: 입력 주파수 직접 사용 → 주파수 불일치로 전력 손실
- **매핑 후**: 모델이 최적 공진 주파수로 매핑 → 전력 출력 향상

## 🛠️ 코드 구조

```python
# 1. 모델 로드
model, freq_test, y_pred_freq = load_model_predictions('svm', 1024)

# 2. 매핑 전 전력 계산
pre_mapping = calculate_pre_mapping_power(freq_test, [40, 50, 60])

# 3. 매핑 후 전력 계산
post_mapping = calculate_post_mapping_power(freq_test, y_pred_freq, [40, 50, 60])

# 4. 개선율 계산
improvement = (post_mapping['mean'] - pre_mapping['mean']) / pre_mapping['mean'] * 100
```

## 📚 참고 문서

- **VEH_Power_Improvement_Analysis_Summary.md**: 전체 분석 결과 상세 설명
- **calculate_power_improvement.py**: 소스 코드 및 주석

---

**작성일**: 2026년 1월 12일
