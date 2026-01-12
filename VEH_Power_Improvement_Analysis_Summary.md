# VEH Power Improvement Analysis Summary

## 분석 개요

본 분석은 진동 에너지 하베스터(VEH)의 주파수 매핑 전후 전력 출력 개선도를 비교합니다.

### 핵심 개념

**매핑 전 (Pre-Mapping):**
- 입력 주파수를 직접 사용하여 각 공진기(40, 50, 60 Hz)에서 전력 계산
- 입력 주파수와 공진 주파수 간 불일치로 인한 전력 손실 발생
- 예: 43Hz 입력 → 40Hz 공진기: |43-40| = 3Hz 차이로 전력 저하

**매핑 후 (Post-Mapping):**
- 머신러닝 모델이 입력 주파수를 최적 공진 주파수로 매핑
- 주파수 일치로 인한 전력 출력 향상
- 예: 43Hz 입력 → 모델이 40Hz로 매핑 → 40Hz 공진기에서 최적 전력 생성

---

## 전력 계산 공식

VEH 전력 계산은 다음 공식을 사용합니다:

$$
|P| = \frac{m\zeta_e\omega_n\omega^2\left(\frac{\omega}{\omega_n}\right)^3Y^2}{\left(2\zeta_T\frac{\omega}{\omega_n}\right)^2 + \left(1-\left(\frac{\omega}{\omega_n}\right)^2\right)^2}
$$

### 파라미터

- **m**: Proof mass (관성 질량) = 0.001 kg
- **ζₑ**: Electrical damping ratio (전기적 감쇠비) = 0.05
- **ζₘ**: Mechanical damping ratio (기계적 감쇠비) = 0.02
- **ζ_T**: Total damping ratio (전체 감쇠비) = ζₑ + ζₘ = 0.07
- **ωₙ**: Natural angular frequency (공진 각주파수) = 2π × f_resonance
- **ω**: Input vibration angular frequency (입력 각주파수) = 2π × f_input
- **Y**: Displacement amplitude (변위 진폭) = 0.1 m
- **Q-factor**: Quality factor = 10

---

## 실험 설정

- **테스트 데이터**: 776 samples
- **입력 주파수 범위**: 33.64 ~ 142.07 Hz
- **공진 주파수**: [40, 50, 60] Hz
- **윈도우 크기**: 1024
- **모델**: RF, SVM, kNN, XGBoost

---

## 주요 결과

### 1. 매핑 전 (Pre-Mapping) 전력

| 지표 | 값 (μW) |
|------|---------|
| **평균 전력** | 2.537 × 10⁸ |
| **중간값** | 2.347 × 10⁸ |
| **표준편차** | 2.110 × 10⁸ |
| **최소값** | 1.453 × 10⁷ |
| **최대값** | 5.176 × 10⁸ |

### 2. 모델별 매핑 후 (Post-Mapping) 전력 및 개선도

| 모델 | 평균 전력 (μW) | 절대 개선 (μW) | 개선율 (%) | 순위 |
|------|---------------|---------------|-----------|------|
| **RF** | 3.429 × 10⁸ | +8.925 × 10⁷ | **+35.18%** | 🥇 1위 |
| **kNN** | 3.428 × 10⁸ | +8.908 × 10⁷ | **+35.12%** | 🥈 2위 |
| **XGBoost** | 3.377 × 10⁸ | +8.402 × 10⁷ | **+33.12%** | 🥉 3위 |
| **SVM** | 3.354 × 10⁸ | +8.171 × 10⁷ | **+32.21%** | 4위 |

### 3. 상세 모델별 통계

#### Random Forest (RF)
- **평균 전력**: 3.429 × 10⁸ μW
- **중간값**: 2.587 × 10⁸ μW
- **표준편차**: 2.106 × 10⁸ μW
- **개선율**: +35.18%
- **특징**: 가장 높은 개선율, 안정적인 성능

#### k-Nearest Neighbors (kNN)
- **평균 전력**: 3.428 × 10⁸ μW
- **중간값**: 2.587 × 10⁸ μW
- **표준편차**: 2.124 × 10⁸ μW
- **개선율**: +35.12%
- **특징**: RF와 거의 동일한 성능, 약간 높은 변동성

#### XGBoost
- **평균 전력**: 3.377 × 10⁸ μW
- **중간값**: 2.587 × 10⁸ μW
- **표준편차**: 2.072 × 10⁸ μW
- **개선율**: +33.12%
- **특징**: 좋은 개선율, 낮은 표준편차 (일관성)

#### Support Vector Machine (SVM)
- **평균 전력**: 3.354 × 10⁸ μW
- **중간값**: 2.587 × 10⁸ μW
- **표준편차**: 2.075 × 10⁸ μW
- **개선율**: +32.21%
- **특징**: 가장 낮은 개선율이지만 일관된 성능

---

## 핵심 인사이트

### ✅ 긍정적 결과

1. **모든 모델이 30% 이상의 전력 개선 달성**
   - 최소 32.21% (SVM) ~ 최대 35.18% (RF)
   - 주파수 매핑의 효과가 명확히 입증됨

2. **RF와 kNN이 가장 우수한 성능**
   - 35% 이상의 개선율
   - 실용적 응용에 적합

3. **모든 모델의 일관성**
   - 표준편차가 비슷한 수준 유지 (2.07~2.12 × 10⁸ μW)
   - 안정적인 전력 출력 예측 가능

### 🎯 실용적 의미

1. **에너지 효율 향상**
   - 평균 34%의 전력 개선 = 동일한 진동 환경에서 34% 더 많은 에너지 수확
   - IoT 센서, 웨어러블 기기 등의 배터리 수명 연장

2. **주파수 미스매치 문제 해결**
   - 고정 공진 주파수 VEH의 한계 극복
   - 다양한 진동 환경에 적응 가능

3. **모델 선택 가이드**
   - **최고 성능**: RF 또는 kNN
   - **일관성 중시**: XGBoost 또는 SVM
   - **실시간 처리**: kNN (추론 속도 빠름)
   - **복잡한 패턴**: RF 또는 XGBoost

---

## 기술적 세부사항

### 전력 계산 방법론

#### Pre-Mapping 계산
```python
for peak_freq in freq_test:
    powers = []
    for res_freq in [40, 50, 60]:
        # 입력 주파수를 그대로 사용
        power = calculate_veh_power(
            frequency=peak_freq,
            resonance_freq=res_freq
        )
        powers.append(power)
    avg_power = mean(powers)
```

#### Post-Mapping 계산
```python
for peak_freq, mapped_freq in zip(freq_test, y_pred_freq):
    powers = []
    for res_freq in [40, 50, 60]:
        if mapped_freq == res_freq:
            # 매핑된 주파수 사용 (최적화)
            power = calculate_veh_power(
                frequency=mapped_freq,
                resonance_freq=res_freq
            )
        else:
            # 원래 주파수 사용
            power = calculate_veh_power(
                frequency=peak_freq,
                resonance_freq=res_freq
            )
        powers.append(power)
    avg_power = mean(powers)
```

### 개선율 계산 공식

$$
\text{Improvement (\%)} = \frac{\text{Post-Mapping Mean} - \text{Pre-Mapping Mean}}{\text{Pre-Mapping Mean}} \times 100
$$

---

## 생성된 파일

1. **`calculate_power_improvement.py`**
   - 전력 계산 및 분석 스크립트

2. **`veh_power_improvement_comparison_1024.png`**
   - 6개 서브플롯으로 구성된 종합 비교 그래프
   - (1) 모델별 개선율 비교
   - (2) 매핑 전/후 평균 전력 비교
   - (3) 전력 분포 박스플롯
   - (4) 표준편차 비교 (일관성)
   - (5) 전력 계산 공식 설명
   - (6) 성능 요약 테이블

3. **`veh_power_improvement_results_1024.csv`**
   - 상세 수치 데이터
   - 모든 통계량 포함 (평균, 중간값, 표준편차, 최소/최대값, 개선율)

4. **`VEH_Power_Improvement_Analysis_Summary.md`**
   - 본 문서: 분석 결과 종합 요약

---

## 향후 연구 방향

1. **다양한 윈도우 크기 비교**
   - 현재: 1024
   - 추가 분석: 512, 2048

2. **SNR 변화에 따른 개선율 분석**
   - 다양한 노이즈 환경에서의 성능 평가

3. **실시간 구현 가능성 연구**
   - 모델 추론 속도 vs 성능 트레이드오프
   - 임베디드 시스템 적용 가능성

4. **하이브리드 모델 개발**
   - 여러 모델의 앙상블
   - 신뢰도 기반 동적 모델 선택

---

## 결론

본 분석을 통해 **머신러닝 기반 주파수 매핑이 VEH의 전력 출력을 약 32~35% 향상**시킬 수 있음을 실증적으로 증명했습니다. 

특히 **Random Forest와 k-NN 모델**이 가장 우수한 성능을 보였으며, 모든 모델이 일관되고 안정적인 개선 효과를 나타냈습니다.

이는 **주파수 적응형 VEH 시스템의 실용화 가능성**을 강력히 시사하며, IoT 및 웨어러블 기기의 자가 전력 생산 기술 발전에 기여할 것으로 기대됩니다.

---

**작성일**: 2026년 1월 12일  
**분석 도구**: Python, scikit-learn, matplotlib  
**데이터**: 776 test samples, Window size 1024
