import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
from pathlib import Path


# 저역통과 필터 함수
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, data)


# 샘플링 레이트 계산 함수
def calculate_sample_rate(df, time_column='Time_us'):
    if time_column not in df.columns:
        time_column = df.columns[0]

    time_data = df[time_column].dropna().values

    if len(time_data) < 2:
        raise ValueError("시간 데이터가 충분하지 않습니다.")

    time_diffs = np.diff(time_data)
    avg_time_diff_us = np.mean(time_diffs)
    avg_time_diff_sec = avg_time_diff_us / 1_000_000
    sample_rate = 1 / avg_time_diff_sec

    return sample_rate


# Threshold 계산 함수
def calculate_threshold(amps, method="std", n_std=2.75):
    if method == "std":
        mean = np.mean(amps)
        std = np.std(amps)
        threshold = mean + n_std * std
    elif method == "percentile":
        threshold = np.percentile(amps, 97.5)
    else:
        raise ValueError("지원하지 않는 threshold 방법입니다.")
    return threshold


# FFT 분석 함수
def fft_analysis(data, sample_rate=296, fft_size=None, threshold_method="std", n_std=2.75):
    data = data - np.mean(data)
    n = fft_size if fft_size else len(data)
    data = data[:n]

    y = fft(data)
    x = fftfreq(n, 1 / sample_rate)
    positive_freqs = x[:n // 2]

    # 원본 Amplitude 계산
    positive_amps = np.abs(y[:n // 2]) * 2 / n

    # 정규화: 최대값을 100으로 스케일링
    max_amp = np.max(positive_amps)
    if max_amp > 0:
        positive_amps_normalized = (positive_amps / max_amp) * 100
    else:
        positive_amps_normalized = positive_amps

    # 공진 주파수
    resonance_freq = positive_freqs[np.argmax(positive_amps_normalized)]

    # Threshold 계산
    threshold = calculate_threshold(positive_amps_normalized, threshold_method, n_std)

    # Threshold 이상 구간 탐색
    threshold_ranges = []
    above_threshold = positive_amps_normalized >= threshold
    in_range = False
    for i in range(len(positive_freqs)):
        if above_threshold[i] and not in_range:
            range_start = positive_freqs[i]
            in_range = True
        elif not above_threshold[i] and in_range:
            range_end = positive_freqs[i - 1]
            threshold_ranges.append((range_start, range_end))
            in_range = False
    if in_range:
        threshold_ranges.append((range_start, positive_freqs[-1]))

    return positive_freqs, positive_amps_normalized, resonance_freq, threshold_ranges, threshold, positive_amps


# 30-60Hz 대역 에너지 계산
def calculate_band_energy(freqs, amps, band_ranges=[(30, 35), (35, 45), (45, 55), (55, 60)]):
    """
    각 주파수 대역의 에너지 비율 계산
    """
    band_energies = {}

    for band_start, band_end in band_ranges:
        mask = (freqs >= band_start) & (freqs <= band_end)
        band_energy = np.sum(amps[mask] ** 2)
        band_energies[f"{band_start}-{band_end}Hz"] = band_energy

    return band_energies


# 단일 파일 분석 함수 (Accel_Z만)
def analyze_single_file(
        file_path,
        output_dir,
        plot_dir,
        fft_size=8192,
        apply_filter=True,
        filter_order=5,
        threshold_method="std",
        n_std=2.75,
        time_column='Time_us'
):
    df = pd.read_csv(file_path)
    file_title = os.path.splitext(os.path.basename(file_path))[0]

    # Accel_Z 확인
    if 'Accel_Z' not in df.columns:
        print(f"[경고] Accel_Z 열이 CSV 파일에 없습니다: {file_path}")
        return None

    # 샘플링 레이트 자동 계산
    sample_rate = calculate_sample_rate(df, time_column)
    print(f"  계산된 샘플링 레이트: {sample_rate:.2f} Hz")

    data = df['Accel_Z'].dropna().values

    if apply_filter:
        cutoff = sample_rate / 4
        data = butter_lowpass_filter(data, cutoff=cutoff, fs=sample_rate, order=filter_order)

    freqs, amps_normalized, resonance_freq, threshold_ranges, threshold, amps_raw = fft_analysis(
        data,
        sample_rate=sample_rate,
        fft_size=fft_size,
        threshold_method=threshold_method,
        n_std=n_std
    )

    # 30-60Hz 대역 에너지 계산
    band_energies = calculate_band_energy(freqs, amps_raw)

    # 최대 에너지 대역 찾기
    max_band = max(band_energies, key=band_energies.get)
    max_energy = band_energies[max_band]

    # Threshold 범위 문자열 변환
    threshold_ranges_str = ""
    if threshold_ranges:
        ranges_list = [f"{r[0]:.2f}-{r[1]:.2f}Hz" for r in threshold_ranges]
        threshold_ranges_str = "; ".join(ranges_list)
    else:
        threshold_ranges_str = "없음"

    result = {
        'File': file_title,
        'Axis': 'Accel_Z',
        'Resonance_Frequency_Hz': round(resonance_freq, 2),
        'Threshold_Value': round(threshold, 4),
        'Threshold_Method': threshold_method,
        'Threshold_Ranges': threshold_ranges_str,
        'Sample_Rate': round(sample_rate, 2),
        'FFT_Size': fft_size,
        'Filter_Applied': apply_filter,
        'Filter_Order': filter_order if apply_filter else 'N/A',
        'Max_Energy_Band': max_band,
        'Max_Energy_Value': round(max_energy, 4),
        **{f'Energy_{k}': round(v, 4) for k, v in band_energies.items()}
    }

    # 그래프 그리기
    plt.figure(figsize=(12, 6))

    plt.plot(freqs, amps_normalized, label="Amplitude", linewidth=0.8)
    plt.axvline(resonance_freq, color='r', linestyle='--', label=f'Resonance: {resonance_freq:.2f} Hz')
    plt.axhline(y=threshold, color='g', linestyle=':', label=f'Threshold: {threshold:.3f}')

    # 30-60Hz 영역 강조
    plt.axvspan(30, 60, alpha=0.2, color='yellow', label='30-60Hz Band')

    plt.title(f"FFT Spectrum - Accel_Z - {file_title}\nMax Energy: {max_band} ({max_energy:.2f})")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (Normalized %)")
    plt.xlim(0, 150)  # 0-150Hz 범위만 표시
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # 그래프 저장
    plot_path = os.path.join(plot_dir, f"{file_title}_accel_z_fft.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    return result


# 배치 분석 함수
def analyze_all_files(
        data_dir,
        fft_size=8192,
        apply_filter=True,
        filter_order=5,
        threshold_method="std",
        n_std=2.75,
        time_column='Time_us'
):
    """
    data_v3 폴더의 모든 CSV 파일에 대해 Accel_Z FFT 분석 수행
    """
    data_path = Path(data_dir)

    # results 폴더 생성
    results_dir = data_path / "results"
    results_dir.mkdir(exist_ok=True)

    # fft_plots 폴더 생성
    plot_dir = results_dir / "fft_plots"
    plot_dir.mkdir(exist_ok=True)

    print(f"결과 저장 폴더: {results_dir}")
    print(f"그래프 저장 폴더: {plot_dir}")

    # CSV 파일 찾기
    csv_files = sorted(list(data_path.glob("*.csv")))

    if not csv_files:
        print(f"[오류] {data_dir}에서 CSV 파일을 찾을 수 없습니다.")
        return None

    print(f"\n총 {len(csv_files)}개의 파일을 찾았습니다.")
    print(f"[파일 목록]")
    for f in csv_files:
        print(f"  - {f.name}")

    all_results = []

    print(f"\n{'=' * 60}")
    print(f"Accel_Z FFT 분석 시작 ({len(csv_files)}개 파일)")
    print(f"{'=' * 60}")

    # 모든 파일 처리
    for idx, file_path in enumerate(csv_files, 1):
        print(f"\n[{idx}/{len(csv_files)}] 분석 중: {file_path.name}")

        try:
            result = analyze_single_file(
                file_path=str(file_path),
                output_dir=str(results_dir),
                plot_dir=str(plot_dir),
                fft_size=fft_size,
                apply_filter=apply_filter,
                filter_order=filter_order,
                threshold_method=threshold_method,
                n_std=n_std,
                time_column=time_column
            )

            if result:
                all_results.append(result)
                print(f"  ✓ 완료 - 최대 에너지 대역: {result['Max_Energy_Band']}")

        except Exception as e:
            import traceback
            print(f"  ✗ 오류 발생: {e}")
            traceback.print_exc()
            continue

    # 전체 결과 CSV 저장
    if all_results:
        results_df = pd.DataFrame(all_results)

        output_path = results_dir / "fft_analysis_accel_z.csv"
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')

        print(f"\n{'=' * 60}")
        print(f"[완료] 전체 분석 결과 저장됨: {output_path}")
        print(f"총 {len(all_results)}개의 파일 분석 완료")
        print(f"{'=' * 60}")

        # 요약 통계
        print("\n[분석 요약]")
        print(f"  분석된 파일 수: {len(all_results)}개")
        print(f"  축: Accel_Z")
        print(f"  그래프 저장 위치: {plot_dir}")

        # 대역별 최대 에너지 통계
        print("\n[대역별 최대 에너지 분포]")
        band_counts = results_df['Max_Energy_Band'].value_counts()
        for band, count in band_counts.items():
            print(f"  {band}: {count}개 파일")

        return results_df
    else:
        print("\n[오류] 분석된 결과가 없습니다.")
        return None


# 실행
if __name__ == "__main__":
    DATA_DIR = "/Users/seohyeon/AT_freq_tuning/data_v3"

    results = analyze_all_files(
        data_dir=DATA_DIR,
        fft_size=8192,
        apply_filter=True,
        filter_order=5,
        threshold_method="std",
        n_std=2.75,
        time_column='Time_us'
    )

    if results is not None:
        print("\n[결과 미리보기]")
        print(results.head(10))