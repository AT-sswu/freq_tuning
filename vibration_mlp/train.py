import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mlp_model import create_model


class VibrationDataset(Dataset):
    """진동 데이터셋"""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_model(data_path, save_dir, epochs=100, batch_size=16, lr=0.001):
    """
    MLP 모델 학습
    """
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")

    # 데이터 로드
    data = np.load(data_path)
    X = data['X']
    y = data['y']

    print(f"\n데이터 형상:")
    print(f"  X: {X.shape}")
    print(f"  y: {y.shape}")

    # 정규화
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_normalized = (X - X_mean) / (X_std + 1e-8)

    y_mean, y_std = y.mean(axis=0), y.std(axis=0)
    y_normalized = (y - y_mean) / (y_std + 1e-8)

    # 데이터셋 생성
    dataset = VibrationDataset(X_normalized, y_normalized)

    # Train/Val 분할 (80:20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"\n학습 데이터: {train_size}개")
    print(f"검증 데이터: {val_size}개")

    # 모델 생성
    model = create_model(input_dim=X.shape[1], output_dim=y.shape[1])
    model = model.to(device)

    # 손실 함수 및 옵티마이저
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    # 학습 루프
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    save_path = Path(save_dir) / "best_model.pth"

    print(f"\n{'=' * 60}")
    print(f"학습 시작 (에폭: {epochs}, 배치 크기: {batch_size}, 학습률: {lr})")
    print(f"{'=' * 60}")

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Learning rate 조정
        scheduler.step(val_loss)

        # 로그 출력
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # 최상의 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'X_mean': X_mean,
                'X_std': X_std,
                'y_mean': y_mean,
                'y_std': y_std
            }, save_path)

    print(f"\n{'=' * 60}")
    print(f"학습 완료! 최상의 검증 손실: {best_val_loss:.4f}")
    print(f"모델 저장 위치: {save_path}")
    print(f"{'=' * 60}")

    # 손실 그래프 저장
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = Path(save_dir) / "training_loss.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"손실 그래프 저장: {plot_path}")

    return model, train_losses, val_losses


if __name__ == "__main__":
    DATA_PATH = "/Users/seohyeon/AT_freq_tuning/fft_analysis/results/preprocessed_data_log.npz"
    SAVE_DIR = "/Users/seohyeon/AT_freq_tuning/model_results"

    model, train_losses, val_losses = train_model(
        data_path=DATA_PATH,
        save_dir=SAVE_DIR,
        epochs=200,
        batch_size=16,
        lr=0.001
    )