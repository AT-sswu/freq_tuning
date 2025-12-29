import torch
import torch.nn as nn


class FrequencyTuningMLP(nn.Module):
    """
    주파수 튜닝을 위한 MLP 모델

    Input: 주파수 도메인 특징 (6차원)
    Output: 각 공진 주파수(30, 40, 50, 60Hz)에서의 예상 출력 전력 (4차원)
    """

    def __init__(self, input_dim=6, hidden_dims=[64, 128, 64], output_dim=4, dropout=0.3):
        super(FrequencyTuningMLP, self).__init__()

        layers = []
        prev_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def create_model(input_dim=6, output_dim=4, hidden_dims=[64, 128, 64], dropout=0.3):
    """
    모델 생성 함수
    """
    model = FrequencyTuningMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        dropout=dropout
    )
    return model


if __name__ == "__main__":
    # 모델 테스트
    model = create_model()
    print(model)

    # 더미 입력으로 테스트
    dummy_input = torch.randn(10, 6)  # (배치 크기 10, 특징 6개)
    output = model(dummy_input)
    print(f"\n입력 형상: {dummy_input.shape}")
    print(f"출력 형상: {output.shape}")
    print(f"출력 예시:\n{output[:3]}")