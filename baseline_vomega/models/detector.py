"""
[TEMPLATE — 모델 슬롯이 비어 있습니다]

이 파일은 새 모델을 끼울 때 채워야 하는 인터페이스만 정의합니다.
train.py / inference.py 는 아래 세 함수를 그대로 호출하므로,
새 모델로 갈아끼울 때는 **함수 시그니처는 유지**하고 내부 구현만 바꾸세요.

참고 구현:
- Faster R-CNN: ../../baseline_v2.0/models/detector.py
"""


def build_model(num_classes, cfg):
    """
    새 모델 인스턴스를 생성해서 반환합니다.

    Args:
        num_classes: foreground 클래스 개수 (background는 모델 내부에서 처리)
        cfg: configs/default.yaml 을 dict로 로드한 것

    Returns:
        torch.nn.Module — train.py 가 곧바로 학습 루프에 넣을 수 있는 모델

    구현 시 체크리스트:
      - cfg["model"]["pretrained"] 반영
      - cfg["inference"]["max_detections"] / score_threshold / nms_threshold 반영
        (모델이 직접 지원하지 않으면 inference.py 단에서 후처리해도 됨)
      - 분류 헤드의 출력 클래스 수를 num_classes 에 맞게 교체
    """
    raise NotImplementedError(
        "baseline_v0 는 템플릿입니다. models/detector.py 의 build_model 을 구현하세요."
    )


def load_checkpoint(model, checkpoint_path, device):
    """체크포인트에서 가중치 + 메타데이터를 복원합니다.

    Returns: (model, epoch, best_map)
    """
    raise NotImplementedError


def save_checkpoint(model, optimizer, epoch, best_map, path):
    """현재 가중치 + optimizer 상태 + 메타데이터를 path 에 저장합니다."""
    raise NotImplementedError
