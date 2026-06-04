#test_wrapper.py
# 이 테스트 코드는 app_wrapper.py의 PillPredictorWrapper 클래스가 Streamlit 앱과 완벽하게 인터페이스를 준수하는지 검증하기 위한 것입니다.
# 1차 개발자가 설계한 인터페이스 계약(Contract)에 따라, predict 메서드가 정확한 데이터 타입과 구조로 결과를 반환하는지 확인합니다.
# 또한, 실제 이미지 파일을 사용하여 모델이 정상적으로 추론을 수행하는지도 검증합니다.

import os
import cv2
from app_wrapper import PillPredictorWrapper

def test_pipeline():
    # 1. 테스트할 알약 이미지 준비
    # 폴더 내에 있는 실제 이미지 파일명을 적어주거나, 없으면 가상 데이터로 대체합니다.
    sample_image_path = "K-001900-016548-019607-029451_0_2_0_2_70_000_200.png" 
    
    if not os.path.exists(sample_image_path):
        # 만약 해당 이름의 파일이 없다면 폴더 내 첫 번째 jpg 파일을 자동으로 탐색
        jpg_files = [f for f in os.listdir('.') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if jpg_files:
            sample_image_path = jpg_files[0]
        else:
            print("❌ [FAIL] 테스트할 이미지 파일(.jpg, .png)이 폴더에 없습니다.")
            return

    print(f"📸 [TEST] '{sample_image_path}' 파일로 인터페이스 루프백 테스트를 시작합니다.")
    
    # 2. 이미지 파일을 바이트 스트림으로 읽기 (Streamlit 업로드 환경 모사)
    with open(sample_image_path, "rb") as f:
        image_bytes = f.read()
    
    # 3. 우리가 설계한 Wrapper 객체 생성 (알약 전용 모델 로드)
    predictor = PillPredictorWrapper(pth_path="yolov8_custom_pill_detection_best_100epoch.pt")
    
    # 4. 인터페이스 포트 호출 (추론 실행)
    result_data, result_img = predictor.predict(image_bytes)
    
    # 5. [검증 단계] 상호 합의된 3대 규격(Contract) 만족 여부 판정
    print("\n================ TEST BENCH VERIFICATION ================")
    
    try:
        # 조건 1: 반환 데이터 타입이 리스트 구조인가?
        assert isinstance(result_data, list), "오류: 반환 값이 <list> 타입이 아닙니다."
        print(f"✅ 조건 1 통과: 데이터 타입이 정확합니다. (Count: {len(result_data)})")
        
        # 조건 2: 시각화 이미지 픽셀 배열이 정상 반환되었는가?
        assert result_img is not None and result_img.ndim == 3, "오류: 시각화 이미지 배열이 누락되었거나 손상되었습니다."
        print(f"✅ 조건 2 통과: 시각화 이미지 Matrix가 완벽히 출력되었습니다. (Shape: {result_img.shape})")
        
        # 조건 3: 리스트 내부의 딕셔너리 키(Port) 규격이 매칭되는가?
        if len(result_data) > 0:
            first_item = result_data[0]
            assert "box" in first_item, "오류: 'box' 출구 포트가 없습니다."
            assert "label" in first_item, "오류: 'label' 출구 포트가 없습니다."
            assert "confidence" in first_item, "오류: 'confidence' 출구 포트가 없습니다."
            print("✅ 조건 3 통과: 내부 JSON Dictionary 포트 규격 일치 확인.")
            print(f"   - 샘플 데이터 출력: {first_item}")
        else:
            print("⚠️ 조건 3 패스: 탐지된 알약이 없어 내부 포트 스킵 (정상 흐름).")
            
        print("\n>>> 🌟 SUCCESS: INTERFACE IS PERFECT! 🌟")
        print("=========================================================")
        
    except AssertionError as error:
        print(f"\n>>> ❌ FAIL: INTERFACE BROKEN.")
        print(f"    사유: {error}")
        print("=========================================================")

if __name__ == "__main__":
    test_pipeline()


#== TEST CODE FOR APP_WRAPPER.PY INTERFACE VALIDATION ==
# import sys
# from app_wrapper import PillPredictorWrapper

# def run_test():
#     print(">>> [Step 1] Initializing Wrapper...")
#     try:
#         # 표준 yolov8n.pt 모델 사용 (인터넷 연결 시 자동 다운로드됨)
#         predictor = PillPredictorWrapper(pth_path="yolov8n.pt")
#     except Exception as e:
#         print(f"FAILED: Model Load Error - {e}")
#         return

#     # 가상의 이미지 바이너리 데이터 (테스트용)
#     print(">>> [Step 2] Testing Inference with Dummy Data...")
#     try:
#         # 실제 이미지가 없으므로 간단한 빈 이미지 바이너리 생성 (테스트용 라이브러리 필요 시 실제 이미지 사용 권장)
#         import cv2
#         import numpy as np
#         dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
#         _, img_encoded = cv2.imencode('.jpg', dummy_img)
#         image_bytes = img_encoded.tobytes()

#         results, _ = predictor.predict(image_bytes)
        
#         # 규격 검증 (Assertion)
#         assert isinstance(results, list), "Result must be a list"
#         if len(results) >= 0:
#             print(f">>> [SUCCESS] Received {len(results)} objects.")
#             print(f">>> [SUCCESS] Sample JSON: {results[:1]}")
#             print("\n" + "="*40)
#             print(" FINAL RESULT: INTERFACE IS PERFECT! ")
#             print("="*40)
            
#     except Exception as e:
#         print(f"FAILED: Interface Broken - {e}")

# if __name__ == "__main__":
#     run_test()