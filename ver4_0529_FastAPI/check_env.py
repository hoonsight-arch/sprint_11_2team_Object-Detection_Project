import importlib
import sys

def check_libraries():
    required_libs = ["ultralytics", "streamlit", "cv2", "numpy", "PIL"]
    missing_libs = []

    print("🔍 [시스템 환경 점검 시작]")
    print(f"Python 버전: {sys.version.split()[0]}")
    print("-" * 30)

    for lib in required_libs:
        try:
            importlib.import_module(lib)
            print(f"✅ {lib}: 설치됨")
        except ImportError:
            print(f"❌ {lib}: 설치되지 않음")
            missing_libs.append(lib)

    print("-" * 30)
    if not missing_libs:
        print("🚀 모든 준비가 끝났습니다! 코드를 실행하셔도 좋습니다.")
    else:
        print(f"⚠️ 다음 라이브러리를 설치해야 합니다: {', '.join(missing_libs)}")
        print("명령어: pip install " + " ".join(["opencv-python" if l=="cv2" else "pillow" if l=="PIL" else l for l in missing_libs]))

if __name__ == "__main__":
    check_libraries()