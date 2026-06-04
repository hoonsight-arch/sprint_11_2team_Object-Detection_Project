"""
전처리/후처리 병목 구간 프로파일러
사용법: app_wrapper.py와 main.py에서 import하여 각 스테이지에 삽입
"""
import time
import functools
from contextlib import contextmanager
from collections import defaultdict
from typing import Optional


class StageTimer:
    """단일 요청 내 스테이지별 소요 시간을 누적·기록하는 타이머"""

    def __init__(self):
        self._times: dict[str, float] = {}
        self._start: Optional[float] = None
        self._stage: Optional[str] = None

    @contextmanager
    def measure(self, stage: str):
        t0 = time.perf_counter()
        yield
        self._times[stage] = round((time.perf_counter() - t0) * 1000, 2)  # ms

    def result(self) -> dict:
        total = sum(self._times.values())
        return {
            "stages_ms": self._times,
            "total_ms": round(total, 2),
        }

    def print_report(self, label: str = "PROFILE"):
        r = self.result()
        print(f"\n[{label}] ── 스테이지별 소요 시간 ──────────────────")
        for stage, ms in r["stages_ms"].items():
            bar = "█" * int(ms / 10)
            print(f"  {stage:<25} {ms:>8.2f} ms  {bar}")
        print(f"  {'합계':<25} {r['total_ms']:>8.2f} ms")
        print("─" * 50)


def timing_middleware_factory():
    """FastAPI 전체 요청 레이턴시 미들웨어 (선택 사용)"""
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request

    class TimingMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            t0 = time.perf_counter()
            response = await call_next(request)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
            print(f"[MIDDLEWARE] {request.method} {request.url.path} → {elapsed_ms:.2f} ms")
            return response

    return TimingMiddleware


# 전역 통계 누적 (선택: 장기 모니터링용)
_stats: dict[str, list[float]] = defaultdict(list)

def record_stats(timings: dict):
    for stage, ms in timings.get("stages_ms", {}).items():
        _stats[stage].append(ms)

def print_stats_summary():
    print("\n[STATS] ── 누적 평균 소요 시간 (전체 요청) ─────────────")
    for stage, samples in _stats.items():
        avg = sum(samples) / len(samples)
        print(f"  {stage:<25} avg={avg:>8.2f} ms  n={len(samples)}")
    print("─" * 50)
