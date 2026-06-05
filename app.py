# -*- coding: utf-8 -*-
"""
배포용 Streamlit 대시보드 app.py
- API 키 입력창 없음: Streamlit Cloud Secrets에서만 로드
- 차량별 경로 지도 분리: 차량 1 / 차량 2 / ... 탭으로 표시
- 직선 연결이 아니라 OSRM 도로 경로 사용, 실패 시 해당 구간만 직선 fallback
- 경로 방향 화살표: 각 구간에 2개씩 표시
- 결과가 사라지지 않도록 st.session_state 사용
- 계산 기준/과정 설명 섹션 포함
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import PolyLineTextPath


# ============================================================
# 0. 기본 설정
# ============================================================

st.set_page_config(
    page_title="여의도 따릉이 수거·재배치 대시보드",
    page_icon="🚲",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# 배포용 CSV 파일명
FILE_BASE_DEMAND = DATA_DIR / "기본예상수요.csv"
FILE_WEATHER_THRESHOLD = DATA_DIR / "날씨구간화기준.csv"
FILE_WEATHER_COEF = DATA_DIR / "통합기상조건보정계수.csv"
FILE_STATION_PRIORITY = DATA_DIR / "대여소우선순위.csv"
FILE_YEOUIDO_FILTER = DATA_DIR / "여의도_대여소_필터.csv"
FILE_AREA_LIST = DATA_DIR / "서울시_주요_121장소_목록.csv"

BIKE_API_BASE = "http://openapi.seoul.go.kr:8088/{key}/json/bikeList/{start}/{end}/"
CITYDATA_API_BASE = "http://openapi.seoul.go.kr:8088/{key}/json/citydata/1/5/{area_nm}"
OSRM_ROUTE_URL = "https://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}"

DEFAULT_CENTER = [37.5269, 126.9245]  # 여의도 인근
KST = ZoneInfo("Asia/Seoul")
# 발표/운영용 고정 출발지: 여의도 복지관 인근 진입 지점
DEFAULT_DEPOT_LAT = 37.518133
DEFAULT_DEPOT_LON = 126.930776
DEFAULT_DEPOT_NAME = "여의도 복지관"


# ============================================================
# 1. CSS: 글씨 잘림 방지 / 카드 크기 축소
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }
    h1 { font-size: 2.0rem !important; }
    h2 { font-size: 1.45rem !important; margin-top: 1.2rem !important; }
    h3 { font-size: 1.15rem !important; }

    .small-note {
        font-size: 0.88rem;
        color: #5f6673;
        line-height: 1.45;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 0.65rem;
        margin: 0.7rem 0 1.0rem 0;
    }
    .metric-card {
        border: 1px solid #e7eaf0;
        border-radius: 14px;
        padding: 0.75rem 0.85rem;
        background: #ffffff;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        min-height: 78px;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #667085;
        margin-bottom: 0.35rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .metric-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #252b37;
        line-height: 1.2;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .section-box {
        border: 1px solid #e7eaf0;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        background: #fbfcff;
        margin: 0.75rem 0 1rem 0;
    }
    .route-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin: 0.5rem 0 0.25rem 0;
    }
    .warn-box {
        border: 1px solid #ffd6a7;
        background: #fff8ef;
        color: #7a4b00;
        padding: 0.75rem 1rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 2. 공통 유틸
# ============================================================

def norm_station_id(x: Any) -> str:
    """stationId가 ST-123 / 123 / 123.0 등으로 섞여도 매칭되도록 정규화."""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    m = re.search(r"(\d+)", s)
    return m.group(1) if m else s


def to_float(x: Any, default: float = 0.0) -> float:
    if x is None or pd.isna(x):
        return default
    s = str(x).strip()
    if s in ["", "-", "None", "nan"]:
        return default
    s = s.replace("%", "").replace("mm", "").replace("㎜", "").replace("m/s", "")
    try:
        return float(s)
    except Exception:
        nums = re.findall(r"-?\d+\.?\d*", s)
        return float(nums[0]) if nums else default


def find_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    cols = {str(c).strip(): c for c in df.columns}
    for c in candidates:
        if c in cols:
            return cols[c]
    # contains match fallback
    for cand in candidates:
        for c in df.columns:
            if cand.replace(" ", "") in str(c).replace(" ", ""):
                return c
    if required:
        raise KeyError(f"필수 컬럼을 찾을 수 없습니다: {candidates} / 현재 컬럼: {list(df.columns)}")
    return None


def current_time_context(now: Optional[datetime] = None) -> Dict[str, Any]:
    now = now or datetime.now(KST)
    hour = now.hour

    # 기존 분석 코드의 시간대그룹 명칭과 최대한 맞춤
    if 0 <= hour < 6:
        tg = "심야"
    elif 6 <= hour < 10:
        tg = "출근시간"
    elif 10 <= hour < 17:
        tg = "낮시간"
    elif 17 <= hour < 21:
        tg = "퇴근시간"
    else:
        tg = "야간"

    weektype = "주말" if now.weekday() >= 5 else "평일"
    month = now.month
    if month in [3, 4, 5]:
        season = "봄"
    elif month in [6, 7, 8]:
        season = "여름"
    elif month in [9, 10, 11]:
        season = "가을"
    else:
        season = "겨울"

    return {
        "now": now,
        "시간대": hour,
        "시간대그룹": tg,
        "평일주말": weektype,
        "월": month,
        "계절": season,
    }


def metric_grid(items: List[Tuple[str, str]]) -> None:
    html = ['<div class="metric-grid">']
    for label, value in items:
        html.append(
            f'<div class="metric-card"><div class="metric-label" title="{label}">{label}</div>'
            f'<div class="metric-value" title="{value}">{value}</div></div>'
        )
    html.append('</div>')
    st.markdown("".join(html), unsafe_allow_html=True)


# ============================================================
# 3. 데이터 로드
# ============================================================

@st.cache_data(show_spinner=False)
def read_csv_safely(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_static_data() -> Dict[str, pd.DataFrame]:
    return {
        "base": read_csv_safely(FILE_BASE_DEMAND),
        "threshold": read_csv_safely(FILE_WEATHER_THRESHOLD),
        "coef": read_csv_safely(FILE_WEATHER_COEF),
        "priority": read_csv_safely(FILE_STATION_PRIORITY),
        "filter": read_csv_safely(FILE_YEOUIDO_FILTER),
        "areas": read_csv_safely(FILE_AREA_LIST),
    }


# ============================================================
# 4. API 호출
# ============================================================

@st.cache_data(ttl=60, show_spinner=False)
def fetch_bike_api(api_key: str, max_rows: int = 3000) -> pd.DataFrame:
    if not api_key:
        return pd.DataFrame()

    rows = []
    step = 1000
    for start in range(1, max_rows + 1, step):
        end = min(start + step - 1, max_rows)
        url = BIKE_API_BASE.format(key=api_key, start=start, end=end)
        try:
            r = requests.get(url, timeout=12)
            r.raise_for_status()
            data = r.json()
            part = data.get("rentBikeStatus", {}).get("row", [])
            if not part:
                break
            rows.extend(part)
            total = int(data.get("rentBikeStatus", {}).get("list_total_count", len(rows)))
            if end >= total:
                break
        except Exception as e:
            st.warning(f"따릉이 API 호출 실패: {e}")
            break

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    rename_map = {
        "stationId": "대여소_ID_API",
        "stationName": "대여소명_API",
        "parkingBikeTotCnt": "현재자전거수",
        "rackTotCnt": "거치대수",
        "shared": "거치율",
        "stationLatitude": "위도",
        "stationLongitude": "경도",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    for c in ["현재자전거수", "거치대수", "거치율", "위도", "경도"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "대여소_ID_API" in df.columns:
        df["station_key"] = df["대여소_ID_API"].apply(norm_station_id)
    return df


@st.cache_data(ttl=300, show_spinner=False)
def fetch_citydata_api(api_key: str, area_nm: str) -> Dict[str, Any]:
    if not api_key or not area_nm:
        return {}
    url = CITYDATA_API_BASE.format(key=api_key, area_nm=requests.utils.quote(area_nm))
    try:
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        data = r.json()
        root = data.get("CITYDATA") or data.get("SeoulRtd.citydata") or data
        if isinstance(root, dict):
            return root
        return {}
    except Exception as e:
        st.warning(f"도시데이터 API 호출 실패: {e}")
        return {}


def parse_weather(citydata: Dict[str, Any]) -> Dict[str, Any]:
    weather = citydata.get("WEATHER_STTS", {}) if isinstance(citydata, dict) else {}
    if isinstance(weather, list) and weather:
        weather = weather[0]
    if not isinstance(weather, dict):
        weather = {}

    return {
        "TEMP": to_float(weather.get("TEMP"), np.nan),
        "HUMIDITY": to_float(weather.get("HUMIDITY"), np.nan),
        "WIND_SPD": to_float(weather.get("WIND_SPD"), np.nan),
        "PRECIPITATION": to_float(weather.get("PRECIPITATION"), 0.0),
        "PRECPT_TYPE": str(weather.get("PRECPT_TYPE", "")),
        "WEATHER_TIME": str(weather.get("WEATHER_TIME", "")),
    }


# ============================================================
# 5. 실시간 조건 구간화
# ============================================================

def parse_threshold_value(text: str, key: str) -> Optional[float]:
    """날씨구간화기준.csv의 문장형 기준에서 숫자 추출."""
    if not isinstance(text, str):
        return None
    nums = re.findall(r"-?\d+\.\d+|-?\d+", text)
    if not nums:
        return None
    # 기준 문자열은 대개 q1, q2 순서. 필요한 key에 따라 처리
    vals = [float(x) for x in nums]
    if key == "q1":
        return vals[0]
    if key == "q2":
        return vals[1] if len(vals) > 1 else vals[0]
    if key == "median":
        # 강수량은 보통 0과 중앙값이 같이 있으므로 마지막 숫자를 중앙값으로 사용
        return vals[-1]
    return vals[0]


def get_weather_thresholds(threshold_df: pd.DataFrame) -> Dict[str, float]:
    out = {
        "temp_q1": 10.0,
        "temp_q2": 25.0,
        "humid_q1": 40.0,
        "humid_q2": 70.0,
        "wind_q1": 2.0,
        "wind_q2": 5.0,
        "rain_median": 5.0,
    }
    if threshold_df.empty:
        return out

    var_col = find_col(threshold_df, ["변수"], required=False)
    crit_col = find_col(threshold_df, ["구간화 기준", "기준"], required=False)
    if var_col is None or crit_col is None:
        return out

    for _, row in threshold_df.iterrows():
        var = str(row.get(var_col, ""))
        crit = str(row.get(crit_col, ""))
        if "기온" in var:
            out["temp_q1"] = parse_threshold_value(crit, "q1") or out["temp_q1"]
            out["temp_q2"] = parse_threshold_value(crit, "q2") or out["temp_q2"]
        elif "습도" in var:
            out["humid_q1"] = parse_threshold_value(crit, "q1") or out["humid_q1"]
            out["humid_q2"] = parse_threshold_value(crit, "q2") or out["humid_q2"]
        elif "풍속" in var:
            out["wind_q1"] = parse_threshold_value(crit, "q1") or out["wind_q1"]
            out["wind_q2"] = parse_threshold_value(crit, "q2") or out["wind_q2"]
        elif "강수" in var:
            out["rain_median"] = parse_threshold_value(crit, "median") or out["rain_median"]
    return out


def classify_weather(weather: Dict[str, Any], thresholds: Dict[str, float]) -> Dict[str, str]:
    temp = weather.get("TEMP", np.nan)
    humid = weather.get("HUMIDITY", np.nan)
    wind = weather.get("WIND_SPD", np.nan)
    rain = weather.get("PRECIPITATION", 0.0)

    def tri(x, q1, q2, labels):
        if pd.isna(x):
            return labels[1]
        if x <= q1:
            return labels[0]
        elif x <= q2:
            return labels[1]
        return labels[2]

    temp_cond = tri(temp, thresholds["temp_q1"], thresholds["temp_q2"], ["저온", "적정", "고온"])
    humid_cond = tri(humid, thresholds["humid_q1"], thresholds["humid_q2"], ["낮음", "보통", "높음"])
    wind_cond = tri(wind, thresholds["wind_q1"], thresholds["wind_q2"], ["약풍", "보통", "강풍"])

    if rain <= 0:
        rain_cond = "비없음"
    elif rain <= thresholds["rain_median"]:
        rain_cond = "약한비"
    else:
        rain_cond = "강한비"

    snow_cond = "눈없음"  # 도시데이터 API에서 적설값이 없으므로 배포용은 눈없음 처리
    full = f"{temp_cond}_{rain_cond}_{humid_cond}_{wind_cond}_{snow_cond}"

    return {
        "기온조건": temp_cond,
        "강수조건": rain_cond,
        "습도조건": humid_cond,
        "풍속조건": wind_cond,
        "적설조건": snow_cond,
        "기상조건": full,
    }


# ============================================================
# 6. 후보 계산
# ============================================================

def filter_yeouido_stations(bike_df: pd.DataFrame, filter_df: pd.DataFrame) -> pd.DataFrame:
    if bike_df.empty:
        return bike_df

    df = bike_df.copy()
    if not filter_df.empty:
        sid_col = find_col(filter_df, ["대여소_ID", "대여소ID", "stationId", "SBIKE_SPOT_ID"], required=False)
        if sid_col:
            keys = set(filter_df[sid_col].apply(norm_station_id).astype(str))
            if keys:
                out = df[df["station_key"].astype(str).isin(keys)].copy()
                if not out.empty:
                    return out

    # fallback: 이름 키워드 필터
    name_col = "대여소명_API" if "대여소명_API" in df.columns else None
    if name_col:
        keywords = ["여의", "국회", "샛강", "IFC", "더현대", "한강", "파크원", "KBS"]
        pat = "|".join(keywords)
        out = df[df[name_col].astype(str).str.contains(pat, case=False, na=False)].copy()
        return out if not out.empty else df
    return df


def prepare_base_for_current(base_df: pd.DataFrame, ctx: Dict[str, Any]) -> pd.DataFrame:
    if base_df.empty:
        return pd.DataFrame()
    df = base_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    sid_col = find_col(df, ["대여소_ID", "대여소ID"], required=False)
    if sid_col:
        df["station_key"] = df[sid_col].apply(norm_station_id)

    # 현재 조건과 일치하는 행 우선. 없으면 조건을 점차 완화.
    conds = []
    for colname, val in [("시간대그룹", ctx["시간대그룹"]), ("평일주말", ctx["평일주말"]), ("월", ctx["월"]), ("계절", ctx["계절"]), ("시간대", ctx["시간대"] )]:
        if colname in df.columns:
            conds.append((colname, val))

    cur = df.copy()
    for col, val in conds:
        tmp = cur[cur[col].astype(str) == str(val)].copy()
        if not tmp.empty:
            cur = tmp

    # 여전히 같은 대여소 여러 행이면 평균
    demand_out_col = find_col(cur, ["기본예상대여수요"], required=False)
    demand_in_col = find_col(cur, ["기본예상반납수요"], required=False)
    name_col = find_col(cur, ["대여소명", "대여소 명"], required=False)
    if "station_key" not in cur.columns or not demand_out_col or not demand_in_col:
        return pd.DataFrame()

    agg = cur.groupby("station_key", as_index=False).agg(
        기본예상대여수요=(demand_out_col, "mean"),
        기본예상반납수요=(demand_in_col, "mean"),
    )
    if name_col:
        names = cur.groupby("station_key", as_index=False)[name_col].first().rename(columns={name_col: "대여소명_과거"})
        agg = agg.merge(names, on="station_key", how="left")
    return agg


def get_weather_coef(coef_df: pd.DataFrame, time_group: str, weather_condition: str) -> Tuple[float, float]:
    if coef_df.empty:
        return 1.0, 1.0
    df = coef_df.copy()
    tg_col = find_col(df, ["시간대그룹"], required=False)
    wc_col = find_col(df, ["기상조건"], required=False)
    out_col = find_col(df, ["대여수요_날씨보정계수", "대여수요보정계수"], required=False)
    in_col = find_col(df, ["반납수요_날씨보정계수", "반납수요보정계수"], required=False)
    if not all([tg_col, wc_col, out_col, in_col]):
        return 1.0, 1.0

    exact = df[(df[tg_col].astype(str) == str(time_group)) & (df[wc_col].astype(str) == str(weather_condition))]
    if exact.empty:
        # 같은 시간대만 평균 fallback
        exact = df[df[tg_col].astype(str) == str(time_group)]
    if exact.empty:
        return 1.0, 1.0
    return float(pd.to_numeric(exact[out_col], errors="coerce").mean()), float(pd.to_numeric(exact[in_col], errors="coerce").mean())


def build_candidates(
    bike_df: pd.DataFrame,
    base_current: pd.DataFrame,
    priority_df: pd.DataFrame,
    coef_out: float,
    coef_in: float,
    L: float,
    U: float,
    top_pickup: int,
    top_delivery: int,
) -> pd.DataFrame:
    if bike_df.empty:
        return pd.DataFrame()

    df = bike_df.copy()
    if not base_current.empty:
        df = df.merge(base_current, on="station_key", how="left")
    else:
        df["기본예상대여수요"] = 0
        df["기본예상반납수요"] = 0

    # priority 붙이기
    if not priority_df.empty:
        pr = priority_df.copy()
        sid_col = find_col(pr, ["대여소_ID", "대여소ID"], required=False)
        if sid_col:
            pr["station_key"] = pr[sid_col].apply(norm_station_id)
            score_col = find_col(pr, ["우선순위기초점수", "우선순위점수"], required=False)
            freq_col = find_col(pr, ["평소이용빈도"], required=False)
            cols = ["station_key"] + [c for c in [score_col, freq_col] if c]
            pr = pr[cols].drop_duplicates("station_key")
            df = df.merge(pr, on="station_key", how="left")

    if "우선순위기초점수" not in df.columns:
        df["우선순위기초점수"] = 0.5
    df["우선순위기초점수"] = pd.to_numeric(df["우선순위기초점수"], errors="coerce").fillna(0.5)

    for c in ["기본예상대여수요", "기본예상반납수요", "현재자전거수", "거치대수"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["예측대여수요"] = df["기본예상대여수요"] * coef_out
    df["예측반납수요"] = df["기본예상반납수요"] * coef_in
    df["예상재고"] = df["현재자전거수"] + df["예측반납수요"] - df["예측대여수요"]

    df["과잉기준"] = U * df["거치대수"]
    df["부족기준"] = L * df["거치대수"]
    df["수거필요량"] = np.maximum(0, np.ceil(df["예상재고"] - df["과잉기준"])).astype(int)
    df["재배치필요량"] = np.maximum(0, np.ceil(df["부족기준"] - df["예상재고"])).astype(int)
    df["처리전불균형"] = df["수거필요량"] + df["재배치필요량"]

    df["후보점수"] = (
        df["처리전불균형"] * 0.55
        + df["우선순위기초점수"] * 10 * 0.30
        + (df["예측대여수요"] - df["예측반납수요"]).abs() * 0.15
    )

    pickups = df[df["수거필요량"] > 0].sort_values("후보점수", ascending=False).head(top_pickup).copy()
    pickups["후보유형"] = "수거"
    pickups["필요량"] = pickups["수거필요량"]

    deliveries = df[df["재배치필요량"] > 0].sort_values("후보점수", ascending=False).head(top_delivery).copy()
    deliveries["후보유형"] = "재배치"
    deliveries["필요량"] = deliveries["재배치필요량"]

    cand = pd.concat([pickups, deliveries], ignore_index=True)
    if cand.empty:
        return cand

    # 지도용 정리
    cand["대여소명"] = cand.get("대여소명_API", cand.get("대여소명_과거", ""))
    keep_cols = [
        "station_key", "대여소_ID_API", "대여소명", "후보유형", "필요량", "현재자전거수", "거치대수", "거치율",
        "예측대여수요", "예측반납수요", "예상재고", "수거필요량", "재배치필요량", "처리전불균형",
        "우선순위기초점수", "후보점수", "위도", "경도"
    ]
    keep_cols = [c for c in keep_cols if c in cand.columns]
    return cand[keep_cols].copy()


# ============================================================
# 7. 휴리스틱 경로 추천
# ============================================================

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1) * math.cos(p2) * math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def nearest_index(current: Tuple[float, float], rows: pd.DataFrame) -> Optional[int]:
    if rows.empty:
        return None
    d = rows.apply(lambda r: haversine_m(current[0], current[1], r["위도"], r["경도"]), axis=1)
    return int(d.idxmin())


def balanced_greedy_routes(candidates: pd.DataFrame, depot_lat: float, depot_lon: float, vehicle_count: int, capacity: int) -> Tuple[List[List[Dict[str, Any]]], pd.DataFrame, Dict[str, float]]:
    """
    차량 1대가 모든 후보를 독식하지 않도록 후보를 차량별로 먼저 분배한 뒤,
    각 차량 안에서 수거→재배치 순서로 경로를 구성한다.

    - 수거 후보와 재배치 후보를 후보점수 순으로 정렬
    - 차량 번호별로 round-robin 분배
    - 각 차량은 자기에게 배정된 후보 안에서 가까운 지점 우선 방문
    - 배포용 휴리스틱이므로 최적해가 아니라 발표용 경로 추천임
    """
    cand = candidates.copy().reset_index(drop=True)
    cand["남은수거"] = cand.get("수거필요량", 0).astype(float)
    cand["남은재배치"] = cand.get("재배치필요량", 0).astype(float)
    cand["처리수거량"] = 0.0
    cand["처리재배치량"] = 0.0
    cand["배정차량"] = np.nan

    if cand.empty or vehicle_count <= 0:
        return [], cand, {"before": 0, "processed": 0, "after": 0, "improve": 0}

    # 후보 유형별로 정렬 후 차량에 균등 배정
    pickup_idx = cand[cand["남은수거"] > 0].sort_values("후보점수", ascending=False).index.tolist()
    delivery_idx = cand[cand["남은재배치"] > 0].sort_values("후보점수", ascending=False).index.tolist()

    for n, idx in enumerate(pickup_idx):
        cand.loc[idx, "배정차량"] = n % vehicle_count
    for n, idx in enumerate(delivery_idx):
        # 이미 수거 후보이기도 한 경우 기존 배정 유지
        if pd.isna(cand.loc[idx, "배정차량"]):
            cand.loc[idx, "배정차량"] = n % vehicle_count

    # 혹시 배정이 안 된 후보가 있으면 점수순으로 분배
    unassigned = cand[cand["배정차량"].isna()].sort_values("후보점수", ascending=False).index.tolist()
    for n, idx in enumerate(unassigned):
        cand.loc[idx, "배정차량"] = n % vehicle_count

    routes: List[List[Dict[str, Any]]] = []

    for k in range(vehicle_count):
        load = 0.0
        cur = (depot_lat, depot_lon)
        route = [{"type": "depot", "name": DEFAULT_DEPOT_NAME, "lat": depot_lat, "lon": depot_lon, "action": "출발", "amount": 0, "load_after": load}]

        assigned = cand[cand["배정차량"] == k].copy()
        if assigned.empty:
            # 이 차량에 배정된 후보가 없으면 출발지 표시만 유지
            route.append({"type": "depot", "name": DEFAULT_DEPOT_NAME, "lat": depot_lat, "lon": depot_lon, "action": "대기", "amount": 0, "load_after": int(round(load))})
            routes.append(route)
            continue

        safety = 0
        while safety < 80:
            safety += 1
            assigned_idx = cand[cand["배정차량"] == k].index
            pickups = cand.loc[assigned_idx][(cand.loc[assigned_idx, "남은수거"] > 0) & (load < capacity)].copy()
            deliveries = cand.loc[assigned_idx][(cand.loc[assigned_idx, "남은재배치"] > 0) & (load > 0)].copy()

            if load <= 0 and not pickups.empty:
                idx = nearest_index(cur, pickups)
                amount = min(capacity - load, cand.loc[idx, "남은수거"])
                cand.loc[idx, "남은수거"] -= amount
                cand.loc[idx, "처리수거량"] += amount
                load += amount
                row = cand.loc[idx]
                action = "수거"
            elif load > 0 and not deliveries.empty:
                idx = nearest_index(cur, deliveries)
                amount = min(load, cand.loc[idx, "남은재배치"])
                cand.loc[idx, "남은재배치"] -= amount
                cand.loc[idx, "처리재배치량"] += amount
                load -= amount
                row = cand.loc[idx]
                action = "재배치"
            elif not pickups.empty and load < capacity:
                idx = nearest_index(cur, pickups)
                amount = min(capacity - load, cand.loc[idx, "남은수거"])
                cand.loc[idx, "남은수거"] -= amount
                cand.loc[idx, "처리수거량"] += amount
                load += amount
                row = cand.loc[idx]
                action = "수거"
            else:
                break

            if amount <= 0:
                break

            cur = (float(row["위도"]), float(row["경도"]))
            route.append({
                "type": action,
                "station_key": row.get("station_key", ""),
                "name": row.get("대여소명", ""),
                "lat": float(row["위도"]),
                "lon": float(row["경도"]),
                "action": action,
                "amount": int(round(amount)),
                "load_after": int(round(load)),
            })

        route.append({"type": "depot", "name": DEFAULT_DEPOT_NAME, "lat": depot_lat, "lon": depot_lon, "action": "복귀", "amount": 0, "load_after": int(round(load))})
        routes.append(route)

    cand["최적화후재고"] = cand["예상재고"] - cand["처리수거량"] + cand["처리재배치량"]
    cand["남은불균형"] = cand["남은수거"] + cand["남은재배치"]

    before = float(candidates["처리전불균형"].sum()) if not candidates.empty else 0.0
    processed = float(cand["처리수거량"].sum() + cand["처리재배치량"].sum())
    after = float(cand["남은불균형"].sum())
    improve = ((before - after) / before * 100) if before > 0 else 0.0
    summary = {"before": before, "processed": processed, "after": after, "improve": improve}

    return routes, cand, summary

# ============================================================
# 8. OSRM 실제 도로 경로 + 화살표 지도
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def osrm_route(lat1: float, lon1: float, lat2: float, lon2: float) -> Dict[str, Any]:
    params = {"overview": "full", "geometries": "geojson", "steps": "false"}
    url = OSRM_ROUTE_URL.format(lon1=lon1, lat1=lat1, lon2=lon2, lat2=lat2)
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        routes = data.get("routes", [])
        if not routes:
            raise ValueError("OSRM route empty")
        route = routes[0]
        coords = route["geometry"]["coordinates"]
        latlon = [(float(lat), float(lon)) for lon, lat in coords]
        return {
            "ok": True,
            "coords": latlon,
            "distance_m": float(route.get("distance", 0)),
            "duration_s": float(route.get("duration", 0)),
        }
    except Exception:
        return {
            "ok": False,
            "coords": [(lat1, lon1), (lat2, lon2)],
            "distance_m": haversine_m(lat1, lon1, lat2, lon2),
            "duration_s": 0,
        }


def bearing_deg(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, p1)
    lat2, lon2 = map(math.radians, p2)
    dlon = lon2 - lon1
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dlon)
    brng = math.degrees(math.atan2(y, x))
    return (brng + 360) % 360


def interpolate_point(coords: List[Tuple[float, float]], frac: float) -> Tuple[float, float, float]:
    """경로 전체 길이 기준 frac 위치의 lat, lon, bearing 반환."""
    if len(coords) < 2:
        lat, lon = coords[0]
        return lat, lon, 0
    seg_lengths = [haversine_m(coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1]) for i in range(len(coords)-1)]
    total = sum(seg_lengths)
    if total <= 0:
        lat, lon = coords[0]
        return lat, lon, 0
    target = total * frac
    acc = 0.0
    for i, seg in enumerate(seg_lengths):
        if acc + seg >= target:
            ratio = (target - acc) / seg if seg > 0 else 0
            lat = coords[i][0] + (coords[i+1][0] - coords[i][0]) * ratio
            lon = coords[i][1] + (coords[i+1][1] - coords[i][1]) * ratio
            brg = bearing_deg(coords[i], coords[i+1])
            return lat, lon, brg
        acc += seg
    lat, lon = coords[-1]
    return lat, lon, bearing_deg(coords[-2], coords[-1])


def add_direction_arrows_on_polyline(polyline: folium.PolyLine) -> None:
    """PolylineTextPath로 경로 선 위에 방향 화살표를 표시한다.
    지도 아이콘을 회전시키는 방식보다 도로 흐름을 훨씬 명확하게 보여준다.
    """
    try:
        PolyLineTextPath(
            polyline,
            "   ➜   ",
            repeat=True,
            offset=7,
            attributes={
                "fill": "#1f7a3a",
                "font-weight": "bold",
                "font-size": "18px",
                "opacity": "0.85",
            },
        ).add_to(polyline._parent)
    except Exception:
        # 일부 배포 환경에서 플러그인이 실패해도 지도는 유지
        pass

def add_station_marker(m: folium.Map, node: Dict[str, Any], seq: int) -> None:
    if node["type"] == "depot":
        folium.Marker(
            [node["lat"], node["lon"]],
            tooltip=node["name"],
            popup=f"<b>{node['name']}</b>",
            icon=folium.Icon(color="black", icon="home", prefix="fa"),
        ).add_to(m)
        return

    color = "red" if node["action"] == "수거" else "blue"
    popup = (
        f"<b>{seq}. {node['name']}</b><br>"
        f"작업: {node['action']} {node['amount']}대<br>"
        f"작업 후 적재량: {node['load_after']}대"
    )
    folium.Marker(
        [node["lat"], node["lon"]],
        tooltip=f"{seq}. {node['action']} {node['amount']}대 - {node['name']}",
        popup=popup,
        icon=folium.Icon(color=color, icon="info-sign"),
    ).add_to(m)

    # 순서 번호 라벨
    folium.Marker(
        [node["lat"], node["lon"]],
        icon=folium.DivIcon(
            html=f"""
            <div style="background:white; border:2px solid {'#fa5252' if color=='red' else '#4c6ef5'}; border-radius:14px; width:28px; height:28px; text-align:center; font-weight:bold; font-size:14px; line-height:24px; color:#333;">
            {seq}
            </div>
            """,
            icon_size=(28, 28), icon_anchor=(14, -8),
        ),
    ).add_to(m)


def make_overview_map(candidates: pd.DataFrame, depot_lat: float, depot_lon: float) -> folium.Map:
    m = folium.Map(location=[depot_lat, depot_lon], zoom_start=14, tiles="CartoDB positron")
    folium.Marker([depot_lat, depot_lon], tooltip="출발지", icon=folium.Icon(color="black", icon="home", prefix="fa")).add_to(m)
    for _, r in candidates.iterrows():
        color = "red" if r["후보유형"] == "수거" else "blue"
        folium.CircleMarker(
            location=[r["위도"], r["경도"]],
            radius=7,
            color=color,
            fill=True,
            fill_opacity=0.75,
            tooltip=f"{r['후보유형']} {int(r['필요량'])}대 | {r['대여소명']}",
            popup=f"<b>{r['대여소명']}</b><br>{r['후보유형']} 필요량: {int(r['필요량'])}대",
        ).add_to(m)
    return m


def make_vehicle_route_map(route: List[Dict[str, Any]], vehicle_no: int) -> Tuple[folium.Map, pd.DataFrame, Dict[str, float]]:
    # 차량별 개별 지도: 이 지도에는 해당 차량 경로만 들어감
    center_lat = np.mean([n["lat"] for n in route])
    center_lon = np.mean([n["lon"] for n in route])
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="CartoDB positron")

    total_dist = 0.0
    osrm_fail = 0
    table_rows = []

    for i, node in enumerate(route):
        add_station_marker(m, node, i)
        table_rows.append({
            "순서": i,
            "장소": node["name"],
            "작업": node["action"],
            "수량": node.get("amount", 0),
            "작업 후 적재량": node.get("load_after", 0),
        })

    for i in range(len(route) - 1):
        a, b = route[i], route[i+1]
        res = osrm_route(a["lat"], a["lon"], b["lat"], b["lon"])
        coords = res["coords"]
        total_dist += res["distance_m"]
        if not res["ok"]:
            osrm_fail += 1

        line = folium.PolyLine(
            coords,
            color="#2f9e44",
            weight=5,
            opacity=0.78,
            tooltip=f"차량 {vehicle_no}: {a['name']} → {b['name']}",
        ).add_to(m)
        add_direction_arrows_on_polyline(line)

    return m, pd.DataFrame(table_rows), {"distance_m": total_dist, "osrm_fail": osrm_fail}


# ============================================================
# 9. 화면 구성
# ============================================================

st.title("🚲 여의도 따릉이 수거·재배치 경로 추천 대시보드")
st.caption("배포용 버전: 실시간 API + 과거 수요모델 + 휴리스틱 경로 추천. 출발지는 여의도 복지관으로 고정하고, 차량별 경로 지도는 각각 분리해서 표시합니다.")

# session state 초기화
for key in ["candidates", "routes", "processed", "summary", "weather", "weather_cond", "ctx"]:
    if key not in st.session_state:
        st.session_state[key] = None

static = load_static_data()

# Secrets에서만 API 키 로드. 화면 입력창 없음.
try:
    bike_key = st.secrets["SEOUL_BIKE_API_KEY"]
except Exception:
    bike_key = ""
try:
    city_key = st.secrets["SEOUL_CITYDATA_API_KEY"]
except Exception:
    city_key = ""

if not bike_key or not city_key:
    st.markdown(
        "<div class='warn-box'>Streamlit Secrets에 SEOUL_BIKE_API_KEY와 SEOUL_CITYDATA_API_KEY를 넣어야 실시간 API가 작동합니다.</div>",
        unsafe_allow_html=True,
    )

# 사이드바 설정
st.sidebar.header("⚙️ 실행 설정")

# 장소명 선택
areas_df = static["areas"]
area_options = ["여의도"]
if not areas_df.empty:
    possible_cols = ["AREA_NM", "장소명", "핫스팟 장소명", "AREA"]
    area_col = find_col(areas_df, possible_cols, required=False)
    if area_col:
        vals = areas_df[area_col].dropna().astype(str).unique().tolist()
        area_options = vals if vals else area_options

# 여의도 관련 장소를 기본값으로
default_idx = 0
for i, a in enumerate(area_options):
    if "여의" in a:
        default_idx = i
        break
area_nm = st.sidebar.selectbox("도시데이터 장소명", area_options, index=default_idx)

vehicle_count = st.sidebar.number_input("차량 수", min_value=1, max_value=5, value=2, step=1)
capacity = st.sidebar.number_input("차량 용량", min_value=1, max_value=30, value=15, step=1)
L = st.sidebar.slider("부족 기준 거치율 L", min_value=0.0, max_value=1.0, value=0.30, step=0.05)
U = st.sidebar.slider("과잉 기준 거치율 U", min_value=0.0, max_value=1.0, value=0.80, step=0.05)
top_pickup = st.sidebar.slider("수거 후보 수", min_value=1, max_value=20, value=6, step=1)
top_delivery = st.sidebar.slider("재배치 후보 수", min_value=1, max_value=20, value=6, step=1)

st.sidebar.subheader("출발지")
depot_lat = DEFAULT_DEPOT_LAT
depot_lon = DEFAULT_DEPOT_LON
st.sidebar.info(f"{DEFAULT_DEPOT_NAME}\n위도 {depot_lat:.6f}, 경도 {depot_lon:.6f}")

col_run, col_clear = st.sidebar.columns(2)
run_clicked = col_run.button("경로 추천 실행", type="primary", use_container_width=True)
clear_clicked = col_clear.button("초기화", use_container_width=True)

if clear_clicked:
    for key in ["candidates", "routes", "processed", "summary", "weather", "weather_cond", "ctx"]:
        st.session_state[key] = None
    st.rerun()

if run_clicked:
    with st.spinner("실시간 API 호출 및 후보 계산 중..."):
        ctx = current_time_context()
        citydata = fetch_citydata_api(city_key, area_nm)
        weather = parse_weather(citydata)
        thresholds = get_weather_thresholds(static["threshold"])
        weather_cond = classify_weather(weather, thresholds)

        bike_all = fetch_bike_api(bike_key)
        bike_y = filter_yeouido_stations(bike_all, static["filter"])
        base_current = prepare_base_for_current(static["base"], ctx)
        coef_out, coef_in = get_weather_coef(static["coef"], ctx["시간대그룹"], weather_cond["기상조건"])

        candidates = build_candidates(
            bike_y,
            base_current,
            static["priority"],
            coef_out,
            coef_in,
            L,
            U,
            int(top_pickup),
            int(top_delivery),
        )

        if candidates.empty:
            st.error("수거·재배치 후보가 생성되지 않았습니다. 여의도 대여소 필터, API 키, 기준 L/U를 확인해주세요.")
        else:
            routes, processed, summary = balanced_greedy_routes(candidates, depot_lat, depot_lon, int(vehicle_count), int(capacity))
            st.session_state.candidates = candidates
            st.session_state.routes = routes
            st.session_state.processed = processed
            st.session_state.summary = summary
            st.session_state.weather = weather
            st.session_state.weather_cond = weather_cond
            st.session_state.ctx = ctx
            st.success("경로 추천이 완료되었습니다.")


# ============================================================
# 10. 결과 출력: session_state 기반으로 유지
# ============================================================

if st.session_state.summary is None:
    st.info("왼쪽 설정을 확인한 뒤 **경로 추천 실행**을 누르면 결과가 표시됩니다.")

    with st.expander("대시보드 계산 방식 보기", expanded=True):
        st.markdown(
            """
            <div class="section-box">
            <b>계산 흐름</b><br>
            1) 따릉이 API에서 현재 자전거 수와 거치대 수를 가져옵니다.<br>
            2) 도시데이터 API에서 현재 기온·습도·풍속·강수량을 가져옵니다.<br>
            3) 과거 데이터에서 만든 기본예상수요와 날씨 보정계수를 현재 조건에 맞게 선택합니다.<br>
            4) 예상재고 = 현재자전거수 + 예측반납수요 - 예측대여수요 를 계산합니다.<br>
            5) 예상재고가 U×거치대수보다 크면 수거 후보, L×거치대수보다 작으면 재배치 후보로 둡니다.<br>
            6) 배포용 대시보드는 휴리스틱 방식으로 차량별 수거·재배치 순서를 추천합니다.<br>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.stop()

summary = st.session_state.summary
weather = st.session_state.weather or {}
weather_cond = st.session_state.weather_cond or {}
ctx = st.session_state.ctx or current_time_context()
candidates = st.session_state.candidates
processed = st.session_state.processed
routes = st.session_state.routes

# ① 현재 조건
st.subheader("① 현재 조건")
metric_grid([
    ("현재 시점(KST)", ctx["now"].strftime("%m/%d %H:%M")),
    ("출발지", DEFAULT_DEPOT_NAME),
    ("시간대그룹", str(ctx["시간대그룹"])),
    ("평일/주말", str(ctx["평일주말"])),
    ("계절", str(ctx["계절"])),
    ("기상조건", f"{weather_cond.get('기온조건','-')} / {weather_cond.get('강수조건','-')} / {weather_cond.get('풍속조건','-')}"),
    ("기온", f"{weather.get('TEMP', np.nan):.1f}℃" if not pd.isna(weather.get("TEMP", np.nan)) else "-"),
    ("습도", f"{weather.get('HUMIDITY', np.nan):.1f}%" if not pd.isna(weather.get("HUMIDITY", np.nan)) else "-"),
    ("풍속", f"{weather.get('WIND_SPD', np.nan):.1f}m/s" if not pd.isna(weather.get("WIND_SPD", np.nan)) else "-"),
    ("강수량", f"{weather.get('PRECIPITATION', 0):.1f}mm"),
    ("날씨 업데이트", str(weather.get("WEATHER_TIME", "-"))),
])

# ② 개선 효과
st.subheader("② 수거·재배치 후보 및 개선 효과")
metric_grid([
    ("처리 전 후보 불균형", f"{summary['before']:.0f}대"),
    ("휴리스틱 처리량", f"{summary['processed']:.0f}대"),
    ("처리 후 남은 불균형", f"{summary['after']:.0f}대"),
    ("개선율", f"{summary['improve']:.1f}%"),
])

with st.expander("계산 기준과 해석 보기", expanded=False):
    st.markdown(
        f"""
        <div class="section-box">
        <b>1. 예측수요 계산</b><br>
        현재 조건은 <b>{ctx['시간대그룹']} / {ctx['평일주말']} / {ctx['계절']} / {weather_cond.get('기상조건','-')}</b>입니다.<br>
        기본예상대여수요와 기본예상반납수요는 과거 데이터에서 같은 시간·요일·월·계절 조건의 평균으로 계산된 값입니다.<br><br>
        <b>2. 예상재고 계산</b><br>
        예상재고 = 현재자전거수 + 예측반납수요 - 예측대여수요<br><br>
        <b>3. 수거·재배치 후보 계산</b><br>
        수거필요량 = max(0, 예상재고 - U×거치대수), 현재 U = <b>{U:.2f}</b><br>
        재배치필요량 = max(0, L×거치대수 - 예상재고), 현재 L = <b>{L:.2f}</b><br><br>
        <b>4. 경로 추천 방식</b><br>
        배포용 대시보드는 Gurobi가 아니라 휴리스틱 방식입니다. 차량은 수거 후보에서 자전거를 싣고, 가까운 재배치 후보에 배치하는 방식으로 경로를 구성합니다.<br>
        출발지는 여의도 복지관으로 고정했습니다. 시간 조건은 Streamlit 서버 시간이 아니라 한국시간(KST)을 기준으로 판정합니다.<br>차량 1대가 모든 후보를 처리하지 않도록 수거·재배치 후보를 후보점수 순으로 차량별 균등 배정한 뒤, 각 차량 안에서 가까운 지점 우선으로 경로를 구성합니다.<br>각 차량 지도는 OSRM 도로 경로를 호출해 실제 도로 흐름에 가깝게 표시하며, 경로 선 위의 화살표가 진행 방향을 나타냅니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

# 후보표
col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 수거 후보")
    cols = ["대여소명", "필요량", "현재자전거수", "거치대수", "예상재고", "후보점수"]
    st.dataframe(candidates[candidates["후보유형"] == "수거"][[c for c in cols if c in candidates.columns]], use_container_width=True, hide_index=True)
with col2:
    st.markdown("#### 재배치 후보")
    st.dataframe(candidates[candidates["후보유형"] == "재배치"][[c for c in cols if c in candidates.columns]], use_container_width=True, hide_index=True)

# 후보 개요 지도
st.subheader("③ 후보 위치 개요 지도")
st.caption("이 지도는 후보 위치만 보여줍니다. 차량 경로는 아래 차량별 지도에서 따로 확인합니다.")
overview_map = make_overview_map(candidates, depot_lat, depot_lon)
st_folium(overview_map, width=None, height=480, returned_objects=[])

# 차량별 경로 지도
st.subheader("④ 차량별 경로 지도")
st.caption("각 탭에는 해당 차량의 경로만 표시됩니다. 초록색 선은 OSRM 도로 경로이며, 화살표는 진행 방향입니다.")

tabs = st.tabs([f"차량 {i+1}" for i in range(len(routes))])
route_tables = []
for i, tab in enumerate(tabs):
    with tab:
        vmap, table, stats = make_vehicle_route_map(routes[i], i + 1)
        metric_grid([
            ("차량", f"{i+1}번"),
            ("방문 지점 수", f"{max(0, len(routes[i]) - 2)}개"),
            ("예상 이동거리", f"{stats['distance_m']/1000:.2f}km"),
            ("OSRM 실패 구간", f"{stats['osrm_fail']}개"),
        ])
        st_folium(vmap, width=None, height=560, returned_objects=[])
        st.markdown("#### 방문 순서")
        st.dataframe(table, use_container_width=True, hide_index=True)
        table["차량"] = i + 1
        route_tables.append(table)

# 처리 결과표
st.subheader("⑤ 처리 결과 상세")
show_cols = [
    "대여소명", "후보유형", "필요량", "현재자전거수", "거치대수", "예상재고", "처리수거량", "처리재배치량", "남은수거", "남은재배치", "남은불균형"
]
st.dataframe(processed[[c for c in show_cols if c in processed.columns]], use_container_width=True, hide_index=True)

# 다운로드는 선택 사항. 화면 표시는 이미 위에서 끝남.
with st.expander("결과 다운로드", expanded=False):
    if route_tables:
        all_route_table = pd.concat(route_tables, ignore_index=True)
        st.download_button(
            "차량별 방문 순서 CSV 다운로드",
            data=all_route_table.to_csv(index=False).encode("utf-8-sig"),
            file_name="vehicle_routes.csv",
            mime="text/csv",
        )
    st.download_button(
        "처리 결과 CSV 다운로드",
        data=processed.to_csv(index=False).encode("utf-8-sig"),
        file_name="relocation_result.csv",
        mime="text/csv",
    )
