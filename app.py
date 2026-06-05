# -*- coding: utf-8 -*-
"""
여의도 따릉이 수거·재배치 경로 추천 대시보드
배포용: Streamlit Secrets + 실시간 API + 공급수거/재배치 휴리스틱 + 차량별 도로 경로 지도

필수 data/ CSV:
- 기본예상수요.csv
- 날씨구간화기준.csv
- 통합기상조건보정계수.csv
- 대여소우선순위.csv
- 여의도_대여소_필터.csv
- 서울시_주요_121장소_목록.csv
"""

from __future__ import annotations

import math
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import streamlit as st
import folium
from folium.plugins import PolyLineTextPath
from streamlit_folium import st_folium
from zoneinfo import ZoneInfo

# ============================================================
# 0. 기본 설정
# ============================================================

st.set_page_config(
    page_title="여의도 따릉이 수거·재배치 경로 추천",
    page_icon="🚲",
    layout="wide",
)

DATA_DIR = Path("data")
DEPOT = {
    "name": "여의도 복지관",
    "lat": 37.518133,
    "lon": 126.930776,
}
KST = ZoneInfo("Asia/Seoul")

# 발표용 기본값
DEFAULT_VEHICLE_COUNT = 2
DEFAULT_CAPACITY = 15
DEFAULT_PICKUP_TOP_N = 8
DEFAULT_DELIVERY_TOP_N = 8
DEFAULT_LOW_RATIO = 0.30
DEFAULT_HIGH_RATIO = 0.80

# ------------------------------------------------------------
# 스타일: metric 글자 잘림 방지용 카드
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    .small-caption {font-size: 0.86rem; color: #6b7280; margin-bottom: 0.15rem;}
    .metric-card {
        border: 1px solid #e5e7eb; border-radius: 14px; padding: 14px 16px;
        background: #ffffff; min-height: 88px; box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        overflow-wrap: anywhere;
    }
    .metric-title {font-size: 0.82rem; color: #667085; margin-bottom: 0.45rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
    .metric-value {font-size: 1.35rem; font-weight: 700; color: #1f2937; line-height: 1.15; white-space: normal; overflow-wrap: anywhere;}
    .section-help {color:#6b7280; font-size:0.95rem; margin-top:-0.45rem; margin-bottom:0.8rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def metric_card(title: str, value: str):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# 1. 유틸 함수
# ============================================================

def normalize_station_id(x) -> str:
    """ST-123, 123.0, 123 등을 123 문자열로 정규화."""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    nums = re.findall(r"\d+", s)
    if not nums:
        return s
    return str(int(nums[-1]))


def to_float(x, default=0.0):
    if pd.isna(x):
        return default
    s = str(x).replace(",", "").replace("%", "").strip()
    if s in ["", "-", "None", "nan"]:
        return default
    try:
        return float(s)
    except Exception:
        nums = re.findall(r"-?\d+(?:\.\d+)?", s)
        return float(nums[0]) if nums else default


def get_current_kst(weather_time: Optional[str] = None) -> datetime:
    """도시데이터 WEATHER_TIME이 있으면 그 시간을 우선 사용, 없으면 KST 현재시각."""
    if weather_time:
        try:
            dt = pd.to_datetime(weather_time, errors="coerce")
            if not pd.isna(dt):
                if dt.tzinfo is None:
                    return dt.to_pydatetime().replace(tzinfo=KST)
                return dt.tz_convert("Asia/Seoul").to_pydatetime()
        except Exception:
            pass
    return datetime.now(KST)


def get_time_group(hour: int) -> str:
    # 기존 분석에서 많이 쓰던 시간대명과 최대한 맞춤
    if 0 <= hour < 6:
        return "심야"
    if 6 <= hour < 10:
        return "출근시간"
    if 10 <= hour < 17:
        return "낮시간"
    if 17 <= hour < 21:
        return "퇴근시간"
    return "야간"


def get_weektype(dt: datetime) -> str:
    return "주말" if dt.weekday() >= 5 else "평일"


def get_season(month: int) -> str:
    if month in [3, 4, 5]:
        return "봄"
    if month in [6, 7, 8]:
        return "여름"
    if month in [9, 10, 11]:
        return "가을"
    return "겨울"


def parse_threshold_value(text: str) -> List[float]:
    if pd.isna(text):
        return []
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", str(text))]


def safe_read_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        st.error(f"data/{name} 파일이 없습니다. GitHub data 폴더를 확인해주세요.")
        st.stop()
    # UTF-8-SIG 우선, 실패 시 CP949
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, encoding="cp949")

# ============================================================
# 2. 데이터 로드
# ============================================================

@st.cache_data(show_spinner=False)
def load_static_data():
    base = safe_read_csv("기본예상수요.csv")
    threshold = safe_read_csv("날씨구간화기준.csv")
    coef = safe_read_csv("통합기상조건보정계수.csv")
    priority = safe_read_csv("대여소우선순위.csv")
    station_filter = safe_read_csv("여의도_대여소_필터.csv")
    places = safe_read_csv("서울시_주요_121장소_목록.csv")

    for df in [base, priority, station_filter]:
        for col in df.columns:
            if "대여소" in col and ("ID" in col.upper() or "번호" in col):
                df["station_norm"] = df[col].apply(normalize_station_id)
                break
        if "station_norm" not in df.columns:
            # 가장 그럴듯한 ID 컬럼 찾기
            cand = [c for c in df.columns if "ID" in c.upper() or "번호" in c]
            if cand:
                df["station_norm"] = df[cand[0]].apply(normalize_station_id)

    return base, threshold, coef, priority, station_filter, places

base_df, threshold_df, coef_df, priority_df, station_filter_df, places_df = load_static_data()

# ============================================================
# 3. API 호출
# ============================================================

@st.cache_data(ttl=60, show_spinner=False)
def fetch_bike_api(api_key: str, start: int = 1, end: int = 1000) -> pd.DataFrame:
    """서울시 공공자전거 실시간 대여정보. 1~1000, 1001~2000 등 반복 호출."""
    rows = []
    for s, e in [(1, 1000), (1001, 2000), (2001, 3000)]:
        url = f"http://openapi.seoul.go.kr:8088/{api_key}/json/bikeList/{s}/{e}/"
        try:
            r = requests.get(url, timeout=10)
            js = r.json()
            data = js.get("rentBikeStatus", {}).get("row", [])
            if not data:
                continue
            rows.extend(data)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    rename = {
        "stationId": "station_id",
        "stationName": "station_name",
        "parkingBikeTotCnt": "current_bikes",
        "rackTotCnt": "rack_count",
        "shared": "shared",
        "stationLatitude": "lat",
        "stationLongitude": "lon",
    }
    df = df.rename(columns=rename)
    for c in ["current_bikes", "rack_count", "shared", "lat", "lon"]:
        if c in df.columns:
            df[c] = df[c].apply(to_float)
    df["station_norm"] = df["station_id"].apply(normalize_station_id)
    return df


@st.cache_data(ttl=300, show_spinner=False)
def fetch_citydata_api(api_key: str, area_nm: str) -> Dict:
    """서울시 실시간 도시데이터. JSON 우선, 실패 시 빈 값."""
    area_enc = quote(area_nm)
    url = f"http://openapi.seoul.go.kr:8088/{api_key}/json/citydata/1/5/{area_enc}"
    out = {
        "area_nm": area_nm,
        "temp": np.nan,
        "humidity": np.nan,
        "wind_spd": np.nan,
        "precipitation": 0.0,
        "precpt_type": "",
        "weather_time": "",
        "raw_ok": False,
    }
    try:
        r = requests.get(url, timeout=10)
        js = r.json()
        city = js.get("CITYDATA", {})
        weather = city.get("WEATHER_STTS", [])
        if isinstance(weather, dict):
            weather = [weather]
        if weather:
            w = weather[0]
            out.update(
                {
                    "temp": to_float(w.get("TEMP"), np.nan),
                    "humidity": to_float(w.get("HUMIDITY"), np.nan),
                    "wind_spd": to_float(w.get("WIND_SPD"), np.nan),
                    "precipitation": to_float(w.get("PRECIPITATION"), 0.0),
                    "precpt_type": str(w.get("PRECPT_TYPE", "")),
                    "weather_time": str(w.get("WEATHER_TIME", "")),
                    "raw_ok": True,
                }
            )
    except Exception:
        pass
    return out

# ============================================================
# 4. 날씨 조건 판정
# ============================================================

def get_threshold_row(var_keyword: str) -> Optional[pd.Series]:
    if "변수" not in threshold_df.columns:
        return None
    m = threshold_df[threshold_df["변수"].astype(str).str.contains(var_keyword, na=False)]
    return m.iloc[0] if len(m) else None


def classify_weather(weather: Dict) -> Dict:
    temp = weather.get("temp", np.nan)
    hum = weather.get("humidity", np.nan)
    wind = weather.get("wind_spd", np.nan)
    rain = weather.get("precipitation", 0.0)

    def q_class(value, keyword, labels):
        row = get_threshold_row(keyword)
        nums = parse_threshold_value(row.get("구간화 기준", "")) if row is not None else []
        # 날씨구간화기준 문자열에는 기준 숫자가 2개 이상 있을 수 있음. 앞 2개를 사용.
        if len(nums) >= 2 and not pd.isna(value):
            q1, q2 = nums[0], nums[1]
            if value <= q1:
                return labels[0]
            if value <= q2:
                return labels[1]
            return labels[2]
        return labels[1]

    temp_cond = q_class(temp, "기온", ["저온", "적정", "고온"])
    hum_cond = q_class(hum, "습도", ["낮음", "보통", "높음"])
    wind_cond = q_class(wind, "풍속", ["약풍", "보통", "강풍"])

    # 강수: 기준표의 중앙값 사용. 없으면 0 기준만 사용.
    rain_row = get_threshold_row("강수")
    nums = parse_threshold_value(rain_row.get("구간화 기준", "")) if rain_row is not None else []
    rain_cut = nums[-1] if nums else 0.0
    if pd.isna(rain) or rain <= 0:
        rain_cond = "비없음"
    elif rain <= rain_cut:
        rain_cond = "약한비"
    else:
        rain_cond = "강한비"

    snow_cond = "눈없음"
    weather_cond = f"{temp_cond}_{rain_cond}_{hum_cond}_{wind_cond}_{snow_cond}"
    return {
        "기온조건": temp_cond,
        "강수조건": rain_cond,
        "습도조건": hum_cond,
        "풍속조건": wind_cond,
        "적설조건": snow_cond,
        "기상조건": weather_cond,
    }

# ============================================================
# 5. 수요·후보 계산
# ============================================================

def pick_column(df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
    for k in keywords:
        for c in df.columns:
            if k in str(c):
                return c
    return None


def build_realtime_station_table(bike_df: pd.DataFrame) -> pd.DataFrame:
    """실시간 따릉이 API를 여의도 대여소 목록과 결합."""
    if bike_df.empty:
        return pd.DataFrame()

    # 여의도 필터 ID 목록 사용
    if "station_norm" in station_filter_df.columns:
        ids = set(station_filter_df["station_norm"].astype(str))
        out = bike_df[bike_df["station_norm"].astype(str).isin(ids)].copy()
    else:
        out = bike_df[bike_df["station_name"].astype(str).str.contains("여의|국회|샛강|IFC|KBS|산업은행|유진투자|국민일보|양카라", na=False)].copy()

    # 이름 정리: "202. 국민일보 앞" 형태 유지
    out["대여소명"] = out["station_name"].astype(str).str.strip()
    out["대여소_ID"] = out["station_norm"]
    return out


def get_base_demand_for_now(now: datetime, weather_cond: str) -> pd.DataFrame:
    """현재 시간 조건에 맞는 기본수요 + 기상보정계수 결합."""
    b = base_df.copy()
    # 컬럼명 탐색
    col_time_group = pick_column(b, ["시간대그룹", "시간대 그룹"])
    col_week = pick_column(b, ["평일주말", "평일/주말"])
    col_month = pick_column(b, ["월"])
    col_season = pick_column(b, ["계절"])

    time_group = get_time_group(now.hour)
    weektype = get_weektype(now)
    season = get_season(now.month)
    month = now.month

    mask = pd.Series(True, index=b.index)
    if col_time_group:
        mask &= b[col_time_group].astype(str).eq(time_group)
    if col_week:
        mask &= b[col_week].astype(str).eq(weektype)
    if col_month:
        mask &= pd.to_numeric(b[col_month], errors="coerce").fillna(-1).astype(int).eq(month)
    if col_season:
        mask &= b[col_season].astype(str).eq(season)

    current = b[mask].copy()
    # 너무 빡세서 없으면 시간대/평일/계절 순으로 완화
    if current.empty and col_time_group:
        current = b[b[col_time_group].astype(str).eq(time_group)].copy()
    if current.empty:
        current = b.copy()

    # 대여소별 평균으로 축약
    id_col = "station_norm" if "station_norm" in current.columns else pick_column(current, ["대여소_ID", "대여소ID", "대여소 번호"])
    name_col = pick_column(current, ["대여소명", "대여소 명"])
    out_col = pick_column(current, ["기본예상대여수요", "대여수요"])
    in_col = pick_column(current, ["기본예상반납수요", "반납수요"])
    if id_col is None or out_col is None or in_col is None:
        return pd.DataFrame()

    agg = current.groupby(id_col, as_index=False).agg(
        기본예상대여수요=(out_col, "mean"),
        기본예상반납수요=(in_col, "mean"),
    )
    agg = agg.rename(columns={id_col: "station_norm"})
    if name_col:
        names = current.groupby(id_col, as_index=False)[name_col].first().rename(columns={id_col: "station_norm", name_col: "대여소명_base"})
        agg = agg.merge(names, on="station_norm", how="left")

    # 보정계수: 현재 시간대그룹 + 현재 기상조건 우선
    c = coef_df.copy()
    c_time = pick_column(c, ["시간대그룹", "시간대 그룹"])
    c_weather = pick_column(c, ["기상조건"])
    rent_coef_col = pick_column(c, ["대여수요_날씨보정계수", "대여수요보정계수"])
    ret_coef_col = pick_column(c, ["반납수요_날씨보정계수", "반납수요보정계수"])

    rent_coef, ret_coef = 1.0, 1.0
    if c_time and c_weather and rent_coef_col and ret_coef_col:
        m = c[(c[c_time].astype(str) == time_group) & (c[c_weather].astype(str) == weather_cond)]
        if m.empty:
            m = c[c[c_time].astype(str) == time_group]
        if not m.empty:
            rent_coef = to_float(m[rent_coef_col].mean(), 1.0)
            ret_coef = to_float(m[ret_coef_col].mean(), 1.0)

    agg["대여수요_날씨보정계수"] = rent_coef
    agg["반납수요_날씨보정계수"] = ret_coef
    agg["예측대여수요"] = agg["기본예상대여수요"] * rent_coef
    agg["예측반납수요"] = agg["기본예상반납수요"] * ret_coef
    return agg


def merge_priority(df: pd.DataFrame) -> pd.DataFrame:
    p = priority_df.copy()
    if "station_norm" not in p.columns:
        return df
    score_col = pick_column(p, ["우선순위기초점수", "우선순위"])
    freq_col = pick_column(p, ["평소이용빈도", "평균총이용건수"])
    cols = ["station_norm"]
    if score_col:
        cols.append(score_col)
    if freq_col:
        cols.append(freq_col)
    p2 = p[cols].copy().drop_duplicates("station_norm")
    rename = {}
    if score_col:
        rename[score_col] = "우선순위점수"
    if freq_col:
        rename[freq_col] = "평소이용빈도"
    p2 = p2.rename(columns=rename)
    out = df.merge(p2, on="station_norm", how="left")
    out["우선순위점수"] = pd.to_numeric(out.get("우선순위점수", 0), errors="coerce").fillna(0)
    out["평소이용빈도"] = pd.to_numeric(out.get("평소이용빈도", 0), errors="coerce").fillna(0)
    return out


def build_candidates(rt_df: pd.DataFrame, demand_df: pd.DataFrame, low_ratio: float, high_ratio: float,
                     pickup_top_n: int, delivery_top_n: int) -> pd.DataFrame:
    """실시간 재고 + 예측수요로 재배치 후보와 공급수거 후보 생성."""
    if rt_df.empty:
        return pd.DataFrame()
    df = rt_df.merge(demand_df, on="station_norm", how="left")
    df = merge_priority(df)

    for c in ["예측대여수요", "예측반납수요", "기본예상대여수요", "기본예상반납수요"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["예상재고"] = df["current_bikes"] + df["예측반납수요"] - df["예측대여수요"]
    df["부족기준재고"] = df["rack_count"] * low_ratio
    df["과잉기준재고"] = df["rack_count"] * high_ratio
    df["수거필요량_raw"] = (df["예상재고"] - df["과잉기준재고"]).clip(lower=0)
    df["재배치필요량_raw"] = (df["부족기준재고"] - df["예상재고"]).clip(lower=0)

    # 재배치 후보: 부족량 큰 곳 우선
    deliveries = df[df["재배치필요량_raw"] > 0].copy()
    deliveries["후보유형"] = "재배치"
    deliveries["필요량"] = np.ceil(deliveries["재배치필요량_raw"]).astype(int)
    deliveries["후보점수"] = deliveries["필요량"] * (1 + deliveries["우선순위점수"].fillna(0))
    deliveries = deliveries.sort_values(["후보점수", "필요량"], ascending=False).head(delivery_top_n)

    # 1차 과잉수거 후보
    pickups_over = df[df["수거필요량_raw"] > 0].copy()
    pickups_over["후보유형"] = "과잉수거"
    pickups_over["필요량"] = np.ceil(pickups_over["수거필요량_raw"]).astype(int)
    pickups_over["후보점수"] = pickups_over["필요량"] * (1 + pickups_over["우선순위점수"].fillna(0))

    # 2차 공급수거 후보: 현재 자전거가 많고, 예측대여수요·평소이용빈도는 낮은 곳
    # 수거 가능량은 부족기준 이하로 떨어지지 않는 범위
    supply = df.copy()
    supply["공급가능량"] = (supply["current_bikes"] - supply["부족기준재고"]).clip(lower=0)
    # 최소 1대 이상 공급 가능해야 함
    supply = supply[supply["공급가능량"] >= 1].copy()
    # 이미 재배치 후보는 공급 후보에서 제외
    delivery_ids = set(deliveries["station_norm"].astype(str))
    supply = supply[~supply["station_norm"].astype(str).isin(delivery_ids)]
    # 과잉 후보 제외 후 별도 공급 후보
    over_ids = set(pickups_over["station_norm"].astype(str))
    supply = supply[~supply["station_norm"].astype(str).isin(over_ids)]

    if not supply.empty:
        def minmax(s):
            s = pd.to_numeric(s, errors="coerce").fillna(0)
            if s.max() == s.min():
                return pd.Series(0.0, index=s.index)
            return (s - s.min()) / (s.max() - s.min())
        supply["현재자전거점수"] = minmax(supply["current_bikes"])
        supply["공급가능점수"] = minmax(supply["공급가능량"])
        supply["낮은예측대여점수"] = 1 - minmax(supply["예측대여수요"])
        supply["낮은이용빈도점수"] = 1 - minmax(supply["평소이용빈도"])
        supply["후보점수"] = (
            0.35 * supply["현재자전거점수"] +
            0.35 * supply["공급가능점수"] +
            0.20 * supply["낮은예측대여점수"] +
            0.10 * supply["낮은이용빈도점수"]
        )
        supply["후보유형"] = "공급수거"
        supply["필요량"] = np.floor(supply["공급가능량"].clip(upper=DEFAULT_CAPACITY)).astype(int)
    else:
        supply["후보유형"] = []

    # 수거 후보는 과잉 먼저, 부족하면 공급 후보로 채움
    pickups = pickups_over.sort_values(["후보점수", "필요량"], ascending=False).copy()
    if len(pickups) < pickup_top_n and not supply.empty:
        supplement = supply.sort_values(["후보점수", "current_bikes"], ascending=False).head(pickup_top_n - len(pickups))
        pickups = pd.concat([pickups, supplement], ignore_index=True)
    else:
        pickups = pickups.head(pickup_top_n)

    candidates = pd.concat([pickups, deliveries], ignore_index=True)
    if candidates.empty:
        return candidates

    # 표시용 및 정수화
    candidates["필요량"] = pd.to_numeric(candidates["필요량"], errors="coerce").fillna(0).astype(int)
    candidates["현재자전거수"] = candidates["current_bikes"].astype(int)
    candidates["거치대수"] = candidates["rack_count"].astype(int)
    candidates["위도"] = candidates["lat"]
    candidates["경도"] = candidates["lon"]
    candidates["대여소명"] = candidates["대여소명"].fillna(candidates["station_name"])
    return candidates

# ============================================================
# 6. 경로 생성: 차량 순차 수거→재배치
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))


def nearest_idx(current_lat, current_lon, df: pd.DataFrame) -> Optional[int]:
    if df.empty:
        return None
    dists = df.apply(lambda r: haversine_km(current_lat, current_lon, r["위도"], r["경도"]), axis=1)
    return dists.idxmin()


def sequential_vehicle_routes(candidates: pd.DataFrame, vehicle_count: int, capacity: int) -> Tuple[List[Dict], pd.DataFrame]:
    """
    차량 1 → 차량 2 순서로 실행.
    각 차량은 빈 차량으로 출발해 수거 후보에서 싣고, 재배치 후보에 배치하고, 마지막 재배치 지점에서 종료.
    방문 번호 누락 방지를 위해 각 route_step에 visit_order를 0부터 연속 부여.
    """
    if candidates.empty:
        return [], pd.DataFrame()

    df = candidates.copy().reset_index(drop=True)
    df["남은수거"] = np.where(df["후보유형"].isin(["과잉수거", "공급수거"]), df["필요량"], 0).astype(int)
    df["남은재배치"] = np.where(df["후보유형"].eq("재배치"), df["필요량"], 0).astype(int)
    df["처리수거량"] = 0
    df["처리재배치량"] = 0

    routes = []

    for k in range(1, vehicle_count + 1):
        remaining_vehicles = vehicle_count - k + 1
        remaining_delivery_total = int(df["남은재배치"].sum())
        if remaining_delivery_total <= 0:
            # 빈 차량도 출발/종료만 보여주지 않음. 단, 차량 탭에는 0개로 표시됨.
            routes.append({"vehicle": k, "steps": [], "distance_km": 0.0, "osrm_failures": 0})
            continue

        # 한 차량이 독식하지 않도록 목표 처리량 배분
        vehicle_target = min(capacity, math.ceil(remaining_delivery_total / remaining_vehicles))
        load = 0
        cur_lat, cur_lon = DEPOT["lat"], DEPOT["lon"]
        steps = [{
            "visit_order": 0,
            "name": DEPOT["name"],
            "lat": DEPOT["lat"],
            "lon": DEPOT["lon"],
            "action": "출발",
            "qty": 0,
            "load_after": 0,
            "station_norm": "DEPOT",
        }]

        # 1) 목표량만큼 수거
        pickup_total = 0
        while pickup_total < vehicle_target:
            pickup_df = df[df["남은수거"] > 0].copy()
            if pickup_df.empty:
                break
            idx = nearest_idx(cur_lat, cur_lon, pickup_df)
            if idx is None:
                break
            can_pick = int(min(df.loc[idx, "남은수거"], capacity - load, vehicle_target - pickup_total))
            if can_pick <= 0:
                break
            df.loc[idx, "남은수거"] -= can_pick
            df.loc[idx, "처리수거량"] += can_pick
            load += can_pick
            pickup_total += can_pick
            cur_lat, cur_lon = float(df.loc[idx, "위도"]), float(df.loc[idx, "경도"])
            steps.append({
                "visit_order": len(steps),
                "name": str(df.loc[idx, "대여소명"]),
                "lat": cur_lat,
                "lon": cur_lon,
                "action": "수거",
                "qty": can_pick,
                "load_after": load,
                "station_norm": str(df.loc[idx, "station_norm"]),
            })

        # 수거하지 못했으면 경로 생성 불가. 다음 차량으로 넘김.
        if load <= 0:
            routes.append({"vehicle": k, "steps": [], "distance_km": 0.0, "osrm_failures": 0})
            continue

        # 2) 싣고 있는 만큼 재배치
        while load > 0 and df["남은재배치"].sum() > 0:
            delivery_df = df[df["남은재배치"] > 0].copy()
            if delivery_df.empty:
                break
            idx = nearest_idx(cur_lat, cur_lon, delivery_df)
            if idx is None:
                break
            can_drop = int(min(df.loc[idx, "남은재배치"], load))
            if can_drop <= 0:
                break
            df.loc[idx, "남은재배치"] -= can_drop
            df.loc[idx, "처리재배치량"] += can_drop
            load -= can_drop
            cur_lat, cur_lon = float(df.loc[idx, "위도"]), float(df.loc[idx, "경도"])
            steps.append({
                "visit_order": len(steps),
                "name": str(df.loc[idx, "대여소명"]),
                "lat": cur_lat,
                "lon": cur_lon,
                "action": "재배치",
                "qty": can_drop,
                "load_after": load,
                "station_norm": str(df.loc[idx, "station_norm"]),
            })

        # 출발지만 있으면 빈 경로 처리
        if len(steps) <= 1:
            steps = []

        routes.append({"vehicle": k, "steps": steps, "distance_km": 0.0, "osrm_failures": 0})

    # 결과 테이블
    df["남은불균형"] = df["남은수거"] + df["남은재배치"]
    return routes, df

# ============================================================
# 7. OSRM 경로와 지도
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def osrm_route(lat1: float, lon1: float, lat2: float, lon2: float) -> Tuple[List[Tuple[float, float]], float, bool]:
    """OSRM 자동차 도로 경로. 반환: [(lat, lon)], km, failed"""
    url = (
        "https://router.project-osrm.org/route/v1/driving/"
        f"{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson"
    )
    try:
        r = requests.get(url, timeout=12)
        js = r.json()
        if js.get("code") == "Ok" and js.get("routes"):
            route = js["routes"][0]
            coords = route["geometry"]["coordinates"]
            pts = [(lat, lon) for lon, lat in coords]
            km = float(route.get("distance", 0)) / 1000
            return pts, km, False
    except Exception:
        pass
    # fallback: 직선
    km = haversine_km(lat1, lon1, lat2, lon2)
    return [(lat1, lon1), (lat2, lon2)], km, True


def add_number_marker(m: folium.Map, lat: float, lon: float, num: int, color: str, tooltip: str):
    """번호가 반드시 보이도록 DivIcon 번호 마커 + 작은 정보 마커를 함께 표시."""
    folium.Marker(
        [lat, lon],
        tooltip=tooltip,
        icon=folium.DivIcon(
            html=f"""
            <div style="
                width:28px;height:28px;border-radius:50%;background:white;
                border:3px solid {color};color:#111827;font-weight:800;
                display:flex;align-items:center;justify-content:center;
                font-size:15px;box-shadow:0 1px 4px rgba(0,0,0,.25);">
                {num}
            </div>
            """
        ),
    ).add_to(m)


def add_arrowed_polyline(m: folium.Map, pts: List[Tuple[float, float]], color: str = "#178c3a"):
    """도로 경로 + 방향 화살표. PolyLineTextPath로 선 위 방향 표시."""
    if len(pts) < 2:
        return
    line = folium.PolyLine(pts, color=color, weight=6, opacity=0.82).add_to(m)
    # 반복 화살표: 너무 촘촘하지 않게 130px 간격
    try:
        PolyLineTextPath(
            line,
            "▶",
            repeat=True,
            offset=8,
            attributes={"fill": "#087f23", "font-weight": "bold", "font-size": "18"},
        ).add_to(m)
    except Exception:
        pass


def jitter(lat: float, lon: float, idx: int) -> Tuple[float, float]:
    """동일 좌표/겹침 마커 번호 누락 방지를 위한 아주 작은 오프셋."""
    if idx == 0:
        return lat, lon
    angle = (idx * 47) * math.pi / 180
    r = 0.000055 * (1 + (idx % 3) * 0.25)
    return lat + r * math.sin(angle), lon + r * math.cos(angle)


def make_realtime_map(rt_df: pd.DataFrame, candidates: pd.DataFrame) -> folium.Map:
    m = folium.Map(location=[DEPOT["lat"], DEPOT["lon"]], zoom_start=14, tiles="CartoDB positron")
    folium.Marker(
        [DEPOT["lat"], DEPOT["lon"]],
        tooltip=f"출발지: {DEPOT['name']}",
        icon=folium.Icon(color="black", icon="home", prefix="fa"),
    ).add_to(m)
    cand_ids = set(candidates["station_norm"].astype(str)) if not candidates.empty else set()
    pickup_ids = set(candidates[candidates["후보유형"].isin(["과잉수거", "공급수거"])]["station_norm"].astype(str)) if not candidates.empty else set()
    delivery_ids = set(candidates[candidates["후보유형"].eq("재배치")]["station_norm"].astype(str)) if not candidates.empty else set()

    for _, r in rt_df.iterrows():
        sid = str(r.get("station_norm", ""))
        color = "gray"
        if sid in pickup_ids:
            color = "red"
        elif sid in delivery_ids:
            color = "blue"
        elif r.get("current_bikes", 0) >= max(1, r.get("rack_count", 1) * 0.6):
            color = "orange"
        folium.CircleMarker(
            [r["lat"], r["lon"]],
            radius=5 + min(10, float(r.get("current_bikes", 0)) / 3),
            color=color,
            fill=True,
            fill_opacity=0.65,
            tooltip=f"{r.get('대여소명','')} | 현재 {int(r.get('current_bikes',0))}대 / 거치대 {int(r.get('rack_count',0))}",
        ).add_to(m)
    return m


def make_candidate_map(candidates: pd.DataFrame) -> folium.Map:
    m = folium.Map(location=[DEPOT["lat"], DEPOT["lon"]], zoom_start=14, tiles="CartoDB positron")
    folium.Marker([DEPOT["lat"], DEPOT["lon"]], tooltip="출발지: 여의도 복지관", icon=folium.Icon(color="black", icon="home", prefix="fa")).add_to(m)
    if candidates.empty:
        return m
    for _, r in candidates.iterrows():
        color = "red" if r["후보유형"] in ["과잉수거", "공급수거"] else "blue"
        folium.Marker(
            [r["위도"], r["경도"]],
            tooltip=f"{r['대여소명']} | {r['후보유형']} {int(r['필요량'])}대",
            icon=folium.Icon(color=color, icon="info-sign"),
        ).add_to(m)
    return m


def make_vehicle_map(route: Dict) -> Tuple[folium.Map, pd.DataFrame]:
    m = folium.Map(location=[DEPOT["lat"], DEPOT["lon"]], zoom_start=14, tiles="CartoDB positron")
    steps = route.get("steps", [])
    if not steps:
        folium.Marker([DEPOT["lat"], DEPOT["lon"]], tooltip="출발지: 여의도 복지관", icon=folium.Icon(color="black", icon="home", prefix="fa")).add_to(m)
        return m, pd.DataFrame([{"순서": 0, "장소": DEPOT["name"], "작업": "배정 없음", "수량": 0, "작업 후 적재량": 0}])

    # 경로 구간 그리기
    total_km = 0.0
    failures = 0
    full_bounds = []
    for i in range(len(steps) - 1):
        a, b = steps[i], steps[i+1]
        pts, km, fail = osrm_route(a["lat"], a["lon"], b["lat"], b["lon"])
        total_km += km
        failures += int(fail)
        add_arrowed_polyline(m, pts)
        full_bounds.extend(pts)

    # 번호 마커: 0은 출발, 이후 1..n. 모든 방문 순서가 보이도록 연속 번호 사용
    coord_seen = {}
    for s in steps:
        key = (round(float(s["lat"]), 6), round(float(s["lon"]), 6))
        coord_seen[key] = coord_seen.get(key, 0) + 1
        jlat, jlon = jitter(float(s["lat"]), float(s["lon"]), coord_seen[key]-1)
        if s["action"] == "출발":
            folium.Marker([jlat, jlon], tooltip="0. 출발: 여의도 복지관", icon=folium.Icon(color="black", icon="home", prefix="fa")).add_to(m)
        else:
            color = "#ef4444" if s["action"] == "수거" else "#2563eb"
            add_number_marker(
                m,
                jlat,
                jlon,
                int(s["visit_order"]),
                color,
                f"{int(s['visit_order'])}. {s['name']} | {s['action']} {int(s['qty'])}대 | 적재 {int(s['load_after'])}대",
            )

    if full_bounds:
        try:
            m.fit_bounds(full_bounds, padding=(30, 30))
        except Exception:
            pass

    route["distance_km"] = total_km
    route["osrm_failures"] = failures

    table = pd.DataFrame([
        {
            "순서": int(s["visit_order"]),
            "장소": s["name"],
            "작업": s["action"],
            "수량": int(s["qty"]),
            "작업 후 적재량": int(s["load_after"]),
        }
        for s in steps
    ])
    return m, table

# ============================================================
# 8. Streamlit UI
# ============================================================

st.title("🚲 여의도 따릉이 수거·재배치 경로 추천 대시보드")
st.caption("배포용 버전: 실시간 API + 과거 수요모델 + 공급수거/재배치 휴리스틱 경로 추천. 차량별 경로 지도는 각각 분리해서 표시합니다.")

# Secrets 불러오기
try:
    BIKE_KEY = st.secrets["SEOUL_BIKE_API_KEY"]
    CITY_KEY = st.secrets["SEOUL_CITYDATA_API_KEY"]
except Exception:
    st.error("Streamlit Secrets에 SEOUL_BIKE_API_KEY, SEOUL_CITYDATA_API_KEY를 입력해주세요.")
    st.stop()

with st.sidebar:
    st.header("⚙️ 실행 설정")
    # 장소명 목록
    place_col = pick_column(places_df, ["AREA_NM", "장소명", "장소", "핫스팟"])
    if place_col:
        place_options = places_df[place_col].dropna().astype(str).unique().tolist()
    else:
        place_options = ["여의도"]
    default_idx = 0
    for cand in ["여의도", "여의도한강공원", "여의도공원", "국회의사당"]:
        matches = [i for i, p in enumerate(place_options) if cand in p]
        if matches:
            default_idx = matches[0]
            break
    area_nm = st.selectbox("도시데이터 장소명", place_options, index=default_idx)

    vehicle_count = st.slider("차량 수", 1, 4, DEFAULT_VEHICLE_COUNT)
    capacity = st.slider("차량 용량", 5, 30, DEFAULT_CAPACITY)
    pickup_top_n = st.slider("수거/공급 후보 수", 3, 20, DEFAULT_PICKUP_TOP_N)
    delivery_top_n = st.slider("재배치 후보 수", 3, 20, DEFAULT_DELIVERY_TOP_N)
    low_ratio = st.slider("부족 기준 비율", 0.1, 0.6, DEFAULT_LOW_RATIO, 0.05)
    high_ratio = st.slider("과잉 기준 비율", 0.5, 1.2, DEFAULT_HIGH_RATIO, 0.05)

    st.info("출발지는 여의도 복지관으로 고정됩니다.\n위도 37.518133 / 경도 126.930776")
    run_btn = st.button("🚚 경로 추천 실행", type="primary")
    reset_btn = st.button("결과 초기화")

if reset_btn:
    for k in ["result_ready", "context", "rt_df", "candidates", "routes", "result_df"]:
        st.session_state.pop(k, None)
    st.rerun()

if run_btn:
    with st.spinner("실시간 API 호출 및 경로 계산 중입니다..."):
        bike_raw = fetch_bike_api(BIKE_KEY)
        rt_df = build_realtime_station_table(bike_raw)
        city_weather = fetch_citydata_api(CITY_KEY, area_nm)
        weather_cls = classify_weather(city_weather)
        now = get_current_kst(city_weather.get("weather_time"))
        demand_now = get_base_demand_for_now(now, weather_cls["기상조건"])
        candidates = build_candidates(rt_df, demand_now, low_ratio, high_ratio, pickup_top_n, delivery_top_n)
        routes, result_df = sequential_vehicle_routes(candidates, vehicle_count, capacity)

        st.session_state["result_ready"] = True
        st.session_state["context"] = {
            "now": now,
            "time_group": get_time_group(now.hour),
            "weektype": get_weektype(now),
            "season": get_season(now.month),
            "weather": city_weather,
            "weather_cls": weather_cls,
            "area_nm": area_nm,
            "capacity": capacity,
            "vehicle_count": vehicle_count,
            "low_ratio": low_ratio,
            "high_ratio": high_ratio,
        }
        st.session_state["rt_df"] = rt_df
        st.session_state["candidates"] = candidates
        st.session_state["routes"] = routes
        st.session_state["result_df"] = result_df
    st.success("경로 추천이 완료되었습니다.")

if "result_ready" not in st.session_state:
    st.info("왼쪽 설정을 확인한 뒤 **경로 추천 실행**을 눌러주세요.")
    st.stop()

ctx = st.session_state["context"]
rt_df = st.session_state["rt_df"]
candidates = st.session_state["candidates"]
routes = st.session_state["routes"]
result_df = st.session_state["result_df"]

# ============================================================
# 9. 결과 화면
# ============================================================

st.subheader("① 현재 조건")
cols = st.columns(8)
with cols[0]: metric_card("현재 시점", ctx["now"].strftime("%m/%d %H:%M"))
with cols[1]: metric_card("시간대그룹", ctx["time_group"])
with cols[2]: metric_card("평일/주말", ctx["weektype"])
with cols[3]: metric_card("계절", ctx["season"])
with cols[4]: metric_card("기상조건", ctx["weather_cls"]["기상조건"])
with cols[5]: metric_card("기온", f"{to_float(ctx['weather'].get('temp'), np.nan):.1f}℃")
with cols[6]: metric_card("습도", f"{to_float(ctx['weather'].get('humidity'), np.nan):.1f}%")
with cols[7]: metric_card("풍속", f"{to_float(ctx['weather'].get('wind_spd'), np.nan):.1f}m/s")

cols2 = st.columns(4)
with cols2[0]: metric_card("강수량", f"{to_float(ctx['weather'].get('precipitation'), 0):.1f}mm")
with cols2[1]: metric_card("날씨 업데이트", str(ctx["weather"].get("weather_time", "-"))[:16])
with cols2[2]: metric_card("도시데이터 장소", ctx["area_nm"])
with cols2[3]: metric_card("출발지", "여의도 복지관")

st.subheader("② 수거·재배치 후보 및 개선 효과")
if candidates.empty:
    st.warning("현재 설정에서 후보가 생성되지 않았습니다. 부족 기준/후보 수를 조정해보세요.")
else:
    before_imb = int(candidates["필요량"].sum())
    processed = int(result_df.get("처리수거량", pd.Series(dtype=int)).sum() + result_df.get("처리재배치량", pd.Series(dtype=int)).sum()) if not result_df.empty else 0
    after_imb = int(result_df.get("남은불균형", pd.Series(dtype=int)).sum()) if not result_df.empty else before_imb
    improvement = (before_imb - after_imb) / before_imb * 100 if before_imb > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("처리 전 후보 불균형", f"{before_imb}대")
    with c2: metric_card("휴리스틱 처리량", f"{processed}대")
    with c3: metric_card("처리 후 남은 불균형", f"{after_imb}대")
    with c4: metric_card("개선율", f"{improvement:.1f}%")

    with st.expander("계산 기준과 해석 보기", expanded=False):
        st.markdown(
            f"""
            **현재 시간 조건**은 Streamlit 서버 시간이 아니라 한국시간 또는 도시데이터의 날씨 업데이트 시간을 기준으로 판정합니다.  
            현재 조건은 `{ctx['time_group']}`, `{ctx['weektype']}`, `{ctx['season']}`, `{ctx['weather_cls']['기상조건']}`입니다.

            **예상재고 계산식**  
            `예상재고 = 현재자전거수 + 예측반납수요 - 예측대여수요`

            **재배치 후보**  
            `예상재고 < 부족기준비율 × 거치대수` 인 대여소입니다.

            **수거 후보**  
            먼저 `예상재고 > 과잉기준비율 × 거치대수` 인 과잉 대여소를 찾습니다.  
            과잉 대여소가 부족하면, 현재 자전거 수가 많고 예측 대여수요와 평소 이용 빈도가 낮은 대여소를 **공급수거 후보**로 추가합니다.  
            공급수거 후보는 수거 후에도 부족 기준 이하로 떨어지지 않는 범위에서만 수거합니다.

            **차량 경로 방식**  
            차량 1부터 순차적으로 계산합니다. 각 차량은 여의도 복지관에서 빈 차량으로 출발해 수거 후보에서 자전거를 싣고, 재배치 후보에 배치한 뒤 마지막 재배치 지점에서 종료합니다.  
            한 차량이 모든 수요를 독식하지 않도록 남은 재배치 수요를 남은 차량 수로 나누어 차량별 목표 처리량을 정합니다.
            """
        )

st.subheader("③ 실시간 대여소 현황 지도")
st.markdown('<div class="section-help">빨강은 수거/공급 후보, 파랑은 재배치 후보, 주황은 현재 자전거가 많은 대여소입니다.</div>', unsafe_allow_html=True)
st_folium(make_realtime_map(rt_df, candidates), width=1200, height=520)

st.subheader("④ 후보 위치 개요 지도")
st.markdown('<div class="section-help">이 지도는 후보 위치만 보여줍니다. 차량별 실제 경로는 아래 차량별 탭에서 따로 확인합니다.</div>', unsafe_allow_html=True)
st_folium(make_candidate_map(candidates), width=1200, height=520)

st.subheader("⑤ 차량별 경로 지도")
if not routes:
    st.warning("차량 경로가 생성되지 않았습니다.")
else:
    tabs = st.tabs([f"차량 {r['vehicle']}" for r in routes])
    for tab, route in zip(tabs, routes):
        with tab:
            vmap, visit_table = make_vehicle_map(route)
            # make_vehicle_map에서 distance/osrm_failures가 route에 업데이트됨
            a, b, c, d = st.columns(4)
            with a: metric_card("차량", f"{route['vehicle']}번")
            with b: metric_card("방문 지점 수", f"{max(0, len(route.get('steps', [])) - 1)}개")
            with c: metric_card("예상 이동거리", f"{route.get('distance_km', 0):.2f}km")
            with d: metric_card("OSRM 실패 구간", f"{route.get('osrm_failures', 0)}개")
            st_folium(vmap, width=1200, height=520)
            st.markdown("#### 방문 순서")
            st.dataframe(visit_table, use_container_width=True, hide_index=True)

st.subheader("⑥ 처리 결과 상세")
if not result_df.empty:
    show_cols = [
        "대여소명", "후보유형", "필요량", "현재자전거수", "거치대수", "예상재고",
        "처리수거량", "처리재배치량", "남은수거", "남은재배치", "남은불균형"
    ]
    show_cols = [c for c in show_cols if c in result_df.columns]
    st.dataframe(result_df[show_cols], use_container_width=True, hide_index=True)
else:
    st.info("처리 결과가 없습니다.")

# 선택적 다운로드: 화면 표시가 핵심이고, 저장은 보조 기능
st.download_button(
    "처리 결과 CSV 다운로드",
    data=(result_df.to_csv(index=False, encoding="utf-8-sig") if not result_df.empty else "").encode("utf-8-sig"),
    file_name="ddareungi_route_result.csv",
    mime="text/csv",
)
