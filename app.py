# -*- coding: utf-8 -*-
"""
여의도 따릉이 수거·재배치 경로 추천 대시보드, 배포용 경량 버전
- Gurobi 미사용
- 실시간 API + 과거 수요모델 + Greedy 휴리스틱 경로 추천
"""

from __future__ import annotations

import io
import math
import re
import urllib.parse
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import folium
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="여의도 따릉이 수거·재배치 대시보드",
    page_icon="🚲",
    layout="wide",
)

DATA_DIR = "data"
KST = ZoneInfo("Asia/Seoul")

# -----------------------------
# 유틸 함수
# -----------------------------
def get_secret_or_empty(key: str) -> str:
    try:
        return str(st.secrets.get(key, ""))
    except Exception:
        return ""


def to_float(x, default=np.nan):
    try:
        if x is None:
            return default
        if isinstance(x, str):
            x = x.strip().replace(",", "")
            if x in ["", "-", "점검중"]:
                return default
            # 서울 도시데이터 강수량은 "강수없음" 등이 섞일 수 있음
            if "없" in x:
                return 0.0
            m = re.search(r"-?\d+(?:\.\d+)?", x)
            if m:
                return float(m.group(0))
            return default
        return float(x)
    except Exception:
        return default


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    """위경도 직선거리, meter."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def season_from_month(m: int) -> str:
    if m in [3, 4, 5]:
        return "봄"
    if m in [6, 7, 8]:
        return "여름"
    if m in [9, 10, 11]:
        return "가을"
    return "겨울"


def time_group_from_hour(h: int) -> str:
    # 기존 분석 코드의 시간대그룹과 맞추기 위한 일반적 구분
    if 0 <= h <= 5:
        return "심야"
    if 6 <= h <= 9:
        return "출근시간"
    if 10 <= h <= 16:
        return "낮시간"
    if 17 <= h <= 20:
        return "퇴근시간"
    return "야간"


def weektype_from_date(dt: datetime) -> str:
    return "주말" if dt.weekday() >= 5 else "평일"


@st.cache_data
def load_csv_data():
    data = {
        "thresholds": pd.read_csv(f"{DATA_DIR}/날씨구간화기준.csv"),
        "base": pd.read_csv(f"{DATA_DIR}/기본예상수요.csv"),
        "weather_coef": pd.read_csv(f"{DATA_DIR}/통합기상조건보정계수.csv"),
        "priority": pd.read_csv(f"{DATA_DIR}/대여소우선순위.csv"),
        "time_priority": pd.read_csv(f"{DATA_DIR}/시간대별우선순위.csv"),
        "station_filter": pd.read_csv(f"{DATA_DIR}/여의도_대여소_필터.csv"),
        "places": pd.read_csv(f"{DATA_DIR}/서울시_주요_121장소_목록.csv"),
    }
    # ID는 문자열로 통일
    for key in ["base", "priority", "time_priority", "station_filter"]:
        if "대여소_ID" in data[key].columns:
            data[key]["대여소_ID"] = data[key]["대여소_ID"].astype(str).str.strip()
    return data


def parse_thresholds(th_df: pd.DataFrame) -> dict:
    """날씨구간화기준.csv의 기준 문장을 숫자 기준으로 변환."""
    out = {}
    for _, row in th_df.iterrows():
        var = str(row.get("변수", ""))
        text = str(row.get("구간화 기준", ""))
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]
        if var in ["기온", "습도", "풍속"] and len(nums) >= 2:
            out[var] = (nums[0], nums[1])
        elif var == "강수량" and len(nums) >= 1:
            # 0과 중앙값이 같이 잡히므로 마지막 숫자를 중앙값으로 사용
            out[var] = nums[-1]
    return out


def classify_weather(temp, humidity, wind_spd, rain, thresholds: dict) -> dict:
    tq1, tq2 = thresholds.get("기온", (5, 25))
    hq1, hq2 = thresholds.get("습도", (40, 70))
    wq1, wq2 = thresholds.get("풍속", (2, 5))
    rq = thresholds.get("강수량", 0.5)

    temp_cond = "저온" if temp <= tq1 else ("적정" if temp <= tq2 else "고온")
    humid_cond = "낮음" if humidity <= hq1 else ("보통" if humidity <= hq2 else "높음")
    wind_cond = "약풍" if wind_spd <= wq1 else ("보통" if wind_spd <= wq2 else "강풍")
    rain_cond = "비없음" if rain <= 0 else ("약한비" if rain <= rq else "강한비")
    snow_cond = "눈없음"  # 도시데이터 실시간 API에는 적설량이 명확히 없으므로 기본값 처리
    weather_cond = f"{temp_cond}_{rain_cond}_{humid_cond}_{wind_cond}_{snow_cond}"
    return {
        "기온조건": temp_cond,
        "강수조건": rain_cond,
        "습도조건": humid_cond,
        "풍속조건": wind_cond,
        "적설조건": snow_cond,
        "기상조건": weather_cond,
    }


# -----------------------------
# API 호출
# -----------------------------
@st.cache_data(ttl=60)
def fetch_bike_api(api_key: str) -> pd.DataFrame:
    if not api_key:
        raise ValueError("따릉이 API 키가 비어 있습니다.")

    rows = []
    # 서울 전체 대여소는 보통 1~3000 범위 내에 있음
    for start, end in [(1, 1000), (1001, 2000), (2001, 3000)]:
        url = f"http://openapi.seoul.go.kr:8088/{api_key}/json/bikeList/{start}/{end}/"
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        js = r.json()
        if "rentBikeStatus" not in js:
            continue
        rows.extend(js["rentBikeStatus"].get("row", []))

    if not rows:
        raise ValueError("따릉이 API에서 대여소 데이터를 가져오지 못했습니다.")

    df = pd.DataFrame(rows)
    rename = {
        "stationId": "대여소_ID",
        "stationName": "대여소명_API",
        "parkingBikeTotCnt": "현재자전거수",
        "rackTotCnt": "거치대수",
        "shared": "현재거치율",
        "stationLatitude": "위도",
        "stationLongitude": "경도",
    }
    df = df.rename(columns=rename)
    keep = [c for c in rename.values() if c in df.columns]
    df = df[keep].copy()
    df["대여소_ID"] = df["대여소_ID"].astype(str).str.strip()
    for c in ["현재자전거수", "거치대수", "현재거치율", "위도", "경도"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def find_key_recursive(obj, target_key):
    """중첩 dict/list에서 첫 번째 target_key 값 탐색."""
    if isinstance(obj, dict):
        if target_key in obj:
            return obj[target_key]
        for v in obj.values():
            res = find_key_recursive(v, target_key)
            if res is not None:
                return res
    elif isinstance(obj, list):
        for it in obj:
            res = find_key_recursive(it, target_key)
            if res is not None:
                return res
    return None


@st.cache_data(ttl=300)
def fetch_city_weather(api_key: str, area_name: str) -> dict:
    if not api_key:
        raise ValueError("도시데이터 API 키가 비어 있습니다.")
    if not area_name:
        raise ValueError("도시데이터 장소명이 비어 있습니다.")

    enc_area = urllib.parse.quote(area_name)
    url = f"http://openapi.seoul.go.kr:8088/{api_key}/json/citydata/1/5/{enc_area}"
    r = requests.get(url, timeout=12)
    r.raise_for_status()
    js = r.json()

    # CITYDATA 구조 안에서 날씨 값을 유연하게 탐색
    temp = to_float(find_key_recursive(js, "TEMP"), 0)
    humidity = to_float(find_key_recursive(js, "HUMIDITY"), 0)
    wind_spd = to_float(find_key_recursive(js, "WIND_SPD"), 0)
    rain = to_float(find_key_recursive(js, "PRECIPITATION"), 0)
    ptype = find_key_recursive(js, "PRECPT_TYPE")
    weather_time = find_key_recursive(js, "WEATHER_TIME")

    return {
        "AREA_NM": area_name,
        "TEMP": temp,
        "HUMIDITY": humidity,
        "WIND_SPD": wind_spd,
        "PRECIPITATION": rain,
        "PRECPT_TYPE": ptype,
        "WEATHER_TIME": weather_time,
        "raw": js,
    }


# -----------------------------
# 수요·후보 계산
# -----------------------------
def make_realtime_candidates(
    bike_df: pd.DataFrame,
    data: dict,
    weather_cond: str,
    now: datetime,
    L: float,
    U: float,
) -> pd.DataFrame:
    station_filter = data["station_filter"].copy()
    base = data["base"].copy()
    coef = data["weather_coef"].copy()
    priority = data["priority"].copy()

    hour = now.hour
    month = now.month
    season = season_from_month(month)
    weektype = weektype_from_date(now)
    time_group = time_group_from_hour(hour)

    # 여의도 대여소만 필터링
    yeouido_ids = set(station_filter["대여소_ID"].astype(str))
    cur = bike_df[bike_df["대여소_ID"].astype(str).isin(yeouido_ids)].copy()
    cur = cur.merge(station_filter, on="대여소_ID", how="left")
    cur["대여소명"] = cur["대여소명"].fillna(cur.get("대여소명_API", ""))

    # 기본예상수요: 현재 시간 조건과 정확히 맞는 값 우선
    mask = (
        (base["시간대"].astype(int) == hour)
        & (base["평일주말"].astype(str) == weektype)
        & (base["월"].astype(int) == month)
        & (base["계절"].astype(str) == season)
    )
    base_now = base.loc[mask, ["대여소_ID", "기본예상대여수요", "기본예상반납수요", "평균총이용건수", "평균순유출량"]].copy()

    # 혹시 현재 조건이 비어 있으면 시간대그룹 기준 평균으로 fallback
    if base_now.empty:
        base_now = (
            base[base["시간대그룹"].astype(str) == time_group]
            .groupby("대여소_ID", as_index=False)[["기본예상대여수요", "기본예상반납수요", "평균총이용건수", "평균순유출량"]]
            .mean()
        )

    # 날씨보정계수
    wc = coef[(coef["시간대그룹"].astype(str) == time_group) & (coef["기상조건"].astype(str) == weather_cond)]
    if len(wc) == 0:
        rent_coef, return_coef = 1.0, 1.0
        coef_note = "현재 기상조건과 정확히 일치하는 보정계수가 없어 1.0 적용"
    else:
        rent_coef = float(wc.iloc[0]["대여수요_날씨보정계수"])
        return_coef = float(wc.iloc[0]["반납수요_날씨보정계수"])
        coef_note = "현재 기상조건 보정계수 적용"

    out = cur.merge(base_now, on="대여소_ID", how="left")
    out = out.merge(priority[["대여소_ID", "우선순위기초점수", "평소이용빈도", "평균예측불균형"]], on="대여소_ID", how="left")

    for c in ["기본예상대여수요", "기본예상반납수요", "우선순위기초점수"]:
        out[c] = out[c].fillna(0)

    out["대여수요보정계수"] = rent_coef
    out["반납수요보정계수"] = return_coef
    out["예측대여수요"] = out["기본예상대여수요"] * rent_coef
    out["예측반납수요"] = out["기본예상반납수요"] * return_coef
    out["예측순유출량"] = out["예측대여수요"] - out["예측반납수요"]
    out["예상재고"] = out["현재자전거수"] + out["예측반납수요"] - out["예측대여수요"]
    out["부족기준"] = L * out["거치대수"]
    out["과잉기준"] = U * out["거치대수"]
    out["수거필요량"] = np.maximum(0, out["예상재고"] - out["과잉기준"])
    out["재배치필요량"] = np.maximum(0, out["부족기준"] - out["예상재고"])

    # 실제 자전거 대수이므로 반올림 처리. 시연에서는 1대 미만 수요는 제외하기 위해 ceil 사용.
    out["수거필요량"] = np.ceil(out["수거필요량"]).astype(int)
    out["재배치필요량"] = np.ceil(out["재배치필요량"]).astype(int)
    out["총불균형"] = out["수거필요량"] + out["재배치필요량"]
    out["후보우선점수"] = out["총불균형"] * (1 + out["우선순위기초점수"].fillna(0))
    out.attrs["context"] = {
        "hour": hour,
        "month": month,
        "season": season,
        "weektype": weektype,
        "time_group": time_group,
        "weather_cond": weather_cond,
        "rent_coef": rent_coef,
        "return_coef": return_coef,
        "coef_note": coef_note,
    }
    return out


@dataclass
class Step:
    vehicle: int
    order: int
    station_id: str
    station_name: str
    action: str
    qty: int
    load_after: int
    lat: float
    lon: float


def greedy_routes(candidates: pd.DataFrame, start_lat: float, start_lon: float, vehicle_count: int, capacity: int):
    """간단 휴리스틱: 가까운 수거 후보에서 싣고, 가까운 재배치 후보에 배치."""
    work = candidates.copy()
    work["remaining_pickup"] = work["수거필요량"].astype(int)
    work["remaining_dropoff"] = work["재배치필요량"].astype(int)
    steps: list[Step] = []

    for k in range(1, vehicle_count + 1):
        lat, lon = start_lat, start_lon
        load = 0
        order = 1
        safety = 0
        while safety < 80:
            safety += 1
            did = False

            # 1) 적재공간이 있으면 가까운 수거 후보 방문
            pickups = work[work["remaining_pickup"] > 0].copy()
            if load < capacity and not pickups.empty:
                pickups["dist"] = pickups.apply(lambda r: haversine_m(lat, lon, r["위도"], r["경도"]), axis=1)
                # 거리와 우선점수를 함께 고려
                pickups["score"] = pickups["dist"] / (1 + pickups["후보우선점수"].fillna(0))
                r = pickups.sort_values("score").iloc[0]
                qty = int(min(r["remaining_pickup"], capacity - load))
                if qty > 0:
                    idx = r.name
                    work.loc[idx, "remaining_pickup"] -= qty
                    load += qty
                    lat, lon = float(r["위도"]), float(r["경도"])
                    steps.append(Step(k, order, str(r["대여소_ID"]), str(r["대여소명"]), "수거", qty, load, lat, lon))
                    order += 1
                    did = True

            # 2) 싣고 있는 자전거가 있으면 가까운 재배치 후보 방문
            dropoffs = work[work["remaining_dropoff"] > 0].copy()
            if load > 0 and not dropoffs.empty:
                dropoffs["dist"] = dropoffs.apply(lambda r: haversine_m(lat, lon, r["위도"], r["경도"]), axis=1)
                dropoffs["score"] = dropoffs["dist"] / (1 + dropoffs["후보우선점수"].fillna(0))
                r = dropoffs.sort_values("score").iloc[0]
                qty = int(min(r["remaining_dropoff"], load))
                if qty > 0:
                    idx = r.name
                    work.loc[idx, "remaining_dropoff"] -= qty
                    load -= qty
                    lat, lon = float(r["위도"]), float(r["경도"])
                    steps.append(Step(k, order, str(r["대여소_ID"]), str(r["대여소명"]), "배치", qty, load, lat, lon))
                    order += 1
                    did = True

            if not did:
                break
            if work["remaining_pickup"].sum() <= 0 and (work["remaining_dropoff"].sum() <= 0 or load == 0):
                break

    route_df = pd.DataFrame([s.__dict__ for s in steps])
    work["처리후_남은수거"] = work["remaining_pickup"].astype(int)
    work["처리후_남은재배치"] = work["remaining_dropoff"].astype(int)
    return route_df, work


def build_map(candidates: pd.DataFrame, routes: pd.DataFrame, start_lat: float, start_lon: float):
    center = [start_lat, start_lon]
    if not candidates.empty:
        center = [float(candidates["위도"].mean()), float(candidates["경도"].mean())]
    m = folium.Map(location=center, zoom_start=14, tiles="CartoDB positron")
    folium.Marker([start_lat, start_lon], popup="출발지", tooltip="출발지", icon=folium.Icon(color="black", icon="home")).add_to(m)

    # 후보 마커
    for _, r in candidates.iterrows():
        if r["수거필요량"] > 0:
            color, label = "red", f"수거 {int(r['수거필요량'])}대"
        elif r["재배치필요량"] > 0:
            color, label = "blue", f"배치 {int(r['재배치필요량'])}대"
        else:
            color, label = "gray", "후보"
        html = f"""
        <b>{r['대여소명']}</b><br>
        ID: {r['대여소_ID']}<br>
        현재: {r['현재자전거수']}대 / 거치대 {r['거치대수']}개<br>
        예상재고: {r['예상재고']:.1f}<br>
        {label}
        """
        folium.CircleMarker(
            [r["위도"], r["경도"]], radius=7, color=color, fill=True, fill_opacity=0.75,
            popup=folium.Popup(html, max_width=320), tooltip=f"{r['대여소명']} · {label}"
        ).add_to(m)

    # 차량별 경로선
    colors = ["green", "purple", "orange", "darkred", "cadetblue", "darkgreen"]
    if not routes.empty:
        for i, (veh, grp) in enumerate(routes.groupby("vehicle")):
            pts = [[start_lat, start_lon]] + grp.sort_values("order")[["lat", "lon"]].values.tolist()
            folium.PolyLine(pts, color=colors[i % len(colors)], weight=4, opacity=0.8, tooltip=f"차량 {veh} 경로").add_to(m)
            for _, s in grp.iterrows():
                folium.Marker(
                    [s["lat"], s["lon"]],
                    tooltip=f"차량 {s['vehicle']} - {s['order']}. {s['action']} {s['qty']}대",
                    icon=folium.DivIcon(html=f"<div style='font-size:12px;background:white;border:1px solid #444;border-radius:10px;padding:2px'>{int(s['vehicle'])}-{int(s['order'])}</div>")
                ).add_to(m)
    return m


# -----------------------------
# 화면 구성
# -----------------------------
st.title("🚲 여의도 따릉이 실시간 수거·재배치 경로 추천 데모")
st.caption("실시간 API + 과거 수요모델 + 휴리스틱 경로 추천으로 수거·재배치 후보와 차량별 경로를 시각화합니다. 배포용 버전이므로 Gurobi는 포함하지 않습니다.")

data = load_csv_data()
thresholds = parse_thresholds(data["thresholds"])

with st.sidebar:
    st.header("⚙️ 실행 설정")
    bike_key_default = get_secret_or_empty("SEOUL_BIKE_API_KEY")
    city_key_default = get_secret_or_empty("SEOUL_CITYDATA_API_KEY")
    bike_key = st.text_input("따릉이 API 키", value=bike_key_default, type="password")
    city_key = st.text_input("도시데이터 API 키", value=city_key_default, type="password")

    places = data["places"].copy()
    default_idx = 0
    if "AREA_NM" in places.columns:
        names = places["AREA_NM"].dropna().astype(str).tolist()
        for cand in ["여의도", "여의도한강공원", "국회의사당", "더현대서울"]:
            if cand in names:
                default_idx = names.index(cand)
                break
        area_name = st.selectbox("도시데이터 장소명", names, index=default_idx if names else 0)
    else:
        area_name = st.text_input("도시데이터 장소명", value="여의도")

    st.divider()
    vehicle_count = st.slider("차량 수", 1, 5, 2)
    capacity = st.number_input("차량 1대 적재 용량", min_value=1, max_value=50, value=15, step=1)
    L = st.slider("부족 기준 거치율 L", 0.0, 1.0, 0.30, 0.05)
    U = st.slider("과잉 기준 거치율 U", 0.0, 1.0, 0.80, 0.05)
    pickup_n = st.slider("수거 후보 수", 1, 20, 6)
    dropoff_n = st.slider("재배치 후보 수", 1, 20, 6)

    st.divider()
    st.markdown("**출발지 좌표**")
    preset = st.selectbox("출발지 프리셋", ["여의도역", "국회의사당역", "여의나루역", "직접 입력"])
    presets = {
        "여의도역": (37.5216, 126.9243),
        "국회의사당역": (37.5281, 126.9178),
        "여의나루역": (37.5271, 126.9329),
    }
    if preset == "직접 입력":
        start_lat = st.number_input("출발지 위도", value=37.5216, format="%.6f")
        start_lon = st.number_input("출발지 경도", value=126.9243, format="%.6f")
    else:
        start_lat, start_lon = presets[preset]
        st.caption(f"{preset}: {start_lat:.6f}, {start_lon:.6f}")

    run = st.button("실시간 경로 추천 실행", type="primary", use_container_width=True)

# 설명 박스
with st.expander("이 대시보드가 하는 일", expanded=False):
    st.markdown(
        """
        1. 따릉이 실시간 API에서 현재 자전거 수, 거치대 수, 대여소 좌표를 가져옵니다.  
        2. 서울시 도시데이터 API에서 현재 기온, 습도, 풍속, 강수량을 가져옵니다.  
        3. 과거 데이터로 만든 기본예상수요와 날씨 보정계수를 결합합니다.  
        4. 예상재고를 계산하고, 부족/과잉 기준에 따라 수거·재배치 후보를 선정합니다.  
        5. Greedy 휴리스틱으로 차량별 경로를 추천하고, 지도와 표에 보여줍니다.  
        
        ※ 배포 환경 안정성을 위해 이 버전에는 Gurobi가 들어가지 않습니다. Gurobi 기반 MILP는 동일한 후보 입력값을 사용하는 별도 로컬 구현 단계로 분리합니다.
        """
    )

if not run:
    st.info("왼쪽 사이드바 설정을 확인한 뒤 **실시간 경로 추천 실행** 버튼을 눌러줘.")
    st.subheader("📁 현재 포함된 경량 데이터")
    c1, c2, c3 = st.columns(3)
    c1.metric("여의도 대여소 수", f"{len(data['station_filter']):,}개")
    c2.metric("기본예상수요 행 수", f"{len(data['base']):,}행")
    c3.metric("기상조건 보정계수", f"{len(data['weather_coef']):,}개")
    st.dataframe(data["station_filter"].head(20), use_container_width=True)
    st.stop()

try:
    with st.spinner("실시간 따릉이 API 호출 중..."):
        bike_df = fetch_bike_api(bike_key)
    with st.spinner("실시간 도시데이터 API 호출 중..."):
        city_weather = fetch_city_weather(city_key, area_name)
except Exception as e:
    st.error("API 호출 중 오류가 발생했어. API 키, 호출 제한, 장소명을 확인해줘.")
    st.exception(e)
    st.stop()

now = datetime.now(KST)
weather_class = classify_weather(
    temp=to_float(city_weather["TEMP"], 0),
    humidity=to_float(city_weather["HUMIDITY"], 0),
    wind_spd=to_float(city_weather["WIND_SPD"], 0),
    rain=to_float(city_weather["PRECIPITATION"], 0),
    thresholds=thresholds,
)

candidates_all = make_realtime_candidates(
    bike_df=bike_df,
    data=data,
    weather_cond=weather_class["기상조건"],
    now=now,
    L=L,
    U=U,
)
context = candidates_all.attrs.get("context", {})

pickup_df = candidates_all[candidates_all["수거필요량"] > 0].sort_values("후보우선점수", ascending=False).head(pickup_n)
dropoff_df = candidates_all[candidates_all["재배치필요량"] > 0].sort_values("후보우선점수", ascending=False).head(dropoff_n)
selected = pd.concat([pickup_df, dropoff_df], ignore_index=True)

if selected.empty:
    st.warning("현재 설정 기준에서는 수거·재배치 후보가 없습니다. 부족/과잉 기준 L, U를 조정해봐.")
    st.stop()

routes, after_df = greedy_routes(selected, start_lat, start_lon, vehicle_count, capacity)

before_imbalance = int(selected["수거필요량"].sum() + selected["재배치필요량"].sum())
after_imbalance = int(after_df["처리후_남은수거"].sum() + after_df["처리후_남은재배치"].sum())
processed = before_imbalance - after_imbalance
improve = 0 if before_imbalance == 0 else processed / before_imbalance * 100

# -----------------------------
# 결과 출력
# -----------------------------
st.subheader("① 현재 조건")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("현재 시점", now.strftime("%m/%d %H:%M"))
col2.metric("시간대그룹", context.get("time_group", "-"))
col3.metric("평일/주말", context.get("weektype", "-"))
col4.metric("계절", context.get("season", "-"))
col5.metric("기상조건", weather_class["기상조건"])

w1, w2, w3, w4, w5 = st.columns(5)
w1.metric("기온", f"{city_weather['TEMP']}℃")
w2.metric("습도", f"{city_weather['HUMIDITY']}%")
w3.metric("풍속", f"{city_weather['WIND_SPD']}m/s")
w4.metric("강수량", f"{city_weather['PRECIPITATION']}mm")
w5.metric("날씨 업데이트", str(city_weather.get("WEATHER_TIME", "-")))

st.subheader("② 수거·재배치 후보 및 개선 효과")
m1, m2, m3, m4 = st.columns(4)
m1.metric("최적화 전 후보 불균형", f"{before_imbalance:,}대")
m2.metric("휴리스틱 처리량", f"{processed:,}대")
m3.metric("처리 후 남은 불균형", f"{after_imbalance:,}대")
m4.metric("개선율", f"{improve:.1f}%")

left, right = st.columns(2)
with left:
    st.markdown("#### 🔴 수거 후보")
    show_cols = ["대여소_ID", "대여소명", "현재자전거수", "거치대수", "예상재고", "수거필요량", "후보우선점수"]
    st.dataframe(pickup_df[show_cols], use_container_width=True, hide_index=True)
with right:
    st.markdown("#### 🔵 재배치 후보")
    show_cols = ["대여소_ID", "대여소명", "현재자전거수", "거치대수", "예상재고", "재배치필요량", "후보우선점수"]
    st.dataframe(dropoff_df[show_cols], use_container_width=True, hide_index=True)

st.subheader("③ 차량별 추천 경로 지도")
route_map = build_map(selected, routes, start_lat, start_lon)
st_folium(route_map, width=None, height=620)

st.subheader("④ 차량별 방문 순서")
if routes.empty:
    st.warning("경로가 생성되지 않았습니다. 후보 수나 L/U 기준을 조정해봐.")
else:
    st.dataframe(routes, use_container_width=True, hide_index=True)

st.subheader("⑤ 처리 후 남은 불균형")
after_show = after_df[["대여소_ID", "대여소명", "수거필요량", "재배치필요량", "처리후_남은수거", "처리후_남은재배치"]]
st.dataframe(after_show, use_container_width=True, hide_index=True)

# 선택적 다운로드. 화면 표시가 핵심이고, 파일 저장은 보조 기능.
with st.expander("결과 파일로 저장, 선택사항", expanded=False):
    csv_buf = io.StringIO()
    routes.to_csv(csv_buf, index=False, encoding="utf-8-sig")
    st.download_button("차량 경로 CSV 다운로드", csv_buf.getvalue().encode("utf-8-sig"), "route_result.csv", "text/csv")

    html = route_map.get_root().render()
    st.download_button("지도 HTML 다운로드", html.encode("utf-8"), "route_map.html", "text/html")
