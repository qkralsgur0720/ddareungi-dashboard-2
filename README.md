# 여의도 따릉이 수거·재배치 경로 추천 대시보드

Streamlit Cloud 배포용 경량 버전입니다.  
Gurobi는 포함하지 않고, 실시간 API와 과거 수요모델을 결합한 후보 산출 및 Greedy 휴리스틱 경로 추천을 보여줍니다.

## GitHub 업로드 구조

레포지토리 최상단에 아래 파일들이 바로 보여야 합니다.

```text
app.py
requirements.txt
packages.txt
README.md
.gitignore
.streamlit/
data/
```

## Streamlit Secrets

Streamlit Cloud의 App settings > Secrets에 아래 값을 넣어주세요.

```toml
SEOUL_BIKE_API_KEY = "따릉이 API 키"
SEOUL_CITYDATA_API_KEY = "서울시 도시데이터 API 키"
```

## 배포 설정

- Main file path: `app.py`
- Python packages: `requirements.txt` 사용

## 주의

이 배포용 버전은 Gurobi 최적화가 아니라 휴리스틱 경로 추천입니다.  
보고서에서 제시한 Gurobi 기반 MILP 모델은 동일한 후보 입력값을 이용해 로컬 환경에서 별도 실행할 수 있는 최종 최적화 모형으로 설명하면 됩니다.
