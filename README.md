# Water — 공개 원전 냉각수 머신러닝

EIA-923 공개자료에서 미국 원자력 발전소의 월별 냉각수 기록을 수집하고, **취수·배수 온도차(°C)**를 회귀 예측합니다. 같은 입력과 데이터 분할로 Random Forest, Gradient Boosting, CatBoost를 비교합니다.

| 모델 | 실행 장치 | 역할 |
|---|---|---|
| 학습자료 평균 온도차 | CPU | 기준선 |
| Ridge | CPU | 선형 기준선 |
| RandomForestRegressor | CPU 멀티코어 | Random Forest |
| GradientBoostingRegressor | CPU | 일반 Gradient Boosting |
| CatBoostRegressor | NVIDIA GPU 또는 CPU | GPU를 지원하는 Gradient Boosting |

**`--device GPU`는 CatBoost에 적용됩니다. Random Forest와 scikit-learn Gradient Boosting은 CPU로 실행합니다.** GPU가 있는 Windows PC에서도 모든 모델을 비교할 수 있습니다. 데이터가 작아 GPU가 CPU보다 빠르다고 보장하지 않습니다. GPU 실패 시 자동으로 CPU로 바꾸지 않습니다.

## Windows GPU PC에서 시작

필수: Windows 10/11 x64, **Python 3.12 64비트와 Python Launcher(`py`)**, NVIDIA GPU와 드라이버. CatBoost의 공식 Windows/Linux 휠에 GPU 지원이 포함되어 있어 이 구성에 별도 PyTorch/CUDA Toolkit 설치는 필요하지 않습니다. 설치와 데이터 수집에는 인터넷이 필요하며 EIA API 키는 필요하지 않습니다.

```powershell
git clone https://github.com/maredevitas-bot/Water.git
cd Water
.\01_setup.cmd
.\02_prepare.cmd
.\03_train_gpu.cmd
```

Git이 없으면 GitHub에서 **Code → Download ZIP**을 선택해 압축을 해제한 뒤 같은 이름의 파일을 순서대로 더블클릭합니다.

1. `01_setup.cmd`: 이 폴더에 `.venv` 생성, 버전 고정 라이브러리 설치, 실제 GPU 소규모 연산 점검.
2. `02_prepare.cmd`: 2021~2024년 EIA 원본 ZIP 다운로드, 단위 변환·품질 검사·데이터 분할. 첫 실행은 수 분 걸릴 수 있습니다.
3. `03_train_gpu.cmd`: 5종 모델 학습·평가·저장. CatBoost에만 GPU 사용.
4. 선택적으로 `04_verify.cmd`: 자동 검사와 20개 트리 학습 점검. 결과는 `_smoke`로 구분됩니다.

원본 데이터와 학습 결과는 Git에 포함하지 않습니다. **새로 받은 저장소에서는 `02_prepare.cmd`가 필수입니다.** 실행 창이 실패를 표시하면 다음 단계로 넘어가지 말고 오류를 확인하세요. 시스템 전체 PowerShell 실행 정책은 변경하지 않습니다.

GPU가 없거나 CPU로 먼저 확인할 때:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_windows.ps1 -CpuOnly
.\02_prepare.cmd
.\03_train_cpu.cmd
```

## Linux / GPU 서버

Python 3.12와 NVIDIA 드라이버가 있는 Linux x86-64에서:

```bash
git clone https://github.com/maredevitas-bot/Water.git
cd Water
python3.12 -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: -r requirements-lock.txt
.venv/bin/python -m ripple.doctor --gpu
.venv/bin/python -m ripple.fetch
.venv/bin/python -m ripple.prepare
.venv/bin/python -m ripple.train --device GPU
```

CPU 환경에서는 doctor의 `--gpu`를 생략하고 train에 `--device CPU`를 지정합니다. Docker 안에서 실행한다면 호스트 GPU가 컨테이너에 전달되어 있어야 합니다. Linux GPU 실기 검증은 이 저장소의 CPU 자동 검사 범위에 포함되지 않습니다.

## 무엇을 학습하는가

- 출처: [EIA-923 연간 공개자료](https://www.eia.gov/electricity/data/eia923/), Schedule 8D 냉각계통 정보와 Page 1의 `NUC` 원자력 발전량.
- 한 행: 발전소 / 냉각계통 / 연도 / 월.
- 입력: 취수 수온, 달 전체 평균 취수 유량의 `log1p`, 발전소 원자력 월평균 전기출력, 냉각계통 가동시간, 월의 sin/cos, 냉각 방식, 미국 주.
- 목표: 배수 수온 − 취수 수온. 예측 온도차에 취수 수온을 더해 배수 수온을 계산합니다.
- 기본 범위: 2021~2024년, 관류식 냉각계통. 범위와 모델 설정은 `config.json`에 있습니다.

같은 달의 발전량과 취수량이 주어진 **월별 조건부 예측**입니다. 다음 달 예보나 실시간 원전 제어 모델이 아닙니다. 공개자료에는 바이패스 비율, 추가 냉각 전력, 팬 제어, 감발 원인의 실험 정답이 없으므로 이를 임의로 생성해 학습하지 않습니다.

발전량은 발전소 전체 원자력 합계이며 냉각계통별 배분량이 아닙니다. 취수 유량은 월누적 백만 US gallons를 m³로 바꾸고 달 전체 초 수로 나눈 값으로 내부 순환량과 다릅니다. 수온은 화씨에서 섭씨로 변환하며 결측 수온을 0으로 채우지 않습니다. 실측/추정 방법과 원본 행 번호도 보존합니다.

취수 -3~50°C, 배수 -3~70°C, 온도차 -5~40°C 범위 및 양의 발전량·취수량·가동시간을 확인합니다. 이는 자료 검토용 범위입니다. 제외 사유는 `data/processed/nuclear_cooling_all.csv`, 집계는 `reports/quality_report.json`에 남습니다. 부적합 행을 삭제해 숨기지 않습니다.

## 평가와 재현성

| 분할 | 기본 기간과 대상 |
|---|---|
| train | 2021~2022년, 학습 대상 발전소 |
| validation | 2023년, 학습에 등장했던 발전소 |
| test_temporal | 2024년, 학습에 등장했던 발전소 |
| test_unseen_plants | 2024년, 미리 분리한 발전소 |
| unused | 미관측 발전소의 과거 기록 등 |

발전소 ID의 해시와 고정 시드로 약 20%의 발전소를 분리합니다. 발전소 ID·배수 수온·온도차는 입력에 사용하지 않습니다. 전처리는 학습자료에만 적합하며, CatBoost 조기 종료와 최종 모델 선택에는 validation만 사용합니다. 일반 Gradient Boosting과 Random Forest의 트리 수는 설정으로 고정합니다. 테스트 점수로 모델을 선택하지 않습니다.

원본 ZIP의 URL·다운로드 시각·SHA-256은 `data/raw/manifest.json`에 기록합니다. 이미 받은 ZIP은 자동 갱신하지 않으며 해시가 바뀌면 중단합니다. `config.json`을 변경한 경우 `python -m ripple.prepare`를 다시 실행해야 합니다. 2021~2024년 밖의 연도는 원본 스키마와 결측률을 별도로 확인해야 합니다.

`requirements.txt`는 주요 의존성, `requirements-lock.txt`는 간접 의존성을 포함한 고정 버전입니다. GPU 계산에는 같은 시드에서도 작은 비결정성이 생길 수 있습니다.

## 결과와 예측

실행마다 `runs/<UTC시각>_full`에 새 폴더를 만듭니다. 점검 실행은 `_smoke`입니다.

| 결과 파일 | 내용 |
|---|---|
| `metrics.csv` | 모델·분할별 MAE, RMSE, 온도차 R², 편향, 발전소별 MAE 평균 |
| `predictions.csv` | 실제·예측 배수 수온과 예측 온도차 |
| `run.json` | 설정, 데이터 해시, 버전, 모델별 장치, 선택 모델, 학습 입력 범위 |
| `random_forest.joblib` | 전처리를 포함한 Random Forest |
| `gradient_boosting.joblib` | 전처리를 포함한 Gradient Boosting |
| `ridge.joblib`, `catboost.cbm` | 나머지 학습 모델 |
| `feature_importance.csv` | CatBoost 중요도. 인과적 효과를 뜻하지 않음 |

오차 단위는 °C이며 R²는 **온도차** 기준입니다. 같은 취수 수온을 더하므로 온도차와 배수 수온의 예측오차는 같습니다. 최종 모델은 validation RMSE가 가장 작은 모델이며 기준선이 선택될 수도 있습니다.

본 학습 후 PowerShell에서 가장 최근 결과로 예측:

```powershell
$waterRun = Get-ChildItem .\runs -Directory -Filter '*_full' | Sort-Object Name | Select-Object -Last 1
if (-not $waterRun) { throw '먼저 본 학습을 실행하세요.' }
$waterOutput = Join-Path $waterRun.FullName ('example_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.csv')
.\.venv\Scripts\python.exe -m ripple.predict --run $waterRun.FullName --input .\example_inputs.csv --output $waterOutput
```

`example_inputs.csv`는 전처리에서 생성한 실제 자료 5행의 입력 예시입니다. 외부 CSV도 같은 열이 필요합니다. `log1p_withdrawal_m3_s = ln(1 + 취수유량[m³/s])`, `month_sin = sin(2π×월/12)`, `month_cos = cos(2π×월/12)`를 사용합니다. `outside_training_domain`은 학습 범위 이탈 표시이고, False가 정확도를 보증하지는 않습니다. 저장 모델은 본인이 생성하거나 신뢰하는 파일만 불러오세요.

## 자동 점검

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m ripple.train --device CPU --smoke
```

GitHub Actions는 Windows/Linux에서 설치와 오프라인 파이프라인 검사를 수행합니다. 테스트의 합성 데이터는 프로그램 점검 전용이며 EIA 관측자료나 연구 결과로 사용하지 않습니다. 실제 GPU 학습 성공은 해당 PC의 `ripple.doctor --gpu` 및 본 학습 결과로 확인해야 합니다.

실제 자료로 확인한 초기 CPU 실행 결과와 한계는 [초기 검증 기록](benchmarks/cpu_initial/README.md)에 있습니다. 미관측 발전소 평가에서는 학습 모델들이 평균값 기준선의 RMSE를 개선하지 못했습니다.

공식 참고: [CatBoost GPU 설치](https://catboost.ai/docs/en/concepts/python-installation), [GPU 학습](https://catboost.ai/docs/en/features/training-on-gpu), [scikit-learn 앙상블](https://scikit-learn.org/stable/modules/ensemble.html).
