# VGGT와 MASt3R-SLAM의 Pointmap 생성 품질 비교 실험 계획서

> 문서 상태: **사전 실험 설계 v0.1** (2026-11-05)  
> 대상 저장소: `seaeast2/gradresearch_vggt_mast3r_slam`  
> 범위: **3D 기하(Pointmap) 품질의 객관적 비교**. 3DGS 품질 평가는 후속 연구로 분리한다.  
> 주의: 아래 숫자·기준 중 '초기 제안'으로 표시된 값은 논문 결과가 아니라 실험 설계 변수이다. 실제 실행 결과는 아직 없다.

## 1. 연구 배경과 목표

VGGT(Visual Geometry Grounded Transformer)는 여러 장의 이미지를 공동 입력받아 Pointmap, Depthmap 및 카메라 파라미터를 예측한다. MASt3R-SLAM은 MASt3R의 2-view Pointmap 예측을 사용하여 추적·키프레임별 융합·전역 최적화를 수행하는 순차적인 Dense SLAM 시스템이다.

본 실험의 목표는 **같은 실제 장면의 RGB 영상**에서 두 시스템이 생성하는 3D 좌표의 정확성, 공간적 일관성, 이상점(outlier), 표면 커버리지와 비용을 정량·정성 비교하는 것이다. 궁극적으로 '이상 Gaussian 감소를 위한 기하 정합성 기반 최적화' 연구에서 어느 3D 초기 기하를 사용할지 판단하는 기초 실험으로 사용한다.

### 1.1 연구 질문

- **RQ1 — 단일 프레임 기하 정확도:** GT 깊이가 존재하는 동일 픽셀에서 어느 모델의 3D 위치/깊이 예측 오차가 작은가?
- **RQ2 — 전역 기하 품질:** 예측된 카메라 위치를 포함해 하나의 좌표계로 놓았을 때 어느 출력의 Accuracy, Completeness, F-score가 우수한가?
- **RQ3 — 이상점:** 자유공간 침범이나 GT 표면으로부터 멀리 떨어진 점이 얼마나 발생하는가? 신뢰도 필터링으로 얼마나 줄일 수 있는가?
- **RQ4 — 프레임 수와 연속성:** 입력 프레임 수 및 영상 길이가 늘어날 때 각각 어떻게 변하는가?
- **RQ5 — 비용:** 추론 시간·VRAM·완료율 대비 유효한 3D 점의 비율은 어떠한가?

### 1.2 검증할 가설(결과가 아님)

- **H1:** VGGT의 Depth+Camera 역투영 Pointmap은 VGGT 직접 예측 Pointmap과 품질 차이를 보일 것이다.
- **H2:** MASt3R-SLAM의 **융합 후** Pointmap은 원시 예측과 다른 outlier/coverage 양상을 보일 것이다.
- **H3:** 두 시스템의 Confidence로 같은 **점 유지율(retention)** 을 맞춰도 품질 차이가 남을 수 있다.

가설이 기각되어도 그대로 보고한다. 어느 모델이 무조건 우수하다는 전제는 두지 않는다.

## 2. 비교 대상을 정확히 정의

Pointmap이라고 하더라도 **출력 단계가 다르면 공정하게 같은 종류의 결과라고 간주할 수 없다.** 다음 대상을 명시적으로 구분한다.

| ID | 실험 대상 | 추출 내용 | 주 용도 |
|---|---|---|---|
| **V-P** | VGGT 직접 Point Head | `world_points`, `world_points_conf` | 순수 Pointmap Head 예측 평가 |
| **V-D** | VGGT Depth + Camera 역투영 | `depth`, `depth_conf`, `extrinsic`, `intrinsic`에서 생성한 world points | VGGT의 다른 3D 생성 경로 비교 |
| **M-F** | MASt3R-SLAM **융합 후** 키프레임 Pointmap | `X_canon`, `T_WC`, `get_average_conf()` | 완성된 SLAM의 지도 품질 평가 |
| **M-R** (선택) | MASt3R의 pairwise 원시 예측 | 추적·융합 전 각 프레임의 Pointmap 및 Confidence | '모델 예측 자체'를 비교하는 통제 실험 |

**주 비교:** V-P, V-D, M-F.  
**해석상 주의:** V-P/V-D는 VGGT의 feed-forward 출력이고 M-F는 SLAM의 누적·융합·최적화 결과이다. 따라서 M-F와 V-P의 차이를 '신경망 자체의 우열'로 해석하지 않는다. 순수 예측 능력을 논하려면 M-R을 구현해 별도 실험으로 평가한다.

VGGT는 직접 Pointmap Head 출력과 Depth를 카메라 파라미터로 역투영한 Pointmap을 구별하여 기록한다. VGGT 공식 코드도 기본 시각화에서 Depth 기반 점을 사용하고, `--use_point_map`을 지정하면 직접 Pointmap을 사용한다.

MASt3R-SLAM 공식 구현의 `mast3r_slam/frame.py`는 `X_canon`과 `T_WC`를 보유하고, `mast3r_slam/evaluate.py`의 `save_reconstruction()`은 `T_WC.act(X_canon)`을 PLY로 저장한다. 현재 기본 PLY는 Confidence로 필터링된 점만 포함하므로, **픽셀별 평가용으로 필터링 이전의 텐서와 픽셀 대응 정보도 저장하는 export 기능을 추가**해야 한다.

## 3. 데이터셋과 장면 선정

### 3.1 우선순위

1. **1차: TUM RGB-D** — RGB, Depth GT, 카메라 궤적 GT, 타임스탬프가 있어 픽셀 단위/전역 평가를 함께 수행할 수 있다.
2. **2차: 다른 실내 장면**(예: 7-Scenes, 적절한 GT·라이선스·동기화가 확인된 시퀀스) — 장면 종류가 바뀌어도 결과가 유지되는지 확인.
3. **3차 선택: ETH3D 등 3D GT가 존재하는 장면** — 높은 품질의 GT가 준비되었을 때 전역 표면 평가 확대.
4. **보조: 자체 촬영 RGB 영상** — 시각적 사례 및 실패 유형 분석만 수행. 신뢰 가능한 GT 없이 3D 정확도 수치의 근거로 쓰지 않는다.

### 3.2 1차 파일럿 장면(제안)

- TUM `freiburg1_desk`: 책상·물체·가림.
- TUM `freiburg1_room`: 실내 공간과 다양한 깊이.
- 별도 TUM 시퀀스 1~2개: 이동 경로/텍스처가 다른 장면을 추가 선정.

**최종 시퀀스 이름, 사용 가능한 GT 품질, 라이선스·데이터 접근 여부는 최초 데이터 수집 시 확정한다.** 개발용 1~2개와 최종 평가용 장면을 분리하여 임계값을 시험 장면에 맞춰 조정하지 않는다.

### 3.3 시간 동기화 및 입력 정의

- RGB–Depth–Pose GT를 타임스탬프로 매칭한다. 매칭 허용 오차와 누락률을 로그에 기록한다.
- 장면마다 **정확히 같은 원본 프레임 ID 목록**을 추출하여 두 시스템의 입력·평가용 표본을 관리한다.
- 프레임 수 조건: **4 / 8 / 16 / 32** (초기 제안). 너무 큰 baseline으로 MASt3R-SLAM 추적이 실패하면 별도 실패로 보고한다.
- 동일 원본을 사용하되, **각 모델의 공식 전처리(크기, 종횡비, 패딩)는 유지**한다. 네트워크 입력 해상도가 동일하다고 가정하거나 임의로 맞추지 않는다.
- 모든 픽셀별 지표는 최종적으로 **원본 이미지 좌표**에 대응시킨 뒤 계산한다. resize/pad/crop 역매핑, (K) 변환, 유효 마스크를 검사한다.

### 3.4 두 개의 프레임 운영 프로토콜(혼동 금지)

**P1 — 입력 예산 통제(주 실험):** VGGT에는 정해진 N개 이미지를 공동 입력한다. MASt3R-SLAM에도 **동일한 N개 이미지**를 시간순으로 공급하고, 추적 성공/실패를 기록한다. MASt3R-SLAM은 자동 키프레임 선택을 수행하므로, 출력 가능한 키프레임의 원본 ID를 반드시 함께 저장한다. 픽셀별 비교는 동일 프레임 ID에서 양쪽 출력이 실제 존재할 때만 한다. 별도로 해당 조건의 **출력 프레임 커버리지**를 보고한다.

**P2 — 실제 사용 조건(보조 실험):** MASt3R-SLAM은 연속 영상 전체를 입력받고 VGGT는 고정 간격으로 선택한 N개 프레임을 처리한다. 이는 **동일 입력량 비교가 아니므로** 정확도 표에서 P1과 섞지 않고 시스템 운용 실험으로만 해석한다.

큰 프레임 간격에서 P1 추적이 실패할 수 있으므로, 초기에는 연속 4/8/16/32프레임으로 smoke test를 수행하고 확장 실험에서 시간 간격을 늘린다. **실패한 실행을 성공한 실행만으로 대체하지 않는다.**

## 4. 환경 및 재현성 고정

- Linux/Ubuntu + NVIDIA CUDA 지원 GPU를 우선 사용한다.
- 각 모델은 가능하면 **분리된 Python 환경**으로 설치하여 PyTorch/CUDA 의존성 충돌을 방지한다.
- VGGT: 공식 `facebookresearch/vggt`, 공식 체크포인트를 사용. 모델 버전(`VGGT-1B` 등)을 기록한다.
- MASt3R-SLAM: 공식 `rmurai0610/MASt3R-SLAM`, README의 모델·retrieval 체크포인트를 사용한다.
- 사전학습 체크포인트 해시(SHA256), 소스 Git commit SHA, Python/PyTorch/CUDA/cuDNN 버전, GPU 모델·VRAM, 모델 설정·랜덤시드를 저장한다.
- mixed precision(BF16/FP16), TF32, deterministic 설정, batch/frame 수를 함께 기록한다.
- GPU 워밍업 후 추론 시간을 측정하되, **모델 로딩/체크포인트 다운로드, 전처리, 추론, 후처리, export**를 분리하여 측정한다.
- OOM/시간 초과/추적 실패/유효 점 부족도 **실패율**로 보고한다. 시험마다 필요한 최대 VRAM을 기록한다.
- GPU/CUDA 장치 확인, CUDA smoke test 및 실제 추론은 **샌드박스 밖의 호스트 환경**에서 수행한다. 샌드박스에서 GPU가 보이지 않는 결과를 호스트 GPU 부재로 해석하지 않는다. 각 확인·실행의 `execution_context`(예: `host` 또는 `sandbox`), GPU/driver/CUDA 정보를 manifest와 로그에 기록한다.

### 4.1 카메라 보정 조건

**C0 — 미보정(주 비교):** 두 시스템 모두 GT 카메라 내부 파라미터를 **모델 입력으로 사용하지 않는다.** GT는 평가에만 사용한다.  
**C1 — 보정 이용(보조):** MASt3R-SLAM의 `--calib` 사용 결과는 별도 행으로 제시한다. VGGT가 원래 외부 calibration을 같은 방식으로 입력받는 모델이 아니므로 C1을 대칭 비교처럼 해석하지 않는다.

렌즈 왜곡이 큰 장면은 undistortion 적용 여부, 원본/교정 영상 공간, GT Depth 및 (K)의 기준 좌표계를 명확히 통일한다.

## 5. 중간 결과 표준 포맷

### 5.1 실행 단계 코드와 결과물 격리

모든 실제 실행 단계에는 순서대로 `E0`, `E1`, `E2`처럼 영문자 `E`와 숫자로 된 고유 코드를 부여한다. 아래 §8의 단계 코드는 계획·실행 로그·폴더·보고서에서 같은 이름으로 사용한다. 한 번 부여한 코드는 해당 단계의 목적을 식별하는 이름으로 유지하며, 새 단계를 추가하면 사용하지 않은 다음 번호를 쓴다.

각 실험 실행(run)은 고유한 `<run_id>`를 갖고, 각 단계의 산출물·로그·설정·상태 기록은 그 실행 폴더 안의 해당 단계 폴더에만 저장한다. 다른 단계의 파일을 덮어쓰거나 한 폴더에 합치지 않는다. 앞 단계 산출물을 입력으로 쓰는 경우 원본은 그대로 두고, 입력 경로와 SHA-256을 현재 단계의 manifest에 기록한다. 재실행이나 실패 후 수정 실행은 새 `<run_id>`를 만들어 기존 결과를 보존한다.

재현에 필요한 실행 스크립트·설정·패치는 저장소의 `experiments/`, `scripts/`, `docs/`에 두고 Git으로 추적한다. 내려받은 데이터·체크포인트·upstream checkout·실행 결과는 저장소 루트 `data/`, `checkpoints/`, `thirdparty/`, `results/`에 두며 `.gitignore`로 제외한다. `scripts/pointmap_comparison/download_e0_resources.sh`는 E0 리소스를 공개 출처에서 내려받고 SHA-256을 검증한다.

```text
results/pointmap_comparison/<run_id>/
  run_manifest.json             # 실행 전체의 목적, 단계 목록, 시작 커밋
  stages/
    E0/
      manifest.json             # 상태, 명령, 입력 경로/hash, 환경, 산출물/hash
      config/                    # 이 단계에서 실제 사용한 설정 사본
      artifacts/                 # 이 단계가 생성한 결과만 저장
      logs/                      # 이 단계의 stdout/stderr 및 실행 로그
    E0a/                         # 선행 진단 단계도 독립 산출물 폴더 사용
      manifest.json
      config/
      artifacts/
      logs/
    E1/
      manifest.json
      config/
      artifacts/
      logs/
    ...
```

`<run_id>`는 데이터셋/시퀀스와 실행 시각 또는 고유 식별자를 포함해 충돌 없이 정한다. 각 단계 manifest에는 `stage_code`, 단계명, 상태(`planned`/`running`/`completed`/`failed`), 시작·종료 시각, 입력 및 해시, 설정, Git/checkpoint SHA, HW/SW·실행 환경, 산출물 상대 경로와 SHA-256, 실패 사유를 남긴다. 결과 요약표나 보고서도 생성 단계의 `artifacts/`에 저장하고, 여러 단계 결과를 묶은 최종 요약은 별도 후속 단계 폴더에 둔다.

각 단계의 `artifacts/` 안에서 데이터셋·프레임 ID·타임스탬프를 키로 다음을 기록한다. 아래는 한 단계가 생성하는 결과의 예이며, 실제 파일은 반드시 해당 `stages/<stage_code>/artifacts/` 아래에 둔다.

```text
stages/<stage_code>/artifacts/
  <dataset>/<sequence>/<protocol>/<model_variant>/
    frames.csv             # 원본 frame_id, timestamp, 입력/출력 성공, 이미지 변환
    cameras.npz            # 프레임별 camera pose (원래 예측 좌표계)
    pointmaps.npz          # 프레임별 [H,W,3], valid_mask, conf, RGB 인덱스
    points_raw.ply         # 기본 필터 이전 3D 시각화(선택)
    points_filtered.ply    # 선택한 임계값 적용 결과
    timing.json            # 전처리/추론/후처리/peak VRAM
    metrics.json           # 지표, 분모, 결측/실패 원인
    figures/               # RGB/GT/예측/오차맵, 동일 시점 point cloud
```

저장 데이터에서 **원본과 변환 후의 좌표계 관계**를 반드시 추적 가능하게 한다.

- 좌표계 convention: OpenCV 카메라((x) 오른쪽, (y) 아래, (z) 앞), World-to-Camera와 Camera-to-World를 명시적으로 구분.
- `T_WC`와 `T_CW`의 의미 및 행렬 방향을 export에서 검증. 좌표 변환은 점 3개와 카메라 원점의 round-trip 테스트로 확인.
- MASt3R-SLAM의 `T_WC`는 Sim(3)이며 스케일을 포함할 수 있다. 구현에서 **스케일을 포함한 실제 변환**으로 월드 좌표에 배치해야 한다.
- `X_canon`을 바로 VGGT의 월드 Pointmap과 비교하지 않는다.
- Pointmap이 world 프레임인지 local camera 프레임인지 각 파일의 메타데이터에 넣는다.
- 어떤 필터를 적용하기 전 원시 마스크·Confidence·원시 점을 저장한다. PLY만 보유하면 per-pixel 평가를 복원하기 어렵다.
- GT 위치로 추정된 점을 직접 교체/보정하지 않는다.

## 6. 비교 평가 지표

지표는 한 개의 숫자로 합치지 않고 **기하, 전역 포즈, 이상점, 속도**로 분리한다.

### 6.1 픽셀별 깊이 오차(Depth GT 필수)

모델 예측 3D 점을 해당 **예측 카메라 좌표계**로 변환하여 (hat z(u,v))를 얻고 GT 깊이 (z^*(u,v))와 비교한다. VGGT Depth Head 값은 별도로 평가한다.

- **AbsRel:** (mathrm{mean}(|hat z-z^*|/z^*))
- **RMSE:** (sqrt{mathrm{mean}((hat z-z^*)^2)})
- **MedAE:** (mathrm{median}(|hat z-z^*|))
- **유효 픽셀 비율:** GT 유효 픽셀 중 예측값을 낸 비율.

단안 재구성의 scale ambiguity 때문에 **원래 스케일(raw)** 과 **시퀀스 단위 scale 정렬 후** 결과를 분리한다. 스케일은 개발용/정렬용 prefix에서만 추정해 평가 구간에 고정 적용한다. 각 프레임마다 GT로 따로 스케일을 보정해 성능을 과대평가하지 않는다.

### 6.2 픽셀별 3D 오차(Primary metric)

GT 깊이와 GT 카메라 (K^*,T^*)로 **동일 픽셀의 GT 3D 점** (P^*(u,v))를 만든다. 예측점 (hat P(u,v))과 동일 좌표계에서 평가한다.

[
e_{mathrm{3D}}(u,v)=left|,Sigl(hat P(u,v)igr)-P^*(u,v),ight|_2
]

- 평균/중앙값/90·95 percentile 거리 오차(m)
- 임계 거리 내 점 비율((2/5/10)cm, **초기 제안**; GT 센서 정확도에 맞춰 사전 확정)
- valid GT pixel 수, predicted valid pixel 수를 항상 함께 기록.
- **로컬 기하**와 **전역 지도 품질**을 따로 보고한다.
  - 로컬: 각 모델의 추정 카메라 좌표에서 점을 표현한 후 **시퀀스 전체에 동일한 스케일**을 사용하여 비교(자체 Pose 오차 영향 최소화).
  - 전역: 하나의 고정된 Sim(3) 정렬 후 GT world 좌표와 비교(Pose/scale 오차 포함).

### 6.3 전역 3D 지도 품질

모든 평가 대상의 Pointmap을 하나의 world 좌표계로 변환한 뒤, 관측 가능한 GT 표면과 비교한다.

- **Accuracy (pred→GT):** 각 예측점에서 GT 표면까지 최근접 거리.
- **Completeness (GT→pred):** 각 관측 GT 점에서 예측점까지 최근접 거리.
- **F-score@τ:** 정밀도(Precision)와 재현율(Recall) 조화평균. τ는 2/5/10cm에서 사전 확정.
- **Coverage@τ / Density:** 관측 GT 표면이 얼마나 잘 복원되는지와 단위면적당 점 밀도.

**중요:** RGB-D로 생성한 GT cloud는 '관측된 표면의 샘플'이지 전체 장면의 완전한 mesh가 아니다. GT에 관측되지 않은 부분을 즉시 예측 outlier로 간주하지 않는다. 공통 관측 frustum 및 가시성/깊이 마스크를 적용하여 평가 영역을 정의한다. 밀도가 높은 예측이 유리해지지 않도록 모든 모델에 동일한 voxel/downsample 정책을 적용하고, raw/정규화 결과를 구분한다.

### 6.4 이상점과 자유공간 오류(논문 주제 관련 핵심)

- **Surface outlier ratio@τ:** *GT가 관측한 영역 안에서* GT 표면과의 거리가 τ보다 먼 예측점 비율.
- **Free-space violation:** GT depth ray 상에서 확실한 관측 표면 **앞쪽의 빈 공간**에 존재하는 예측점 비율(센서 불확실성 허용 폭 적용).
- **Floaters 시각화:** 카메라 바깥쪽/공중에 떠 있는 고립 점, 중복 표면, 물체 가장자리의 찢어짐 등 사례를 기록.
- **점 유지율 vs 정확도:** Confidence 기준으로 상위 100/80/60/40%를 남겼을 때 지표 변화를 기록.

**주의:** raw confidence의 절대적인 숫자는 모델 간 동일한 확률/의미를 갖지 않을 수 있다. 같은 confidence threshold 숫자를 일괄 적용하지 않고 **percentile 또는 동일한 retained point count**에서 비교한다. 필터가 outlier만 줄인 것이 아니라 유효한 표면까지 삭제했는지 Completeness/Coverage와 함께 제시한다.

### 6.5 보조 지표

- **ATE / RPE:** 카메라 궤적 GT가 있을 때 지도 오차 해석을 위한 보조 지표. Geometry 점수와 혼합하지 않는다.
- **Cross-view depth/reprojection consistency:** 서로 다른 시점의 관측이 일치하는 정도(occlusion 검사 포함). GT 없이 비교할 수 있지만 정확도의 대체 지표는 아니다.
- **런타임·메모리:** 전체 시간, 프레임당 시간, FPS(실제 시스템 경로에서만), peak VRAM, 실패율.

## 7. 정렬·마스킹 및 공정성 원칙

1. **같은 GT 좌표계:** 원본 RGB의 (u,v), GT depth와 GT pose의 픽셀/시간 정렬을 먼저 검증한다.
2. **시퀀스 단일 Sim(3) 정렬:** 단안 모델의 미정 스케일을 고려하여 한 시퀀스에 동일한 (S)를 적용한다. 원칙적으로 **사전 고정된 정렬용 prefix 카메라 중심들**로 Umeyama를 추정하고 나머지 프레임에서 평가한다.
3. 카메라 중심이 거의 한 점이거나 일직선이어서 Sim(3)가 불안정하면 실패 원인을 기록하고, 미리 정한 대체 규칙(더 긴 정렬 prefix 또는 평가 불가 처리)을 적용한다. 전체 GT 점구름으로 모델별 정렬을 마음대로 조정하지 않는다.
4. **가림과 동적 영역:** GT invalid, occlusion, 반사·유리, 움직이는 사람은 정책을 명시한다. 가능하면 '전체 유효 GT'와 '정적/고신뢰 영역' 결과를 나란히 보고한다.
5. **기본 필터 vs 무필터:** 원시 점 비교를 주 분석에 포함하고, 공식 기본 후처리/Confidence 필터는 별도 분석한다.
6. **동일 평가 프레임:** 두 시스템의 유효 출력이 중복되는 subset을 사용하되, 누락 프레임 수와 완전한 실행의 성공률을 반드시 보고한다.
7. **하이퍼파라미터 분리:** threshold, frame sampling, voxel size, GT 불확실성 margin 등을 개발 장면에서만 정하고 최종 장면에는 고정한다.
8. **BA 혼입 방지:** VGGT의 BA는 선택 옵션이므로 기본 비교에서는 사용하지 않는다. `V-D+BA`가 필요하면 **별도 ablation**으로 정의한다. MASt3R-SLAM backend global optimization은 본래 시스템에 포함되므로 해제 시 별도 variant로 표기한다.
9. **시각화 공정성:** 같은 ROI, 카메라 시점, 축, 색 범례, 점 크기 및 동일 point budget으로 비교한다.

## 8. 실행 단계별 계획과 완료 기준

### E0 — 환경 준비·데이터 확인

- [x] 두 공식 저장소와 모델 체크포인트를 분리 설치한다.
- [x] TUM RGB-D 1개 시퀀스 다운로드 및 RGB–Depth–GT Pose 동기화.
- [x] 두 모델의 기본 데모 실행 및 파일 저장 확인.
- [x] 모든 실행의 commit/checkpoint hash 기록.

**완료 기준:** 같은 RGB 프레임 목록에서 두 모델 실행이 가능하고 중간 결과를 저장할 수 있다.

**E0 실행 기록 (2026-11-05):** run `tum-freiburg1_desk-20261105-001`, `execution_context=host`. CUDA/GPU probe, smoke test, 두 모델 추론은 샌드박스 밖 호스트에서 수행했다(RTX 5060, 8,151 MiB, driver 595.99.02, CUDA 13.2; PyTorch `2.14.1+cu132`, Python 3.11.18). 동일한 TUM `freiburg1_desk` RGB 4장(timestamp `1305031453.359684`–`1305031453.459685`)을 두 실행기에 전달했다. 원본은 RGB 613장, depth 595장, GT pose 2,335개이며, 각 RGB 시각에 depth와 pose를 독립 최근접 매칭하고 허용 오차 20 ms를 적용했을 때 596장이 유효했다. VGGT는 4장 입력의 Pointmap/Depth·confidence 및 camera 예측을 NPZ로 저장했다. MASt3R-SLAM 데모는 정상 종료하고 PLY와 trajectory를 저장했지만, 자동 키프레임 선택으로 4장 중 1장만 PLY에 기여했다. 따라서 E0 완료는 **같은 입력 목록에서 두 파이프라인을 실행하고 중간 결과를 저장할 수 있음**을 뜻하며, 프레임별 MASt3R 출력 완전성을 뜻하지 않는다. 체크포인트 SHA256, 소스 commit, 설정·패치, 전체 로그와 출력 목록은 `results/pointmap_comparison/tum-freiburg1_desk-20261105-001/stages/E0/manifest.json` 및 `artifacts/e0_summary.json`에 기록했다.

### E0a — E1 선행: MASt3R-SLAM pointmap 추출·추적성 진단

E1의 표준 export 구현에 앞서 upstream 런타임의 fused keyframe pointmap을 원본 RGB 프레임 및 픽셀 좌표와 안전하게 연결할 수 있는지 확인한다. 별도 run/stage에서 같은 E0 4장 입력을 사용하고, upstream checkout은 수정하지 않은 채 run-local launcher hook을 backend 종료 뒤 실행해 추출한다.

- [x] E0 입력 프레임 목록의 timestamp/frame index와 MASt3R `SharedKeyframes.frame_id` 연결을 검증한다.
- [x] keyframe별 `X_canon`, 평균 confidence, RGB, `T_WC`를 원시 해상도와 이미지 좌표 변환 정보를 포함해 NPZ로 저장한다.
- [x] `T_WC.act(X_canon)`의 world pointmap을 함께 저장하고 Sim(3) 역변환 round-trip 오차를 측정한다.
- [x] representative RGB pixel에서 frame ID, (u,v), local/world 3D, confidence, 색상 조회를 검증한다.
- [x] 입력 4장 대비 추출 keyframe 수와 누락 frame ID를 기록하여 자동 keyframe 선택의 coverage를 정량화한다.

**완료 기준:** 하나 이상의 fused keyframe pointmap에서 원본 프레임 timestamp와 픽셀→3D 값을 추적할 수 있고, local↔world round-trip이 수치 허용오차 내에서 통과한다. 누락 입력 프레임은 실패로 숨기지 않고 별도 기록한다. 이 진단 결과로 E1 export 인터페이스와 공통 프레임 정책을 고정한다.

**E0a 실행 결과 (2026-11-05):** run `tum-freiburg1_desk-20261105-005`, `execution_context=host` (RTX 5060, 8,151 MiB, driver 595.99.02, CUDA 13.2; Python 3.11.18, PyTorch `2.14.1+cu132`). E0와 동일한 4장 입력으로 MASt3R-SLAM을 실행하고, backend 최종 대기열을 비운 뒤 upstream checkout을 수정하지 않는 stage-local overlay에서 fused keyframe `X_canon`, `C/N`, RGB, `T_WC`, `T_WC.act(X_canon)`, 융합/갱신 횟수를 NPZ에 저장했다. 결과 pointmap은 384×512이며, 입력 4장 중 frame 0(원본 TUM frame 17, timestamp `1305031453.359684`)만 keyframe으로 선택되어 coverage는 1/4, 나머지 frame ID는 1, 2, 3이다. 해당 keyframe의 중심 pixel `(u=256,v=192)`에서 local/world 3D, confidence `7.6587`, RGB를 조회했고 resize 기준 원본 좌표는 `(320.125,240.125)`였다. 실제 선택 pose는 초기 identity였으므로 실제 local/world 최대 차이는 0이다. 별도 비항등 Sim(3) (scale 1.35, Z 회전 17°, translation `[0.25,-0.4,1.2]`) 수치 round-trip 최대 오차는 `4.44e-16`이었다. E1에서도 네 장 실행은 identity pose 하나만 제공했으므로 실제 비항등 궤적 pose 검증은 여러 keyframe이 생기는 후속 실행에 남긴다. E1은 추출된 keyframe subset과 누락 frame을 명시하며, 이 4장 실행만으로 프레임 전체 coverage를 주장하지 않는다. 세 번의 launcher 초기화 실패는 별도 `failed` run으로 보존했다. 상세 manifest·검증 JSON·스크립트는 `results/pointmap_comparison/tum-freiburg1_desk-20261105-005/stages/E0a/` 및 `experiments/pointmap_comparison/E0a/`에 있다.

### E0b — E1 구현 전 결정·수용량 확인

§13의 미결 항목을 고정하기 위해 E0 개발 시퀀스·핀된 소스·체크포인트를 재사용한다. upstream checkout과 기존 단계 출력은 수정하지 않는다. GPU 측정은 샌드박스에서 CUDA가 보이지 않으므로 반드시 호스트에서 실행한다. 재현 스크립트는 `experiments/pointmap_comparison/E0b/`, 결과·로그·측정 manifest는 고유 run의 `stages/E0b/` 아래에 저장한다.

- [x] 비교 모델·체크포인트: 공식 `facebook/VGGT-1B`; 소스 commit `a288dd0f14786c93483e45524328726ab7b1b4ce`, checkpoint SHA-256 `d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0`. 원 체크포인트는 비상업 연구용 조건으로만 사용한다. 상업 사용이 필요한 경우 별도 상업 checkpoint와 약관으로 새 실행을 정의한다.
- [x] MASt3R 추출 지점: `SharedKeyframes`의 fused keyframe `X_canon`, `get_average_conf()`, `uimg`, `T_WC`, `N/N_updates`; backend 대기열 종료 후 stage-local export hook에서 저장한다. upstream 소스는 수정하지 않는다. 이는 fused M-F subset이며 모든 입력 프레임의 pointmap이 아니다. E0a에서 선택률 1/4로 확인했다. 원시 pairwise M-R은 별도 ablation으로 남긴다.
- [x] 입력 격자/역매핑: TUM 원본 RGB 640×480. VGGT `mode=pad` 입력은 518×518이며 원본은 518×392로 bicubic 축소하고 위·아래 63픽셀씩 흰색 패딩한다. 픽셀 중심 기준 원본 역변환은 `u=(u_net-pad_left+0.5)*640/518-0.5`, `v=(v_net-pad_top+0.5)*480/392-0.5`이다. MASt3R-SLAM 입력/pointmap은 512×384, crop 변화 없이 0.8 배율 축소되므로 `u=(u_net+0.5)*640/512-0.5`, `v=(v_net+0.5)*480/384-0.5`로 되돌린다. 각 export는 원본 크기, net 크기, padding/crop, pixel convention을 메타데이터에 기록한다.
- [x] Sim(3) 정렬 prefix 및 퇴화 판정: 깊이·GT pose가 모두 RGB와 20 ms 이내 연결된 프레임의 앞 20%(`ceil(0.20N)`, 최소 60개)를 정렬 전용으로 고정하고 이후 프레임은 평가에만 사용한다. prefix camera center 최대 변위가 0.5 m 미만이거나 중심 공분산 고유값 `lambda2/lambda1 < 0.01`이면 궤적을 퇴화로 판정하고 사전 정의된 더 긴 prefix로 확장한다. 확장 후에도 실패하면 해당 run의 global Sim(3) 비교는 평가 불가로 기록한다. 개발 장면에서 596개 유효 프레임 중 120개 prefix는 변위 1.273 m, `lambda2/lambda1=0.0706`으로 기준을 통과했다. 30개는 0.358 m라 최소 길이 60의 근거로 삼았다.
- [x] 개발 장면 기준 평가 상수: GT depth는 uint16 PNG 값을 5000으로 나눠 미터로 바꾸고, 0.30–5.00 m만 유효로 둔다. 주 point-error/F-score 및 surface-outlier 임계값 `tau=0.05 m`; 자유공간 검사에서 GT 표면 불확실성 margin `0.02 m`; 모든 모델 공통 voxel size `0.02 m`. `0.02/0.10 m` 임계값 민감도도 함께 보고하고, 유효 GT 픽셀 수를 항상 기록한다. 이 설정은 freiburg1_desk 개발 장면에서 고정하며 평가 장면에 맞춰 재조정하지 않는다.
- [x] GPU와 최대 입력 수: 호스트 RTX 5060 물리 VRAM 8,151 MiB (CUDA reported allocatable 7.53 GiB), driver 595.99.02, CUDA 13.2; PyTorch `2.14.1+cu132`, bfloat16, 입력 격자 518²에서 N=1/2/4/5는 완료, N=6은 OOM. allocator peak(allocated/reserved)는 N=1 5.35/5.98 GiB, N=2 5.52/6.65 GiB, N=4 6.31/7.07 GiB, N=5 6.71/7.22 GiB였다. N=5는 추론 후 여유가 약 149 MiB뿐이므로 E1의 안전 최대 N은 **4**로 제한한다. 더 큰 N은 해상도/메모리 정책 변경을 별도 사전 실행으로 검증하기 전에는 사용하지 않는다.
- [x] 데이터·모델 라이선스: TUM RGB-D benchmark 페이지는 데이터에 CC BY 4.0을 명시하며 데이터 사용 논문 인용을 요청한다. 프레임/깊이/pose는 이 저장소나 원격 Git에 복제하지 않고 opt-in 다운로드로 받는다. VGGT-1B 원 체크포인트는 비상업 사용으로 제한한다. MASt3R-SLAM/MASt3R 코드·checkpoint에는 CC BY-NC-SA 4.0 및 학습 데이터별 추가 약관이 적용될 수 있으므로 상업 사용·모델 재배포를 금지하고, 개별 checkpoint notice와 학습 데이터 약관을 함께 확인한다. MASt3R retrieval mirror 자산은 출처·SHA를 남기되 Git에는 올리지 않는다.

**E0b 실행 결과 (2026-11-05):** 성공 run `tum-freiburg1_desk-20261105-010`, `execution_context=host`. 소스 기하·좌표 매핑 확인 및 VGGT N별 CUDA 추론은 `results/pointmap_comparison/tum-freiburg1_desk-20261105-010/stages/E0b/`에 저장했다. 앞선 분석 스크립트 경로/CSV 필드 가정 오류 세 run `...-006`, `...-007`, `...-009`는 각자 `failed` manifest와 로그를 보존했고, 성공 결과에는 포함하지 않았다. 실행 스크립트·고정값과 완료 결과 manifest를 함께 보존한다.

### E1 — 표준 export 모듈 구현

- [x] `export_vggt.py`: V-P, V-D, depth/confidence, camera, frame ID와 이미지 좌표 변환 정보를 NPZ/PLY로 저장.
- [x] `export_mast3r_slam.py` 및 run-local final hook: M-F의 `X_canon`, `T_WC`, confidence 및 키프레임 원본 ID를 저장.
- [x] 이번 E1 범위는 계획의 주 비교인 M-F로 고정하고, 선택 variant M-R은 수행하지 않기로 결정했다. M-R은 필요 시 별도 실험으로 추가한다.
- [x] coordinate/unit/RGB pixel map round-trip 검증기를 실행한다.

**완료 기준:** RGB 픽셀 한 개를 역추적하여 두 출력에서 해당 3D 위치와 Confidence를 조회할 수 있다.

**E1 실행 결과 (2026-11-05):** 성공 run `tum-freiburg1_desk-20261105-014`, `execution_context=host` (RTX 5060, PyTorch `2.14.1+cu132`, CUDA 13.2, bfloat16). VGGT는 공유 입력 4장 전부에 대해 V-P/V-D, depth와 confidence, extrinsic/intrinsic, 원본 frame/timestamp, source image size, 518×518 input grid, 63px 상하 padding map을 NPZ로 저장하고 V-P/V-D binary PLY를 프레임별 생성했다. peak allocated/reserved VRAM은 6,775,736,832/7,591,690,240 bytes였다. 추론은 완료했으나 CUDA allocator가 551,550,976-byte 요청의 실패 경고를 기록했다. 결과 생성 자체는 성공했으며, 8GB급 GPU에서 N=4도 여유가 작으므로 후속 추론의 VRAM 상태를 계속 기록한다. VGGT 대표 입력 frame 0의 center net pixel `(259,259)`는 source pixel `(320.118,240.112)`로 역매핑되고, 해당 위치의 V-P XYZ `[-0.001663,-0.001650,0.865488]`, V-D XYZ `[0.000137,-0.000033,0.861579]`, V-P confidence `4.0636`, depth `0.861663`, depth confidence `7.2568`을 기록했다.

MASt3R-SLAM은 동일한 4 RGB 프레임을 입력받아 M-F keyframe 1개를 export했다(frame ID 0, 원본 TUM frame 17, timestamp `1305031453.359684`, 384×512; fusion/update count 모두 4). frame ID 1–3은 선택되지 않아 누락으로 기록했다. NPZ에는 local/world pointmap, confidence, RGB, `T_WC`, source mapping과 validity mask를 저장하고 binary PLY도 생성했다. 검증에서 VGGT와 MASt3R의 실제 resize RGB 최대 차이는 각각 0, 두 모델의 pixel-center map round-trip 최대 오차는 0 px, 비항등 합성 Sim(3) local↔world round-trip 최대 오차는 0으로 통과했다. VGGT와 M-F 대표 픽셀의 source 좌표 차이도 0.0147 px로 대응 기준 0.02 px 이내였다. 대표 MASt3R pixel `(256,192)`의 local/world point는 `[0.000785,-0.002158,0.953714]`, confidence는 `7.6587`이었다. VGGT 및 M-F 대표 픽셀의 3D·confidence·RGB 값과 frame/timestamp 대응은 `e1_export_validation_paired_pixels.json`에 보존했다. 실제 trajectory pose는 identity keyframe 하나뿐이므로 비항등 실제 pose 검증은 여러 keyframe이 생성되는 후속 실행에서 수행한다. E1 run `...-011`은 retrieval checkpoint 경로, `...-012`는 validation JSON 직렬화 문제로 실패해 별도 run에 보존했다. 기존 검증으로 완료된 중간 run `...-013`도 보존했고, 최종 보강 검증은 run `...-014`에서 완료했다. 성공 export·validation·hash manifest는 `results/pointmap_comparison/tum-freiburg1_desk-20261105-014/stages/E1/`에 있고 재현 코드는 `experiments/pointmap_comparison/E1/`에 있다.

### E2 — GT 기반 정량 평가 파이프라인

- [x] `associate_rgb_depth_pose.py`: 유효 GT pixel과 타임스탬프 매칭 결과 저장.
- [x] `align_sim3.py`: 사전 고정한 alignment prefix와 변환 로깅.
- [x] `evaluate_pixel.py`: Depth/3D per-pixel 오류 및 valid fraction 계산.
- [x] `evaluate_cloud.py`: Accuracy/Completeness/F-score, common visibility 영역.
- [x] `evaluate_outliers.py`: surface outlier/free-space violation/retention curve.

**완료 기준:** 합성 3D 점/카메라의 정답이 알려진 단위 테스트에서 정렬과 오차를 정확하게 회수한다.

**E2 실행 결과 (2026-11-05):** run `tum-freiburg1_desk-E2-20261105-015`, `execution_context=host` (CPU 평가; 신경망 추론 없음). 기존 E1의 4개 RGB frame 17–20 export와 E0 원본 데이터를 입력으로 사용했다. 독립 재연결에서 RGB 613장 중 depth/pose 타임스탬프가 각각 20 ms 이내에 모두 매칭된 항목은 596장이었고, 기존 E0 association과 일치했다. 합성 검증은 임의의 비항등 Sim(3) 회수 최대 행렬 오차 `1.99e-11`, 카메라 중심 RMSE `1.87e-10 m`, 픽셀 역변환 오차 `2.84e-14 px`, depth/3D 정확 회수 0 오차로 통과했으며 퇴화 선형 궤적을 거부했다. 고정한 최소 60 프레임 prefix보다 짧아 VGGT는 4장, M-F는 1장 모두 sequence-global Sim(3)을 계산하지 않았다.

픽셀 평가에서 V-P 4/4, V-D 4/4, M-F 1/4 frame을 평가했다. frame macro AbsRel은 각각 `0.2185`, `0.2189`, `0.1698`; depth RMSE는 `0.2811`, `0.2806`, `0.2373 m`; camera-local 3D median error는 `0.2528`, `0.2504`, `0.1569 m`였다. 이 M-F 값은 선택된 한 keyframe만의 결과로 VGGT 전체와 동일 표본 비교가 아니다. 공통 GT 관측 표면에서 frame별 5 cm F-score는 V-P `0.0366–0.0458`, V-D `0.0388–0.0500`, M-F `0.1144`였고, 이는 global 정합 정확도가 아니라 per-frame camera-coordinate 점군 점수다. GT 유효 pixel의 약 99.98–100%는 각 출력 마스크와 겹쳤다.

V-P/V-D 점 차이 중앙값은 `4.37 mm`였으며, 그 크기와 각 출력의 GT 3D 오차 간 pixel-level Spearman rho는 `0.540`/`0.531`이었다(총 1,016,716 대응 pixel). 이는 이 4장 desk pilot에서의 기술적 상관이며, 보정된 불확실도나 다른 장면으로 일반화되는 관계를 뜻하지 않는다. Confidence 상위 40% 유지 시 V-P/V-D 3D 중앙 오차는 감소했지만, 이 역시 프레임별 탐색 결과다. MASt3R fusion count는 관측값 4 하나뿐이라 상관을 추정하지 않았다. 결과, CSV, 로그, 실행 설정과 manifest는 `results/pointmap_comparison/tum-freiburg1_desk-E2-20261105-015/stages/E2/`에 있고, 재현 코드는 `experiments/pointmap_comparison/E2/`에 있다.

### E3 — 파일럿 및 실험 반복

- [x] 1개 장면, 4/8/16/32프레임 입력 예산 실험(P1).
- [x] 다른 장면에서 동일 조건 재현.
- [x] 대표 조건 3회 실행 및 deterministic 여부 표기.
- [x] C0 미보정 주 비교와 C1 보정 보조 실험 분리.
- [x] Confidence 상위 100/80/60/40% 유지에 대한 sensitivity 분석.
- [x] 긴 연속 영상 운용 실험(P2) 별도 수행; 성공/실패와 장치 한계를 기록.

**완료 기준:** 성공/실패를 포함한 모든 조합에서 자동 생성된 CSV/JSON 비교표를 얻는다.

**E3 실행 결과 (2026-11-05):** run `tum-freiburg1_desk-E3-20261105-016`, `execution_context=host`, RTX 5060 8,151 MiB. desk는 E0 입력을 다시 동기화해 RGB 613/depth 595/pose 2,335 중 596개 유효 항목을 확인했다. 두 번째 장면 `freiburg1_room`은 TUM 공식 아카이브(782,381,450 bytes, SHA256 `5ace47a1d2e53696bc939a84999293a04a7226958e848a609666691fd3cc38da`)로 재현했고 RGB 1,362/depth 1,360/pose 4,887 중 1,362개 유효 항목이었다. 각 P1 목록은 timestamp 순서의 첫 N개 유효 RGB로 고정했다.

P1 전체 4/8/16/32 조합에서 두 장면의 MASt3R-SLAM은 모두 종료에 성공했다. VGGT는 desk와 room N=4에서 모두 성공했으며, N=8/16/32 여섯 조건은 모두 실제 CUDA OOM으로 실패했다. N=8 로그는 DPT depth head에서 1.03 GiB 추가 할당 실패를 보인다. 따라서 VGGT의 이 GPU·518² 설정에서는 비교 가능한 N이 4뿐이며 N>4를 타일 입력 등으로 대체하지 않았다. MASt3R-SLAM은 N=4/8/16에서 keyframe 각 1개, N=32에서 각 2개를 출력했다. 입력 대비 keyframe coverage는 25%, 12.5%, 6.25%, 6.25%였다. 이는 SLAM 출력 subset의 coverage이며, 누락 프레임의 dense map을 뜻하지 않는다.

두 장면 N=4에서만 직접 대응 가능한 C0 픽셀 평가가 가능했다. desk의 VGGT V-P/V-D AbsRel은 `0.2185/0.2189`, depth RMSE `0.2811/0.2806 m`, camera-local 3D median `0.2528/0.2504 m`였다. desk M-F(1/4 keyframe)는 각각 `0.1698`, `0.2373 m`, `0.1569 m`였다. room에서는 V-P/V-D AbsRel `0.2687/0.2687`, depth RMSE `0.4380/0.4362 m`, 3D median `0.2899/0.2924 m`; M-F(1/4 keyframe)는 `0.0760`, `0.2305 m`, `0.0564 m`였다. 공통 관측 표면의 frame macro F-score@5cm은 desk V-P/V-D/M-F `0.0417/0.0443/0.1144`, room `0.0228/0.0258/0.5614`였다. 이 M-F 수치는 각 장면에서 한 keyframe에 한정되며 sequence/global 정합 점수로 해석하지 않는다. GT intrinsics는 TUM RGB 등록 깊이 평가에만 적용했고, global Sim(3)은 사전 고정한 60-frame 최소 prefix를 채우지 못해 수행하지 않았다.

C0 주 비교와 분리한 MASt3R C1 보정 실행은 desk/room N=4 모두 성공했다. 두 모델에 대칭적으로 calibration을 공급할 수 없으므로 C1은 보조 실행이다. P1 각 장면에서 대표 조건을 3회 실행했을 때 desk/room VGGT N=4와 MASt3R N=32의 export NPZ SHA256은 각각 세 번 모두 동일했다. 기존 E1 desk N=4 VGGT 및 M-F 파일과 E3 반복본도 byte-identical이었다. 이 조건에서는 반복 간 출력 변동이 관찰되지 않았다.

모든 MASt3R P1 예산에 대해 M-F의 상위 confidence 100/80/60/40% 유지 곡선을 계산했다. 상위 40% retained point의 5cm outlier 비율은 desk N=4/8/16/32에서 `1.000/1.000/1.000/0.868`, room에서 `0.212/0.377/0.457/0.731`이었다. 유지율은 실제 출력 keyframe 내의 point를 기준으로 하므로 전체 입력 프레임 coverage와 별개다. room P2의 전체 1,362개 연속 프레임 MASt3R 실행은 keyframe buffer 사전 할당에서 OOM(추가 3.99 GiB 요청, 가용 399 MiB)으로 종료했다. 같은 시퀀스에서 균등 간격 VGGT 4장 운용은 25.2초에 완료했다. P2 두 결과는 입력량이 달라 P1 정확도와 혼합하지 않는다.

Condition별 성공/실패, latency, 출력 수/coverage, VGGT peak allocated/reserved, MASt3R 100 ms sampled VRAM peak, 정량 지표 및 manifest 경로가 있는 CSV/JSON은 `results/pointmap_comparison/tum-freiburg1_desk-E3-20261105-016/stages/E3/artifacts/summary/`에 있다. 아카이브 hash·모델 hash·모든 condition manifest는 stage `manifest.json`에 기록했다. 재현 코드는 `experiments/pointmap_comparison/E3/`에 있다.

### E4 — 사례 분석과 연구 연결

- [x] 이상점이 많은 사례 5개 기록.
- [x] 책상 모서리, 가림, 무늬가 적은 벽, 어두운 display 표면, 긴 이동 궤적을 분류.
- [x] VGGT의 `||V-P - V-D||`와 GT error `>5cm` label의 관계를 ROI 및 장면별로 측정.
- [x] MASt3R-SLAM의 confidence 및 fusion 횟수와 이상점 간 관계를 측정하고 표본 부족을 표시.
- [x] confidence 점 제거별 outlier 비율 변화와 coverage 손실을 함께 측정.

**완료 기준:** 수치·오차맵·동일 시점 pointcloud 사례가 모두 있는 사전 실험 보고서 작성.

**E4 실행 결과 (2026-11-05):** run `tum-freiburg1_desk-E4-20261105-019`, `execution_context=host` (CPU 분석). 기하학적 outlier label은 RGB-등록 GT의 유효 깊이 `0.30–5.0 m`와 비교한 per-pixel camera-local 3D 오차 `>0.05 m`로 고정했다. 장면 유형은 육안 수동 분류이며 semantic GT가 아니다. desk edge(C01), teddy bear/monitor 가림 경계(C02), room P2 frame 454의 무늬가 적은 벽(C03), desk frame 18의 어두운 monitor display(C04), P2 끝점 frame 1361(C05)을 분석했다. C04는 반사 측정 GT가 있는 사례가 아니므로 “어두운 display 표면”으로만 라벨링했다.

desk 전체 4-frame pilot에서 V-P/V-D 불일치량과 GT 3D 오차 간 Spearman rho는 `0.540/0.531`(1,016,716 pixel), room은 `0.394/0.378`(892,477 pixel)이었다. 장면별 pixel을 합치면 `0.434/0.418`(1,909,193 pixel)이다. 그러나 texture-poor wall C03 ROI에서는 `-0.487/-0.541`로 부호가 반대였다. 따라서 이 불일치는 두 pilot의 전체 픽셀에서는 GT error와 함께 커지는 경향을 보였지만, 실패 유형별 또는 장면 일반의 단독 이상점 점수로 쓰기에는 불충분하다. pixel 수는 독립 시퀀스 수가 아니므로 p-value를 일반화 근거로 사용하지 않는다.

동일 N=4 keyframe의 MASt3R confidence와 픽셀 3D 오차 rho는 desk `-0.469`(255,142 pixel), room `-0.716`(220,491 pixel)로 관측됐다. 이는 confidence가 큰 픽셀에서 error가 작아지는 기술적 경향이며, 확률 보정이나 새 장면에서의 예측 성능을 뜻하지 않는다. N=4 fusion count는 두 장면 모두 `[4]`; N=32에서는 desk `[26,7]`, room `[24,9]`였다. N=32는 각 장면 2개 keyframe뿐이고 outlier 경향도 장면 간 일관적이지 않아 fusion-count 관계는 추정 불가로 남겼다.

Confidence 상위 40% 유지 시 C02의 M-F outlier 비율은 `0.746`에서 `0.555`로 낮아졌지만, ROI GT coverage도 `1.00`에서 `0.40`으로 줄었다. 같은 ROI의 V-P/V-D는 top-40%에도 outlier 비율이 거의 `1.00`으로 남아 개선되지 않았다. C05 장거리 끝점(frame 1361)은 누적 GT 카메라 중심 이동 `15.976 m`, 시작-끝 중심 변위 `1.866 m`이며 V-P/V-D median error `0.539/0.542 m`였다. 해당 full-frame top-40% 유지에서도 5cm outlier 비율은 사실상 1.00으로 남고 coverage는 0.40이었다.

사례별 RGB/GT depth/error/confidence/disagreement 그림 5장과 동일 camera-coordinate GT/V-P/V-D/M-F cloud 그림 5장은 `results/pointmap_comparison/tum-freiburg1_desk-E4-20261105-019/stages/E4/artifacts/figures/` 및 `same_view_clouds/`에 있다. 사례별 오차·coverage·retention CSV와 상관·fusion 표본 수·P2 경로 요약은 같은 run의 `e4_summary.json`, `case_metrics.csv`, `retention_tradeoff.csv`에 저장했다. 점군은 per-view camera frame이며 전역 정합을 수행하지 않았다. 실행 중 분석기 준비 실패 2회는 run `...-017`, `...-018`에 별도 보존하고 성공 run에서는 덮어쓰지 않았다. 재현 코드는 `experiments/pointmap_comparison/E4/`에 있다.

## 9. 보고서에 반드시 포함할 표/그림

| 결과물 | 내용 |
|---|---|
| Table A | 시퀀스별 프레임 수, GT 유효율, 처리 성공률, 결과 출력 프레임 수 |
| Table B | V-P/V-D/M-F의 Depth AbsRel·RMSE, per-pixel 3D median/P90 |
| Table C | 전역 Accuracy/Completeness/F-score@τ, common GT visibility 범위 |
| Table D | Surface outlier·Free-space violation·Coverage·point count |
| Table E | 모델별 latency, peak VRAM, FPS(해당 시), OOM/실패율 |
| Figure A | RGB, GT depth, 예측 depth, depth error heatmap |
| Figure B | 동일 가상 카메라 시점에서 중첩한 GT vs 출력 Point Cloud |
| Figure C | Confidence 상위 점 유지율별 outlier율–coverage 곡선 |
| Figure D | 프레임 수 4/8/16/32에 따른 정확도·속도·VRAM 변화 |
| Figure E | 대표적 ghost/floater 실패 사례 확대도 |

결과는 **장면별 값**과 **장면 간 요약(중앙값/분산)**을 모두 남긴다. 픽셀 수가 많은 장면이 전체 평균을 독점하지 않도록 scene-level aggregate를 별도로 제공한다.

## 10. 권장 저장소 구조(향후 구현 시)

현재 저장소에 다음 코드가 이미 있다는 의미가 아니라, 이 문서를 기반으로 구현할 때의 **제안 구조**이다.

```text
gradresearch_vggt_mast3r_slam/
├── docs/
│   └── POINTMAP_COMPARISON_EXPERIMENT_PLAN.md
├── configs/
│   ├── datasets.yaml
│   └── experiments.yaml
├── scripts/
│   ├── prepare_tum.py
│   ├── export_vggt.py
│   ├── export_mast3r_slam.py
│   └── run_experiments.py
├── evaluation/
│   ├── align_sim3.py
│   ├── evaluate_pixel.py
│   ├── evaluate_cloud.py
│   └── evaluate_outliers.py
├── tests/
│   └── test_geometry_conventions.py
├── data/                      # Git에 원본 데이터/GT를 올리지 않음
└── results/                   # Git에 대용량 결과/체크포인트를 올리지 않음
```

후속 구현 시 `.gitignore`에 원본 이미지/비디오, 데이터셋, 체크포인트, `*.ply`, `*.npz`, 대용량 결과를 제외한다. 표·설정·스크립트·작은 그림 및 요약 지표는 필요한 범위에서 버전 관리한다.

## 11. 예상 위험과 대응

| 위험 | 해석상 오류 | 대응 |
|---|---|---|
| VGGT는 공동 추론, MASt3R-SLAM은 순차 추적 | 서로 다른 작업 조건을 직접 비교 | P1/P2 분리, 같은 입력 예산과 성공률 표기 |
| MASt3R-SLAM은 키프레임만 융합 | 동일 프레임의 per-pixel 값 부족 | 원본 frame_id 저장, 중복 subset 비교, 누락률 보고 |
| 단안 scale ambiguity/Sim(3) drift | 절대 거리 오차 과대 또는 부당한 GT 정렬 | 한 시퀀스의 고정된 사전 정렬만 적용 |
| 데이터셋 Depth의 누락·노이즈 | 실제 표면을 잘못 outlier로 판정 | GT valid mask, 정적 장면, 센서 오차 margin |
| 모델별 Confidence 정의 차이 | 임의 threshold로 한쪽 모델만 유리 | 동일 retained fraction, coverage 동시 평가 |
| VGGT 긴 영상 입력의 VRAM 증가 | 제한된 frame 수만 성공 | peak VRAM, N별 OOM/실패율 측정 |
| 카메라 내참수/왜곡·resize 불일치 | Pointmap 좌표가 잘못 맞춰짐 | 전처리 역변환 단위 테스트 |
| GT가 관측하지 않은 표면 | 미관측 영역을 outlier 처리 | common visibility / GT-free-space 조건 |
| GT를 사용한 시험별 재정렬 | 수치 과대평가 | 사전 정의한 alignment prefix 고정 |

## 12. 단계적 결론 도출 기준

실험 종료 후 다음을 순서대로 판단한다.

1. **예측 품질:** V-P와 V-D 차이가 실제로 존재하는가? 이 차이가 장면을 바꿔도 유지되는가?
2. **융합 효과:** M-F는 높은 정확도/낮은 outlier를 보이는가, 아니면 누적 drift가 오히려 악화시키는가?
3. **공정한 선택:** raw geometry 정확도, 최종 지도 품질, VRAM/시간, 입력 영상 길이 중 어떤 조건에서 각각 유리한가?
4. **논문 기여 가능성:** V-P↔V-D 불일치, 다중 뷰 기하 불일치, MASt3R-SLAM Confidence 중 실제 outlier를 가장 잘 구분하는 지표는 무엇인가?
5. **후속 실험:** 선택된 초기 기하를 3DGS에 동일한 학습 조건으로 투입하여 ghost Gaussian 감소와 rendering quality(예: PSNR/SSIM/LPIPS)를 **별도 실험**으로 검증한다. Pointmap 비교 결과만으로 Gaussian 개선을 주장하지 않는다.

## 13. 구현 전 확인해야 하는 사항

- [x] 비교 대상 VGGT checkpoint/version 선택(`facebook/VGGT-1B`, source commit/checkpoint SHA 기록) 및 라이선스 확인.
- [x] MASt3R-SLAM의 실제 수정 없이 추출 가능한 fused keyframe Pointmap 단계와 export 위치 확정.
- [x] 모델별 내부 입력 크기 및 픽셀 중심 기준 원본 역매핑 확인.
- [x] GT 정렬용 prefix 최소 길이(60 valid frames; 현재 개발 장면 prefix 120)와 카메라 이동 퇴화 기준 결정.
- [x] GT 거리 허용치 `0.05 m`, outlier `tau=0.05 m`, voxel `0.02 m`, 자유공간 GT margin `0.02 m`, valid depth `0.30–5.00 m`를 개발 장면 기준으로 고정.
- [x] 호스트 GPU VRAM 확인 및 안전한 VGGT 최대 입력 `N=4` 설정(N=5는 메모리 여유 부족, N=6 OOM).
- [x] TUM 데이터, VGGT, MASt3R-SLAM 및 MASt3R checkpoint의 사용·재배포 조건 확인; 다운로드 데이터/weights는 Git에서 제외.

**§13 완료 기록:** 결정 근거와 측정은 E0b run `tum-freiburg1_desk-20261105-010`의 manifest 및 measurements에 있다. TUM RGB-D 데이터는 [공식 페이지](https://cvg.cit.tum.de/data/datasets/rgbd-dataset)에 CC BY 4.0으로 표시되어 있고, VGGT 원 체크포인트는 [공식 저장소](https://github.com/facebookresearch/vggt)의 공지대로 비상업 조건이다. MASt3R-SLAM/MASt3R는 [코드 라이선스](https://github.com/rmurai0610/MASt3R-SLAM/blob/main/LICENSE.md)와 [checkpoint notice](https://github.com/naver/mast3r/blob/main/CHECKPOINTS_NOTICE)를 함께 적용한다. 데이터·weight 바이너리는 공개 Git에 넣지 않는다.

---

## 참고 자료 및 구현 근거

1. Wang et al., **VGGT: Visual Geometry Grounded Transformer**, CVPR 2025. 논문: https://arxiv.org/abs/2503.11651
2. VGGT 공식 GitHub/README 및 `demo_viser.py`, `demo_colmap.py`: https://github.com/facebookresearch/vggt
3. Murai et al., **MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors**, CVPR 2025. 논문: https://arxiv.org/abs/2412.12392
4. MASt3R-SLAM 공식 GitHub의 `main.py`, `mast3r_slam/frame.py`, `mast3r_slam/evaluate.py`: https://github.com/rmurai0610/MASt3R-SLAM
5. Wang et al., **DUSt3R: Geometric 3D Vision Made Easy**: https://arxiv.org/abs/2312.14132
6. Leroy et al., **Grounding Image Matching in 3D with MASt3R**: https://arxiv.org/abs/2406.09756
7. VGGT official repository/checkpoint terms: https://github.com/facebookresearch/vggt and https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt
8. MASt3R-SLAM license: https://github.com/rmurai0610/MASt3R-SLAM/blob/main/LICENSE.md
9. MASt3R checkpoint training-data license notice: https://github.com/naver/mast3r/blob/main/CHECKPOINTS_NOTICE
10. TUM RGB-D benchmark license and citation: https://cvg.cit.tum.de/data/datasets/rgbd-dataset
11. TUM RGB-D depth format and scale reference: https://cvg.cit.tum.de/_media/spezial/bib/sturm12iros.pdf

**문서 성격:** 이 문서는 계획과 단계별 실행 기록을 함께 보관한다. §13 고정값은 개발 장면용이며, 이후 평가 결과는 사전 고정된 설정을 그대로 적용하고 별도 결과 문서에 기록한다.
