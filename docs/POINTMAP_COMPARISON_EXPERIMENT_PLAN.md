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

### 4.1 카메라 보정 조건

**C0 — 미보정(주 비교):** 두 시스템 모두 GT 카메라 내부 파라미터를 **모델 입력으로 사용하지 않는다.** GT는 평가에만 사용한다.  
**C1 — 보정 이용(보조):** MASt3R-SLAM의 `--calib` 사용 결과는 별도 행으로 제시한다. VGGT가 원래 외부 calibration을 같은 방식으로 입력받는 모델이 아니므로 C1을 대칭 비교처럼 해석하지 않는다.

렌즈 왜곡이 큰 장면은 undistortion 적용 여부, 원본/교정 영상 공간, GT Depth 및 (K)의 기준 좌표계를 명확히 통일한다.

## 5. 중간 결과 표준 포맷

각 실행에서 데이터셋·프레임 ID·타임스탬프를 키로 다음을 기록한다.

```text
results/
  <dataset>/<sequence>/<protocol>/<model_variant>/<run_id>/
    config.json            # git SHA, ckpt SHA256, 인자, HW/SW, 처리 조건
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

### Phase 0 — 환경 준비·데이터 확인

- [ ] 두 공식 저장소와 모델 체크포인트를 분리 설치한다.
- [ ] TUM RGB-D 1개 시퀀스 다운로드 및 RGB–Depth–GT Pose 동기화.
- [ ] 두 모델의 기본 데모 실행 및 파일 저장 확인.
- [ ] 모든 실행의 commit/checkpoint hash 기록.

**완료 기준:** 같은 RGB 프레임 목록에서 두 모델 실행이 가능하고 중간 결과를 저장할 수 있다.

### Phase 1 — 표준 export 모듈 구현

- [ ] `export_vggt.py`: V-P, V-D, depth/confidence, camera, frame_id를 NPZ/PLY로 저장.
- [ ] `export_mast3r_slam.py` 또는 upstream 최소 패치: M-F의 `X_canon`, `T_WC`, confidence 및 키프레임 원본 ID를 저장.
- [ ] M-R이 필요하면 MASt3R 추론 직후 원시 예측 Pointmap을 저장하도록 **별도** 추출 위치 추가.
- [ ] coordinate/unit/rgb pixel map에 대한 round-trip 단위 테스트.

**완료 기준:** RGB 픽셀 한 개를 역추적하여 두 출력에서 해당 3D 위치와 Confidence를 조회할 수 있다.

### Phase 2 — GT 기반 정량 평가 파이프라인

- [ ] `associate_rgb_depth_pose.py`: 유효 GT pixel과 타임스탬프 매칭 결과 저장.
- [ ] `align_sim3.py`: 사전 고정한 alignment prefix와 변환 로깅.
- [ ] `evaluate_pixel.py`: Depth/3D per-pixel 오류 및 valid fraction 계산.
- [ ] `evaluate_cloud.py`: Accuracy/Completeness/F-score, common visibility 영역.
- [ ] `evaluate_outliers.py`: surface outlier/free-space violation/retention curve.

**완료 기준:** 합성 3D 점/카메라의 정답이 알려진 단위 테스트에서 정렬과 오차를 정확하게 회수한다.

### Phase 3 — 파일럿 및 실험 반복

- [ ] 1개 장면, 4/8/16/32프레임 입력 예산 실험(P1).
- [ ] 다른 장면에서 동일 조건 재현.
- [ ] 결과가 흔들리는 조건에 대해서는 3회 실행(초기 제안; deterministic 여부와 함께 표기).
- [ ] C0 미보정 주 비교와 C1 보정 보조 실험 분리.
- [ ] Confidence 상위 100/80/60/40% 유지에 대한 sensitivity 분석.
- [ ] 긴 연속 영상 운용 실험(P2) 별도 수행.

**완료 기준:** 성공/실패를 포함한 모든 조합에서 자동 생성된 CSV/JSON 비교표를 얻는다.

### Phase 4 — 사례 분석과 연구 연결

- [ ] 이상점이 많은 사례 5개 이상 기록(초기 제안).
- [ ] 책상 모서리, 가림, 무텍스처 벽, 반사 표면, 긴 이동 궤적 등 실패 원인 분류.
- [ ] VGGT의 `||V-P - V-D||`를 불확실성 지표 후보로 계산하고 **실제 outlier label과의 상관**을 확인한다.
- [ ] MASt3R-SLAM의 Confidence 및 융합 횟수와 이상점 간 관계도 함께 평가한다.
- [ ] 점 제거를 제안한다면 **Outlier 감소량과 Coverage 손실**을 반드시 함께 측정한다.

**완료 기준:** 수치·오차맵·동일 시점 pointcloud 사례가 모두 있는 사전 실험 보고서 작성.

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

- [ ] 비교 대상 VGGT checkpoint/version 선택(`VGGT-1B` 등) 및 라이선스 확인.
- [ ] MASt3R-SLAM의 실제 수정 없이 추출 가능한 Pointmap 단계와 export 위치 확정.
- [ ] 모델별 내부 입력 크기, 원본 해상도 역매핑 정확도 확인.
- [ ] GT 정렬용 prefix의 최소 길이·카메라 이동 퇴화 기준 결정.
- [ ] GT 오차 허용치, outlier τ, voxel 크기, valid depth range를 개발 장면에서 고정.
- [ ] GPU VRAM을 확인하고 VGGT의 최대 N에 대한 현실적인 제한 설정.
- [ ] 데이터셋별 재배포·사용 조건 확인.

---

## 참고 자료 및 구현 근거

1. Wang et al., **VGGT: Visual Geometry Grounded Transformer**, CVPR 2025. 논문: https://arxiv.org/abs/2503.11651
2. VGGT 공식 GitHub/README 및 `demo_viser.py`, `demo_colmap.py`: https://github.com/facebookresearch/vggt
3. Murai et al., **MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors**, CVPR 2025. 논문: https://arxiv.org/abs/2412.12392
4. MASt3R-SLAM 공식 GitHub의 `main.py`, `mast3r_slam/frame.py`, `mast3r_slam/evaluate.py`: https://github.com/rmurai0610/MASt3R-SLAM
5. Wang et al., **DUSt3R: Geometric 3D Vision Made Easy**: https://arxiv.org/abs/2312.14132
6. Leroy et al., **Grounding Image Matching in 3D with MASt3R**: https://arxiv.org/abs/2406.09756

**문서 성격:** 아직 실행·검증되지 않은 실험 계획이다. 표의 결과값과 결론은 실험 후 별도 문서에 기록한다.
