# XR Image Segmentation - Technical Manual
## Quest 3 RGBD PointCloud Pipeline 상세 기술 문서

---

## 1. 시스템 개요

### 1.1 프로젝트 목표
Meta Quest 3의 Passthrough Camera와 Environment Depth API를 활용하여 실시간으로:
1. YOLOv11-seg 모델로 동적 물체 인식 및 세그멘테이션
2. 선택된 물체의 RGB-D 데이터 추출
3. 3D PointCloud 생성 및 시각화

### 1.2 하드웨어/소프트웨어 스택

| 구성요소 | 사양 |
|---------|------|
| **헤드셋** | Meta Quest 3 (Horizon OS v74+) |
| **Unity** | 6000.0.61f1 (Unity 6) |
| **ML Runtime** | Unity Sentis |
| **모델** | YOLOv11n-seg (양자화, 3.1MB) |
| **타겟 FPS** | 20-30 FPS |

---

## 2. 센서 시스템 아키텍처

### 2.1 Quest 3 센서 구성

```
┌─────────────────────────────────────────────────────────────┐
│                    Meta Quest 3 Headset                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────────┐                      ┌─────────────┐      │
│   │ Left RGB    │                      │ Right RGB   │      │
│   │ Passthrough │                      │ Passthrough │      │
│   │ Camera      │                      │ Camera      │      │
│   │ (1280x1280) │                      │ (1280x1280) │      │
│   │ @60Hz       │                      │ @60Hz       │      │
│   └──────┬──────┘                      └──────┬──────┘      │
│          │                                    │              │
│          └────────────┬───────────────────────┘              │
│                       │                                      │
│   ┌───────────────────▼───────────────────┐                 │
│   │        Depth Estimation System         │                 │
│   │   (Stereo + ToF hybrid, 320x320)      │                 │
│   │              @30Hz                     │                 │
│   └───────────────────┬───────────────────┘                 │
│                       │                                      │
│   ┌───────────────────▼───────────────────┐                 │
│   │         Environment Depth API          │                 │
│   │    _EnvironmentDepthTexture (R16)     │                 │
│   │    _PreprocessedEnvironmentDepthTexture│                 │
│   │         (RGBAHalf, Linear meters)      │                 │
│   └───────────────────────────────────────┘                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 센서 간 해상도 및 주파수 차이

| 센서 | 해상도 | 주파수 | 좌표계 원점 |
|------|--------|--------|-------------|
| **RGB Passthrough** | 1280x1280 | 60 Hz | 카메라 렌즈 중심 |
| **Depth Map** | 320x320 | 30 Hz | 헤드셋 기준 (보정 필요) |
| **YOLO Input** | 640x640 | - | 이미지 좌상단 |
| **YOLO Mask Output** | 160x160 | - | 이미지 좌상단 |

**중요**: RGB는 60Hz, Depth는 30Hz이므로 **매 2프레임마다 depth가 갱신**됩니다.

---

## 3. 데이터 파이프라인 상세

### 3.1 전체 데이터 흐름

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FRAME N PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐    │
│  │ WebCamTexture│────▶│ TextureConv. │────▶│ Sentis Worker        │    │
│  │ (1280x1280)  │     │ ToTensor     │     │ (YOLOv11-seg)        │    │
│  │ RGB @60Hz    │     │ (640x640x3)  │     │ _layersPerFrame=25   │    │
│  └──────────────┘     └──────────────┘     └──────────┬───────────┘    │
│                                                        │                 │
│  ┌──────────────┐                                     ▼                 │
│  │DepthTexture  │     ┌────────────────────────────────────────────┐   │
│  │(320x320)     │     │              MODEL OUTPUTS                  │   │
│  │ @30Hz        │     ├────────────────────────────────────────────┤   │
│  │              │     │ output0: BoxCoords [N, 4] (cx,cy,w,h)      │   │
│  │ 2가지 형태:   │     │ output1: LabelIds  [N]                     │   │
│  │ R16_UNorm    │     │ output2: Masks     [N, 160, 160] (binary)  │   │
│  │ RGBAHalf     │     │ output3: MaskWeights [N,160,160] (sigmoid) │   │
│  └──────┬───────┘     └──────────────────────────────┬─────────────┘   │
│         │                                             │                 │
│         │         ┌───────────────────────────────────┘                 │
│         │         │                                                     │
│         │         ▼                                                     │
│         │  ┌─────────────────────────────────────────────────────┐     │
│         │  │              TRACKING MODULE                         │     │
│         │  │  - IoU 기반 프레임 간 물체 매칭                        │     │
│         │  │  - SmoothDamp으로 박스 떨림 보정                       │     │
│         │  │  - Lost frame 예측 (최대 15프레임)                    │     │
│         │  └──────────────────────────┬──────────────────────────┘     │
│         │                              │                                │
│         │                              ▼                                │
│         │  ┌─────────────────────────────────────────────────────┐     │
│         │  │         TARGET SELECTION (Trigger Input)             │     │
│         │  │  - 컨트롤러 트리거로 물체 선택                         │     │
│         │  │  - 선택된 물체만 RGBD 추출 대상                        │     │
│         │  └──────────────────────────┬──────────────────────────┘     │
│         │                              │                                │
│         ▼                              ▼                                │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │                  POINT CLOUD EXTRACTION                       │      │
│  │                                                                │      │
│  │  ┌────────────┐    ┌─────────────┐    ┌─────────────────┐    │      │
│  │  │ Mask[i,y,x]│───▶│ Coord Trans │───▶│ Depth Sampling  │    │      │
│  │  │ (160x160)  │    │ Mask→RGB    │    │ w/ Hole Filling │    │      │
│  │  └────────────┘    └─────────────┘    └────────┬────────┘    │      │
│  │                                                 │             │      │
│  │                                                 ▼             │      │
│  │  ┌─────────────────────────────────────────────────────┐     │      │
│  │  │              3D PROJECTION                           │     │      │
│  │  │  worldPos = cameraPos + (rayDir * depthMeters)      │     │      │
│  │  │                                                      │     │      │
│  │  │  rayDir = Rotate(cameraRot, normalize([             │     │      │
│  │  │      (pixel.x - cx) / fx,                           │     │      │
│  │  │      (pixel.y - cy) / fy,                           │     │      │
│  │  │      1.0                                            │     │      │
│  │  │  ]))                                                │     │      │
│  │  └─────────────────────────────────────────────────────┘     │      │
│  │                                                                │      │
│  └───────────────────────────────┬──────────────────────────────┘      │
│                                   │                                     │
│                                   ▼                                     │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │                    OUTPUT: PointBuffer                        │      │
│  │              RGBDPoint[] (worldPos + color)                   │      │
│  │                    최대 102,400 포인트                         │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 좌표계 시스템 (Coordinate Systems)

### 4.1 각 모듈별 좌표계

이 프로젝트에서 가장 복잡하고 오류가 발생하기 쉬운 부분입니다.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    COORDINATE SYSTEM OVERVIEW                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. YOLO INPUT SPACE (640x640)                                      │
│     ┌────────────────────┐                                          │
│     │ (0,0)              │  Y                                       │
│     │   ┌──────────────┐ │  ↓                                       │
│     │   │              │ │                                          │
│     │   │   물체       │ │  X →                                     │
│     │   │              │ │                                          │
│     │   └──────────────┘ │                                          │
│     │            (639,639)│                                          │
│     └────────────────────┘                                          │
│     - 원점: 좌상단                                                   │
│     - Y축: 아래로 증가                                               │
│                                                                      │
│  2. YOLO BBOX SPACE (Centered, -320 ~ +320)                         │
│                    (-320, -320)                                      │
│     ┌────────────────────┐                                          │
│     │         ↑ Y        │                                          │
│     │         │          │                                          │
│     │    ←────┼────→ X   │                                          │
│     │    (0,0)│          │  centerX, centerY는 이미지 중심 기준      │
│     │         ↓          │  Width, Height는 픽셀 단위               │
│     └────────────────────┘                                          │
│                    (+320, +320)                                      │
│                                                                      │
│  3. YOLO MASK SPACE (160x160)                                       │
│     ┌────────────────────┐                                          │
│     │ (0,0)              │  mask[targetIndex, y, x]                 │
│     │                    │  y=0: 상단                                │
│     │                    │  y=159: 하단                              │
│     │            (159,159)│                                          │
│     └────────────────────┘                                          │
│                                                                      │
│  4. UNITY TEXTURE SPACE (1280x1280)                                 │
│     ┌────────────────────┐                                          │
│     │            (1279,1279)│  Unity Texture2D는                    │
│     │                    │  Y=0이 하단!                              │
│     │                    │                                          │
│     │ (0,0)              │                                          │
│     └────────────────────┘                                          │
│                                                                      │
│  5. WORLD SPACE (Unity)                                             │
│              Y ↑                                                     │
│                │                                                     │
│                │                                                     │
│         ←──────┼──────→ X                                           │
│               /│                                                     │
│              / │                                                     │
│             Z  (Forward)                                             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 좌표 변환 공식

#### 4.2.1 Mask → RGB 픽셀 변환

```csharp
// IEExecutor.cs: MapMaskToRGBPixelFloat()

// 1. 마스크 좌표를 0~1 UV로 정규화
float u = maskX / 160f;  // 0 ~ 1
float v = maskY / 160f;  // 0 ~ 1

// 2. BBox를 0~1 공간으로 정규화 (YOLO centered → normalized)
float bboxCenterX_Norm = box.CenterX / 640f;  // -0.5 ~ 0.5
float bboxCenterY_Norm = box.CenterY / 640f;  // -0.5 ~ 0.5
float bboxW_Norm = box.Width / 640f;
float bboxH_Norm = box.Height / 640f;

// 3. 텍스처 U 계산 (X축은 직관적)
float texU = 0.5f + bboxCenterX_Norm - bboxW_Norm * 0.5f + u * bboxW_Norm;

// 4. 텍스처 V 계산 (Y축 변환 - YOLO Y와 Unity Y가 반대)
//    YOLO: Y↓ (아래가 양수)
//    Unity Texture: V↑ (위가 양수)
float boxBottomV = 0.5f - (bboxCenterY_Norm + bboxH_Norm * 0.5f);
float boxTopV = 0.5f - (bboxCenterY_Norm - bboxH_Norm * 0.5f);
float texV = boxBottomV + v * (boxTopV - boxBottomV);

// 5. 픽셀 좌표로 변환
int pixelX = (int)(texU * rgbW);
int pixelY = (int)(texV * rgbH);
```

#### 4.2.2 RGB 픽셀 → Depth 픽셀 변환

```csharp
// 단순 UV 매핑 (RGB와 Depth의 FOV가 유사하다고 가정)
float u = (float)rgbPixel.x / rgbW;
float v = (float)rgbPixel.y / rgbH;

int depthX = (int)(u * (depthW - 1));
int depthY = (int)(v * (depthH - 1));
```

#### 4.2.3 2D 픽셀 + Depth → 3D World 좌표 변환

```csharp
// Camera Intrinsics 사용
// fx, fy: Focal length (픽셀 단위)
// cx, cy: Principal point (광학 중심)

// 1. 카메라 공간에서의 방향 벡터
Vector3 dirInCamera = new Vector3(
    (pixel.x - cx) / fx,
    (pixel.y - cy) / fy,
    1f
);

// 2. 정규화
dirInCamera = dirInCamera.normalized;

// 3. 월드 공간으로 회전
Vector3 dirInWorld = cameraRotation * dirInCamera;

// 4. 최종 월드 좌표
Vector3 worldPos = cameraPosition + dirInWorld * depthMeters;
```

---

## 5. Depth 데이터 처리

### 5.1 두 가지 Depth 텍스처 형식

Quest 3의 Environment Depth API는 **두 가지 형태**의 depth 텍스처를 제공합니다:

| 텍스처 | 형식 | 값 범위 | 처리 방법 |
|--------|------|---------|-----------|
| `_EnvironmentDepthTexture` | R16_UNorm | 0.0 ~ 1.0 (정규화) | NDC 변환 필요 |
| `_PreprocessedEnvironmentDepthTexture` | RGBAHalf | 미터 단위 (직접) | 변환 불필요 |

### 5.2 Depth 선형화 (Linearization)

```csharp
// _EnvironmentDepthTexture (R16_UNorm) 사용 시
float rawDepth = depthTexture[x, y];  // 0 ~ 1

// NDC (Normalized Device Coordinates) 변환
float depthNdc = rawDepth * 2.0f - 1.0f;  // -1 ~ 1

// ZBuffer 파라미터를 이용한 선형 depth 복원
Vector4 zParams = Shader.GetGlobalVector("_EnvironmentDepthZBufferParams");
float depthMeters = zParams.x / (depthNdc + zParams.y);

// _PreprocessedEnvironmentDepthTexture (RGBAHalf) 사용 시
// 이미 미터 단위이므로 변환 불필요!
float depthMeters = Mathf.HalfToFloat(rawDepthHalf);
```

### 5.3 Depth Hole Filling (Trimmed Mean)

Depth 맵에는 종종 **구멍(hole)**이 있습니다 (반사 표면, 투명 물체 등).

```hlsl
// DepthToPointCloud.compute

float SampleDepthFilled(int2 rgbPixel, int rgbW, int rgbH)
{
    // 1. 중심 샘플
    float d = SampleDepthSimple(rgbPixel, rgbW, rgbH);
    if (d >= _MinDepth && d <= _MaxDepth) return d;

    // 2. 5x5 이웃 탐색
    const int radius = 2;
    float sumDepth = 0.0;
    float validCount = 0.0;
    float minVal = 1e10;
    float maxVal = -1e10;

    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            if (x == 0 && y == 0) continue;

            float nd = SampleDepthSimple(rgbPixel + int2(x, y), rgbW, rgbH);
            if (nd >= _MinDepth && nd <= _MaxDepth) {
                sumDepth += nd;
                validCount += 1.0;
                minVal = min(minVal, nd);
                maxVal = max(maxVal, nd);
            }
        }
    }

    // 3. Trimmed Mean: 최소/최대값 제외 (Outlier 제거)
    if (validCount >= 3.0) {
        sumDepth -= minVal;
        sumDepth -= maxVal;
        validCount -= 2.0;
        return sumDepth / validCount;
    }
    else if (validCount > 0.0) {
        return sumDepth / validCount;
    }

    return -1.0;  // 실패
}
```

---

## 6. Camera Intrinsics & Calibration

### 6.1 Camera Intrinsics 구조

```csharp
public struct PassthroughCameraIntrinsics
{
    public Vector2 FocalLength;      // (fx, fy) 픽셀 단위
    public Vector2 PrincipalPoint;   // (cx, cy) 광학 중심
    public Vector2Int Resolution;    // 기준 해상도
    public float Skew;               // 비직교성 계수 (보통 0에 가까움)
}
```

### 6.2 Intrinsics 스케일링

카메라 intrinsics는 **특정 해상도 기준**으로 제공됩니다. 실제 사용 해상도가 다르면 스케일링 필요:

```csharp
// 기준 해상도에서 실제 해상도로 스케일링
float scale = (float)actualWidth / intrinsics.Resolution.x;

float fx_scaled = intrinsics.FocalLength.x * scale;
float fy_scaled = intrinsics.FocalLength.y * scale;
float cx_scaled = intrinsics.PrincipalPoint.x * scale;
float cy_scaled = intrinsics.PrincipalPoint.y * scale;
```

### 6.3 카메라 Pose 획득

```csharp
// PassthroughCameraUtils.cs

public static Pose GetCameraPoseInWorld(PassthroughCameraEye cameraEye)
{
    // 1. 헤드셋 기준 카메라 상대 위치/회전 (캐시됨, 고정값)
    OVRPose headFromCamera = GetCachedCameraPoseRelativeToHead(cameraEye);

    // 2. 현재 프레임의 헤드 월드 포즈
    OVRPose worldFromHead = OVRPlugin.GetNodePoseStateImmediate(Node.Head).Pose;

    // 3. 월드 기준 카메라 포즈 계산
    OVRPose worldFromCamera = worldFromHead * headFromCamera;

    // 4. 추가 회전 보정 (카메라 좌표계 변환)
    worldFromCamera.orientation *= Quaternion.Euler(180, 0, 0);

    return new Pose(worldFromCamera.position, worldFromCamera.orientation);
}
```

---

## 7. 트래킹 시스템

### 7.1 IoU 기반 프레임 간 매칭

```csharp
// TrackingUtils.cs

public static float CalculateIoU(BoundingBox boxA, BoundingBox boxB)
{
    // 교집합 영역 계산
    float xA = Mathf.Max(boxA.Left, boxB.Left);
    float yA = Mathf.Max(boxA.Top, boxB.Top);
    float xB = Mathf.Min(boxA.Right, boxB.Right);
    float yB = Mathf.Min(boxA.Bottom, boxB.Bottom);

    float interArea = Mathf.Max(0, xB - xA) * Mathf.Max(0, yB - yA);

    float boxAArea = boxA.Width * boxA.Height;
    float boxBArea = boxB.Width * boxB.Height;

    // IoU = 교집합 / 합집합
    return interArea / (boxAArea + boxBArea - interArea);
}
```

### 7.2 물체 매칭 알고리즘

```csharp
// IEExecutor.cs: ProcessSuccessState()

// 최적 매칭 점수 계산
for (int i = 0; i < currentFrameBoxes.Count; i++)
{
    float iou = TrackingUtils.CalculateIoU(_lockedTargetBox, currBox);

    if (iou < _minIoUThreshold) continue;  // 최소 IoU 임계값 (0.3)

    // 거리 기반 점수 (중심점 거리)
    float dist = Vector2.Distance(prevCenter, currCenter);
    float distScore = 1.0f - Mathf.Clamp01(dist / (inputSize * 0.5f));

    // 최종 점수: IoU 70% + 거리 30%
    float totalScore = (iou * 0.7f) + (distScore * 0.3f);
}
```

### 7.3 박스 스무딩 (SmoothDamp)

```csharp
// 떨림 방지를 위한 부드러운 보간
Vector2 smoothedPos = Vector2.SmoothDamp(
    currentPos,
    targetPos,
    ref _centerVelocity,
    _currentSmoothTime  // 속도에 따라 0.1 ~ 0.5초 가변
);
```

### 7.4 Lost Frame 처리

물체가 일시적으로 가려지거나 검출 실패 시:

```csharp
if (bestMatch == null)
{
    _consecutiveLostFrames++;

    if (_consecutiveLostFrames <= _maxLostFrames)  // 최대 15프레임
    {
        // 이전 속도로 위치 예측
        PredictBoxMovement();
        _ieMasker.KeepCurrentMask();  // 이전 마스크 유지
    }
    else
    {
        ResetTracking();  // 트래킹 해제
    }
}
```

---

## 8. 주요 파라미터 설정 가이드

### 8.1 IEExecutor Inspector 설정

```
┌─────────────────────────────────────────────────────────────┐
│                    IEExecutor Settings                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [Model Settings]                                            │
│  ├─ Input Size: 640 x 640 (고정, YOLO 입력 크기)             │
│  ├─ Backend: GPUCompute (Quest 3 최적)                       │
│  ├─ Layers Per Frame: 25 (추론 분산, 낮을수록 부드러움)       │
│  └─ Confidence Threshold: 0.5 (검출 신뢰도)                  │
│                                                              │
│  [Natural Tracking Settings]                                 │
│  ├─ Max Lost Frames: 15 (가려짐 허용 프레임)                 │
│  ├─ Min IoU Threshold: 0.3 (매칭 최소 겹침)                  │
│  ├─ Min Smooth Time: 0.1 (빠른 물체용)                       │
│  ├─ Max Smooth Time: 0.5 (느린 물체용)                       │
│  └─ Size Smooth Time: 0.5 (크기 변화 스무딩)                 │
│                                                              │
│  [RGB-D & Depth Settings]                                    │
│  ├─ Depth Manager: (EnvironmentDepthManager 참조)           │
│  ├─ WebCam Manager: (WebCamTextureManager 참조)             │
│  ├─ Capture RGBD: true (포인트클라우드 활성화)               │
│  └─ Max Points: 102400 (최대 포인트 수)                      │
│                                                              │
│  [Optimization Settings]                                     │
│  ├─ Sampling Step: 1-8 (1=최고품질, 8=최고성능)              │
│  ├─ Sub Sampling Factor: 1-4 (픽셀 내 세분화)               │
│  ├─ Point Cloud Confidence: 0.1-0.9 (낮을수록 더 많은 포인트)│
│  ├─ Include Invalid Depth: true (구멍 채우기)               │
│  └─ Fallback Depth: 0 (0=평균 사용)                         │
│                                                              │
│  [Compute Shader Settings]                                   │
│  ├─ Point Cloud Shader: (DepthToPointCloud 참조)            │
│  ├─ Min Depth Range: 0.1m (최소 유효 깊이)                  │
│  └─ Max Depth Range: 3.0m (최대 유효 깊이)                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 권장 설정값

| 시나리오 | Sampling Step | Sub Sampling | Confidence | Max Points |
|---------|---------------|--------------|------------|------------|
| **최고 품질** | 1 | 4 | 0.1 | 102400 |
| **균형** | 2 | 2 | 0.3 | 51200 |
| **최고 성능** | 4 | 1 | 0.5 | 25600 |
| **디버깅** | 1 | 1 | 0.1 | 102400 |

---

## 9. 트러블슈팅 가이드

### 9.1 포인트클라우드가 생성되지 않음

```
증상: CurrentPointCount = 0

체크리스트:
□ Console에서 "Depth range" 로그 확인
  - avg가 0.001m 이하 → Depth 선형화 오류
  - avg가 100m 이상 → Depth 텍스처 형식 불일치

□ _captureRGBD = true 확인
□ Depth Manager 참조 연결 확인
□ Environment Depth 활성화 확인 (Project Settings)
```

### 9.2 포인트클라우드 위치가 어긋남

```
증상: 포인트가 물체와 다른 위치에 생성됨

체크리스트:
□ Mask 오버레이가 물체와 정렬되는지 확인
  - 정렬 안됨 → 좌표 변환 오류
  - 정렬됨 → Depth 샘플링 위치 오류

□ Camera Intrinsics 스케일링 확인
  - 실제 해상도와 intrinsics 해상도 비교

□ 하드코딩 오프셋이 제거되었는지 확인
  - texV += 0.1 (제거됨)
  - depthMeters += 0.1 (제거됨)
```

### 9.3 포인트클라우드가 너무 적음

```
증상: 물체 일부만 포인트로 표현됨

체크리스트:
□ Min/Max Depth Range 확인
  - 물체가 범위 밖에 있을 수 있음

□ Point Cloud Confidence 낮추기 (0.1 권장)
□ Sampling Step 줄이기 (1 권장)
□ Include Invalid Depth = true 확인
```

### 9.4 프레임 드랍 / 성능 저하

```
증상: FPS < 20

체크리스트:
□ Layers Per Frame 올리기 (25 → 35)
□ Sampling Step 올리기 (1 → 4)
□ Max Points 줄이기
□ Backend가 GPUCompute인지 확인
□ Update Interval 확인 (0.05초 = 20FPS 제한)
```

---

## 10. 디버깅 방법

### 10.1 Console 로그 해석

```csharp
// 30프레임마다 출력되는 디버그 로그

[IEExecutor] CPU PointCloud: mask=1234, fallback=56, filtered=78, valid=1100 (89.2%)
//           mask: 마스크 내 픽셀 수
//           fallback: 기본값 사용된 픽셀 수
//           filtered: depth 범위 밖으로 필터링된 픽셀
//           valid: 최종 유효 포인트 수
//           %: 유효율 (높을수록 좋음)

[IEExecutor] Depth range: min=0.456m, max=1.234m, avg=0.789m
//           정상 범위: 0.1m ~ 3.0m
//           avg가 0에 가까우면 → 선형화 오류
//           avg가 매우 크면 → 텍스처 형식 오류
```

### 10.2 시각적 디버깅

1. **마스크 오버레이 확인**
   - IEMasker가 그리는 녹색 마스크가 물체와 정확히 겹치는지

2. **BBox 확인**
   - IEBoxer가 그리는 바운딩 박스 위치/크기 확인

3. **PointCloud 색상 확인**
   - 색상이 올바르면 RGB 매핑 정상
   - 전부 흰색이면 RGB 버퍼 오류

---

## 11. 파일 구조 요약

```
Assets/
├── Scripts/
│   ├── InferenceEngine/
│   │   ├── IEExecutor.cs          # 메인 엔진 (추론, 트래킹, 포인트 추출)
│   │   ├── IEBoxer.cs             # BBox 시각화
│   │   ├── IEMasker.cs            # Mask 시각화
│   │   ├── IEPassthroughTrigger.cs # 입력 처리
│   │   ├── For tracking/
│   │   │   └── TrackingUtils.cs   # IoU 계산
│   │   └── Editor/
│   │       └── IEModelEditorConverter.cs  # 모델 변환
│   │
│   └── PassthroughCamera/
│       ├── WebCamTextureManager.cs    # RGB 카메라 관리
│       └── PassthroughCameraUtils.cs  # Intrinsics, Pose 유틸리티
│
├── Shaders/
│   └── DepthToPointCloud.compute  # GPU 포인트 추출
│
└── Resources/
    └── Model/
        └── yolo11n-seg-sentis.sentis  # YOLO 모델
```

---

## 12. 향후 개선 방향

### 12.1 현재 한계점
1. RGB-Depth 프레임 동기화 없음 (latency 존재)
2. 단일 물체만 트래킹 가능
3. Mesh 생성 미구현

### 12.2 제안 개선사항
1. **프레임 동기화**: Timestamp 기반 RGB-Depth 매칭
2. **다중 물체 트래킹**: 각 물체별 독립 PointBuffer
3. **Mesh Reconstruction**: Poisson/Ball Pivoting 알고리즘 적용
4. **Temporal Filtering**: 시간축 스무딩으로 포인트 떨림 감소

---

*문서 버전: 1.0*
*최종 수정: 2026-01-30*
