# XR Image Segmentation - Project Context

## Goal
YOLOv11 + Quest3를 활용하여 동적 물체에 대한 RGBD 데이터 추출, PointCloud 생성 및 최종 Mesh 구현
- **현재 진행 상황**: 2단계 (Environment Depth API 연동 및 초기 Mesh 생성) 진행 중
- **차기 목표**: Mesh 스케일/왜곡 문제 해결 및 실시간 PointCloud 최적화

## Constraints
- **Platform**: Meta Quest 3 (APK 빌드)
- **Target Performance**: 20-30 FPS
- **Unity Version**: 6000.0.61f1 (Unity 6)

## Architecture
```
Passthrough Camera → IEExecutor (Sentis) → YOLOv11-seg → Tracking (IoU) → Extract Grid (RGB+Depth) → PointCloudRenderer
     (RGB/YUV)          (640x640)           (Inference)    (SmoothDamp)      (Rescaled BBox)        (Mesh Generation)
```

## Recent Changes & Fixes (2026-01-26)
- **Coordinate Scale Mismatch Fix (`IEExecutor.cs`)**:
  - **문제**: 추론 해상도(640x640)와 카메라 해상도(1280x1280) 불일치로 인해 BBox가 잘못된 위치의 Depth를 참조, "치즈처럼 늘어나는" 현상 및 "자글자글한 노이즈" 발생.
  - **해결**: `ExtractGridFromCache`에서 BBox 좌표를 `_inputSize` 기준에서 카메라 텍스처 해상도(`rgbW`, `rgbH`) 기준으로 리스케일링.
  - **추가**: Depth 샘플링 시 이미지 경계(Out of Bounds)를 방지하기 위해 `Mathf.Clamp` 적용.

## Current Issues (Known Bugs)
- **Mesh Stretching (메쉬 늘어짐 현상)**:
  - **증상**: 생성된 Mesh가 실제 물체보다 길게 늘어나 보이거나 비율이 맞지 않음.
  - **추정 원인**:
    1. **Depth Map Y-axis Flip**: Unity 텍스처 좌표계(Bottom-Left)와 Computer Vision 좌표계(Top-Left) 간의 Y축 반전 문제 가능성.
    2. **Focal Length Mismatch**: Passthrough Intriniscs의 `fx`, `fy` 값과 실제 텍스처(Cropped/Scaled) 간의 비율 불일치.
    3. **Aspect Ratio**: 추론(1:1) vs 카메라(Rectangular) 비율 차이로 인한 UV 매핑 왜곡.

## Key Files & Roles

### Core Logic
| File | Role |
|------|------|
| `Scripts/InferenceEngine/IEExecutor.cs` | 메인 엔진: 모델 추론, RGBD 추출(Coordinate Scale 보정됨), 트래킹 |
| `Scripts/InferenceEngine/IEBoxer.cs` | YOLO 출력 기반 Bounding Box 시각화 |
| `Scripts/InferenceEngine/IEMasker.cs` | Segmentation Mask 생성 및 렌더링 |
| `Scripts/InferenceEngine/PointCloudRenderer.cs` | PointCloud 및 Mesh 생성/렌더링 (Grid 기반 Triangulation) |

### Passthrough & Depth API
| File | Role |
|------|------|
| `Scripts/PassthroughCamera/WebCamTextureManager.cs` | Meta Passthrough 카메라 텍스처 관리 |
| `Scripts/PassthroughCamera/PassthroughCameraUtils.cs` | Camera Intrinsics 및 좌표 변환 유틸리티 |

## Configuration Parameters (IEExecutor)
- `_inputSize = (640, 640)`: 추론 엔진 입력 해상도
- `_confidenceThreshold`: 마스크 필터링 임계값
- `_samplingStep`: Mesh 생성 시 그리드 샘플링 간격 (성능 vs 품질 조절)

## TODO List
1. [x] **Environment Depth API 연동**: Meta XR Depth API 연동 완료
2. [x] **Mask 영역 Depth 추출**: Segmentation 마스크 기반 Depth 샘플링 구현 완료
3. [x] **3D Point 변환**: Intrinsics 활용 역투영(Back-projection) 구현 완료
4. [ ] **Mesh 왜곡 보정**: Stretching 현상 해결 (Y-Flip, Intrinsics 검증)
5. [ ] **Mesh Rendering 최적화**: 실시간 업데이트 성능 개선
