# XR Image Segmentation - Project Context

## Goal
YOLOv11 + Quest3를 활용하여 동적 물체에 대한 RGBD 데이터 추출, PointCloud 생성 및 최종 Mesh 구현
- **현재 진행 상황**: 1단계 (BBox Tracking + RGB Segmentation) 완료
- **차기 목표**: Environment Depth API 연동 및 실시간 PointCloud 생성

## Constraints
- **Platform**: Meta Quest 3 (APK 빌드)
- **Target Performance**: 20-30 FPS
- **Unity Version**: 6000.0.61f1 (Unity 6)

## Architecture
```
Passthrough Camera → IEExecutor (Sentis) → YOLOv11-seg → Tracking (IoU) → Mask Render
     (RGB/YUV)          (640x640)           (Inference)    (SmoothDamp)    (160x160)
```

## Key Files & Roles

### Core Logic
| File | Role |
|------|------|
| `Scripts/InferenceEngine/IEExecutor.cs` | 메인 엔진: 모델 추론 관리, RGBD 추출 및 트래킹 로직 핵심 |
| `Scripts/InferenceEngine/IEBoxer.cs` | YOLO 출력 기반 Bounding Box 시각화 및 클래스 라벨 표시 |
| `Scripts/InferenceEngine/IEMasker.cs` | Segmentation Mask 생성 및 특정 물체 마스크 렌더링 |
| `Scripts/InferenceEngine/For tracking/TrackingUtils.cs` | IoU 계산 및 트래킹 관련 유틸리티 함수 |

### Passthrough & Depth API
| File | Role |
|------|------|
| `Scripts/PassthroughCamera/WebCamTextureManager.cs` | Meta Passthrough 카메라 텍스처 획득 및 관리 |
| `Scripts/PassthroughCamera/PassthroughCameraUtils.cs` | **핵심 유틸리티**: 3D Raycast, Camera Intrinsics 및 좌표 변환 |
| `Scripts/InferenceEngine/IEPassthroughTrigger.cs` | 컨트롤러 입력(Trigger, B button) 처리 및 물체 선택/리셋 |

### Model & Assets
- `Resources/Model/yolo11n-seg-sentis.sentis`: YOLOv11 Nano Segmentation 모델 (양자화 완료, 3.1MB)
- `PointcloudShader.shader`: PointCloud 시각화를 위한 전용 셰이더
- `Mat_Pointcloud.mat`: PointCloud 렌더링용 머티리얼

## Configuration Parameters (IEExecutor)
- `_maxLostFrames = 15`: 물체가 가려졌을 때 추적을 유지하는 최대 프레임 수
- `_minIoUThreshold = 0.3f`: 이전 프레임과 현재 프레임의 물체 매칭 임계값
- `_smoothTime = 0.03~0.2f`: 속도에 따라 가변적으로 적용되는 박스 스무딩 시간
- `_layersPerFrame = 25`: 성능 최적화를 위한 프레임당 추론 분산 처리 레이어 수

## Data Flow
1. **Input**: `WebCamTexture`로부터 RGB 데이터 획득
2. **Preprocess**: 640x640 크기의 `Tensor`로 변환
3. **Inference**: Sentis Worker를 통한 비동기 추론 (`_layersPerFrame` 적용)
4. **Output**: BBox(N,4), LabelIds(N), Masks(N,160,160), Weights 획득
5. **Postprocess**: IoU 기반 매칭 → SmoothDamp 적용 → Mask & BBox 렌더링

## TODO List
1. [ ] **Environment Depth API 연동**: Meta XR Depth API를 통한 실시간 깊이 정보 획득
2. [ ] **Mask 영역 Depth 추출**: Segmentation 마스크 내부 픽셀에 해당하는 Depth 값 샘플링
3. [ ] **3D Point 변환**: Camera Intrinsics(fx, fy, cx, cy)를 사용하여 2D 픽셀 + Depth를 3D 월드 좌표로 변환
4. [ ] **PointCloud 시각화**: 생성된 3D 포인트를 효율적으로 렌더링하기 위한 데이터 구조 및 렌더러 구현
5. [ ] **Mesh 생성 (Next Stage)**: PointCloud 기반 실시간 Mesh Reconstruction 알고리즘 적용