# XR Image Segmentation Project

## 프로젝트 개요
YOLO v11 모델을 활용하여 Quest 3에서 동적 물체에 대한 실시간 segmentation 및 RGBD point cloud 생성 프로젝트

### 목표
1. 특정 물체 선택 후 segmentation 추적
2. RGBD 데이터 기반 point cloud 생성
3. Point cloud 기반 mesh 생성 (예정)

### 제약 조건
- **플랫폼**: Meta Quest 3 APK
- **Unity 버전**: 6000.0.61f1
- **목표 프레임레이트**: 20-30 FPS

---

## Meta XR API 사용 가이드

### 1. Passthrough Camera API

#### 개요
Quest 3의 패스스루 카메라에서 RGB 이미지를 획득하는 API

#### 주요 컴포넌트
- **WebCamTextureManager**: 카메라 텍스처 관리
- **PassthroughCameraUtils**: 카메라 유틸리티 함수
- **PassthroughCameraPermissions**: 권한 관리

#### 지원 조건
```csharp
// Quest 3, Quest 3S만 지원
// Horizon OS v74 이상 필요
bool isSupported = PassthroughCameraUtils.IsSupported;
```

#### 초기화 순서
1. `OVRPassthroughLayer`가 씬에 존재하고 활성화되어야 함
2. 카메라 권한 요청 (`PassthroughCameraPermissions.AskCameraPermissions()`)
3. `WebCamTexture` 초기화

#### 핵심 함수

**카메라 텍스처 얻기**
```csharp
WebCamTextureManager webCamManager;
WebCamTexture rgbTexture = webCamManager.WebCamTexture;
Color32[] pixels = new Color32[rgbTexture.width * rgbTexture.height];
rgbTexture.GetPixels32(pixels);
```

**2D 픽셀 → 3D 월드 레이 변환**
```csharp
// screenPoint: RGB 텍스처 상의 픽셀 좌표
Ray worldRay = PassthroughCameraUtils.ScreenPointToRayInWorld(
    PassthroughCameraEye.Left,
    new Vector2Int(pixelX, pixelY)
);
// depth와 결합하여 3D 위치 계산
Vector3 worldPos = worldRay.GetPoint(depthInMeters);
```

**카메라 Intrinsics 얻기**
```csharp
PassthroughCameraIntrinsics intrinsics = PassthroughCameraUtils.GetCameraIntrinsics(PassthroughCameraEye.Left);
// intrinsics.FocalLength: 초점 거리 (픽셀 단위)
// intrinsics.PrincipalPoint: 주점 위치
// intrinsics.Resolution: 최대 해상도
```

**지원 해상도 확인**
```csharp
List<Vector2Int> sizes = PassthroughCameraUtils.GetOutputSizes(PassthroughCameraEye.Left);
```

---

### 2. Environment Depth API

#### 개요
Quest 3의 depth 센서를 통해 환경 깊이 정보를 획득하는 API

#### 주요 컴포넌트
- **EnvironmentDepthManager**: Depth 데이터 관리 (Meta.XR.EnvironmentDepth 네임스페이스)

#### 사용 설정
```csharp
using Meta.XR.EnvironmentDepth;

[SerializeField] private EnvironmentDepthManager _depthManager;
```

#### Depth 텍스처 접근
```csharp
// Depth 사용 가능 여부 확인
if (!_depthManager.IsDepthAvailable) return;

// 전처리된 Depth 텍스처 얻기 (글로벌 셰이더 프로퍼티로 제공됨)
RenderTexture depthRT = Shader.GetGlobalTexture("_PreprocessedEnvironmentDepthTexture") as RenderTexture;
```

#### GPU → CPU 데이터 읽기 (AsyncGPUReadback)
```csharp
// Depth 텍스처는 RHalf(16bit float) 포맷
Texture2D cpuDepthTex = new Texture2D(depthRT.width, depthRT.height, TextureFormat.RHalf, false, true);

AsyncGPUReadback.Request(depthRT, 0, request => {
    if (request.hasError) return;
    var data = request.GetData<ushort>();
    cpuDepthTex.LoadRawTextureData(data);
    cpuDepthTex.Apply();
});
```

#### Depth 값 읽기 (미터 단위)
```csharp
var depthData = cpuDepthTex.GetRawTextureData<ushort>();
int pixelIndex = y * depthWidth + x;
float depthMeters = Mathf.HalfToFloat(depthData[pixelIndex]);

// 유효 범위 필터링 (0.1m ~ 3.0m 권장)
if (depthMeters > 0.1f && depthMeters < 3.0f) {
    // 유효한 depth
}
```

---

## RGB-Depth 좌표 정렬

### 문제
- RGB 이미지와 Depth 이미지의 해상도가 다름
- 두 텍스처 간 좌표계 정렬 필요

### 해결 방법
```csharp
// 1. 마스크 좌표 → RGB 픽셀 좌표 매핑
Vector2Int rgbPixel = MapMaskToRGBPixel(maskX, maskY, boundingBox, rgbWidth, rgbHeight);

// 2. RGB 픽셀 좌표 → 정규화 UV
float u = (float)rgbPixel.x / rgbWidth;
float v = (float)rgbPixel.y / rgbHeight;

// 3. UV → Depth 픽셀 좌표
int depthX = Mathf.FloorToInt(u * (depthWidth - 1));
int depthY = Mathf.FloorToInt(v * (depthHeight - 1));

// 4. Depth 샘플링 후 월드 좌표 계산
float depthMeters = Mathf.HalfToFloat(depthData[depthY * depthWidth + depthX]);
Ray worldRay = PassthroughCameraUtils.ScreenPointToRayInWorld(PassthroughCameraEye.Left, rgbPixel);
Vector3 worldPosition = worldRay.GetPoint(depthMeters);
```

---

## 프로젝트 핵심 파일

| 파일 | 역할 |
|------|------|
| `IEExecutor.cs` | 메인 추론 엔진, RGBD 추출, 트래킹 관리 |
| `IEBoxer.cs` | Bounding Box 시각화 |
| `IEMasker.cs` | Segmentation 마스크 렌더링 |
| `IEPassthroughTrigger.cs` | 컨트롤러 입력 처리 (물체 선택) |
| `IEPointcloud_Render.cs` | Point cloud 메쉬 렌더링 |
| `TrackingUtils.cs` | IoU 계산 등 트래킹 유틸리티 |
| `WebCamTextureManager.cs` | Passthrough 카메라 관리 |
| `PassthroughCameraUtils.cs` | 카메라 유틸리티 (좌표 변환 등) |

---

## 성능 최적화 팁

1. **Depth Readback**: `AsyncGPUReadback` 사용하여 비동기로 처리
2. **Point Cloud 샘플링**: `_samplingStep` 값으로 샘플링 간격 조절 (2~8)
3. **최대 포인트 수**: `_maxPoints`로 제한 (기본 8000)
4. **추론 분산**: `_layersPerFrame`으로 프레임당 처리 레이어 수 조절

---

## 주의 사항

1. **권한**: 카메라 권한이 반드시 승인되어야 함
2. **Passthrough 활성화**: `OVRPassthroughLayer`가 씬에 있고 활성화 필수
3. **Depth 가용성**: `_depthManager.IsDepthAvailable` 체크 필수
4. **좌표계**: Unity 좌표계와 Android Camera2 API 좌표계 차이 고려
5. **텍스처 포맷**: Depth는 `RHalf` 포맷 (16bit half-float)
