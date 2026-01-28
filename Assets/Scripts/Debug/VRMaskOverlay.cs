using UnityEngine;
using UnityEngine.UI;
using PassthroughCameraSamples;

/// <summary>
/// [QuestCameraKit 방식] RGB Passthrough 카메라 좌표계 기준 마스크 오버레이
/// RGB 카메라 시점에서 Canvas를 배치하여 PointCloud와 정확히 정렬
/// </summary>
public class VRMaskOverlay : MonoBehaviour
{
    [Header("Settings")]
    [SerializeField] private IEExecutor _ieExecutor;
    [SerializeField] private float _defaultDistance = 1.2f;  // 트래킹 안 할 때 기본 거리
    [SerializeField] private float _maskAlpha = 0.4f;

    [Header("Dynamic Distance")]
    [Tooltip("트래킹 중일 때 물체 거리에 맞춰 Canvas 거리 자동 조절")]
    [SerializeField] private bool _useDynamicDistance = true;
    [SerializeField] private float _distanceSmoothSpeed = 5.0f;

    [Header("Calibration Controls")]
    [SerializeField] private float _baseScale = 0.0015f;  // 1.2m 거리 기준 스케일
    [SerializeField] private float _adjustmentSpeed = 0.001f;
    [SerializeField] private float _yOffset = 0.0f;
    [SerializeField] private float _xOffset = 0.0f;
    [SerializeField] private float _offsetSpeed = 0.5f;

    // 현재 적용 중인 거리
    private float _currentDistance;
    private float _currentScale;

    // 기준값 (1.2m 거리 기준)
    private const float REFERENCE_DISTANCE = 1.2f;

    // RGB Passthrough 카메라 Intrinsics 캐싱
    private PassthroughCameraIntrinsics _cachedIntrinsics;
    private bool _intrinsicsCached = false;

    private Canvas _vrCanvas;
    private RectTransform _canvasRect;

    private void Start()
    {
        if (_ieExecutor == null) _ieExecutor = FindObjectOfType<IEExecutor>();
        _currentDistance = _defaultDistance;
        _currentScale = _baseScale;
        SetupVRCanvas();
    }

    private void SetupVRCanvas()
    {
        GameObject canvasObj = new GameObject("VR_Mask_Overlay");
        _vrCanvas = canvasObj.AddComponent<Canvas>();
        _vrCanvas.renderMode = RenderMode.WorldSpace;

        CanvasScaler scaler = canvasObj.AddComponent<CanvasScaler>();
        scaler.dynamicPixelsPerUnit = 1;

        // [QuestCameraKit 방식] Canvas는 World Space에 독립적으로 존재
        // 매 프레임마다 RGB Passthrough 카메라 Pose를 따라감
        // 더 이상 Camera.main의 자식으로 설정하지 않음!

        _canvasRect = canvasObj.GetComponent<RectTransform>();
        _canvasRect.sizeDelta = new Vector2(1280, 1280);

        // 초기 위치 설정 (RGB 카메라 기준)
        UpdateCanvasTransform();

        if (_ieExecutor != null)
        {
            _ieExecutor.DisplayLocation = canvasObj.transform;
            if (_ieExecutor.Masker != null)
            {
                _ieExecutor.Masker.UpdateDisplayLocation(canvasObj.transform);
            }
        }
    }

    private void Update()
    {
        bool changed = false;

        // [핵심] 트래킹 중이면 물체 거리에 맞춰 Canvas 거리 자동 조절
        if (_useDynamicDistance && _ieExecutor != null && _ieExecutor.IsTracking)
        {
            float targetDistance = _ieExecutor.TrackedObjectDepth;

            // 부드럽게 거리 전환
            float newDistance = Mathf.Lerp(_currentDistance, targetDistance, _distanceSmoothSpeed * Time.deltaTime);

            if (Mathf.Abs(newDistance - _currentDistance) > 0.001f)
            {
                _currentDistance = newDistance;

                // 거리에 비례하여 스케일 조절 (원근법: 멀수록 크게)
                // 기준: 1.2m 거리에서 _baseScale
                _currentScale = _baseScale * (_currentDistance / REFERENCE_DISTANCE);
                changed = true;
            }
        }
        else if (_ieExecutor != null && !_ieExecutor.IsTracking)
        {
            // 트래킹 안 할 때는 기본 거리로 복귀
            if (Mathf.Abs(_currentDistance - _defaultDistance) > 0.01f)
            {
                _currentDistance = Mathf.Lerp(_currentDistance, _defaultDistance, _distanceSmoothSpeed * Time.deltaTime);
                _currentScale = _baseScale * (_currentDistance / REFERENCE_DISTANCE);
            }
        }

        // 1. 크기 조절 (왼쪽 스틱 상/하) - 기본 스케일 조절
        Vector2 leftStick = OVRInput.Get(OVRInput.Axis2D.PrimaryThumbstick, OVRInput.Controller.LTouch);
        if (Mathf.Abs(leftStick.y) > 0.1f)
        {
            _baseScale += leftStick.y * _adjustmentSpeed * Time.deltaTime;
            _baseScale = Mathf.Max(0.0001f, _baseScale);
            _currentScale = _baseScale * (_currentDistance / REFERENCE_DISTANCE);
            changed = true;
        }

        // 2. 상하 위치 조절 (오른쪽 스틱 상/하)
        Vector2 rightStick = OVRInput.Get(OVRInput.Axis2D.PrimaryThumbstick, OVRInput.Controller.RTouch);
        if (Mathf.Abs(rightStick.y) > 0.1f)
        {
            _yOffset += rightStick.y * _offsetSpeed * Time.deltaTime;
            changed = true;
        }

        // 3. 좌우 위치 조절 (오른쪽 스틱 좌/우)
        if (Mathf.Abs(rightStick.x) > 0.1f)
        {
            _xOffset += rightStick.x * _offsetSpeed * Time.deltaTime;
            changed = true;
        }

        // [QuestCameraKit 방식] 매 프레임 RGB 카메라 Pose를 따라 Canvas 업데이트
        // RGB 카메라 Pose가 매 프레임 변하므로 항상 호출
        UpdateCanvasTransform();

        // [SYNC] IEExecutor에 값 전달 (WYSIWYG: 보이는 대로 찍히게)
        if (changed && _ieExecutor != null)
        {
            // Scale: REFERENCE_DISTANCE (1.2m) 기준 비율
            _ieExecutor.CalibrationScale = _baseScale / 0.0015f;  // 기본값 0.0015 기준

            // X/Y Offset을 픽셀 단위로 변환하여 전달
            // 오프셋도 거리에 비례하여 스케일링
            float distanceRatio = _currentDistance / REFERENCE_DISTANCE;
            float xOffsetInPixels = (_xOffset / distanceRatio) / _baseScale;
            float yOffsetInPixels = (_yOffset / distanceRatio) / _baseScale;

            _ieExecutor.CalibrationXOffset = xOffsetInPixels;
            _ieExecutor.CalibrationYOffset = yOffsetInPixels;
        }
    }

    private void UpdateCanvasTransform()
    {
        if (_vrCanvas == null) return;

        // [QuestCameraKit 방식] RGB Passthrough 카메라 Pose 가져오기
        Pose rgbCameraPose = PassthroughCameraUtils.GetCameraPoseInWorld(PassthroughCameraEye.Left);
        Vector3 rgbCameraPos = rgbCameraPose.position;
        Quaternion rgbCameraRot = rgbCameraPose.rotation;

        // RGB 카메라 Intrinsics 캐싱 (FOV 계산용)
        if (!_intrinsicsCached)
        {
            _cachedIntrinsics = PassthroughCameraUtils.GetCameraIntrinsics(PassthroughCameraEye.Left);
            _intrinsicsCached = true;
        }

        // Canvas를 RGB 카메라 앞에 배치 (RGB 카메라 좌표계 기준)
        // RGB 카메라의 forward 방향으로 _currentDistance만큼 떨어진 위치
        Vector3 forward = rgbCameraRot * Vector3.forward;
        Vector3 right = rgbCameraRot * Vector3.right;
        Vector3 up = rgbCameraRot * Vector3.up;

        // [Fix] Principal Point Offset 보정
        // 실제 렌즈의 광학 중심(cx, cy)과 이미지 센서의 기하학적 중심(w/2, h/2)의 차이를 보정
        // 이 보정이 없으면 마스크가 한쪽으로 치우쳐 보임 (특히 Quest 3에서 두드러짐)
        Vector2 resolution = _cachedIntrinsics.Resolution;
        Vector2 principalPoint = _cachedIntrinsics.PrincipalPoint;
        Vector2 focalLength = _cachedIntrinsics.FocalLength;

        // 중심점 차이 (Pixel 단위)
        // 예: cx가 700이고 w/2가 640이면, +60 픽셀 차이 (오른쪽으로 치우침)
        float diffX = principalPoint.x - (resolution.x / 2.0f);
        float diffY = principalPoint.y - (resolution.y / 2.0f);

        // World Unit으로 변환 (유사 삼각형 비례식: offset / distance = diff / focalLength)
        // 렌즈 중심이 기하 중심보다 오른쪽에 있다면, 캔버스를 왼쪽(-)으로 밀어야 렌즈 축과 정렬됨
        float principalOffsetX = -(diffX / focalLength.x) * _currentDistance;
        float principalOffsetY = -(diffY / focalLength.y) * _currentDistance; // Y축 방향 확인 필요 (일단 동일 로직 적용)

        // X/Y 오프셋도 거리에 비례하여 적용 (RGB 카메라 좌표계)
        float distanceRatio = _currentDistance / REFERENCE_DISTANCE;
        float scaledXOffset = _xOffset * distanceRatio;
        float scaledYOffset = _yOffset * distanceRatio;

        // 최종 위치 계산: 기본 거리 + 사용자 오프셋 + 렌즈 중심 보정(Principal Point)
        Vector3 canvasWorldPos = rgbCameraPos
            + forward * _currentDistance
            + right * (scaledXOffset + principalOffsetX)
            + up * (scaledYOffset + principalOffsetY);

        _vrCanvas.transform.position = canvasWorldPos;
        _vrCanvas.transform.rotation = rgbCameraRot;
        _vrCanvas.transform.localScale = Vector3.one * _currentScale;
    }

    private void LateUpdate()
    {
        if (_vrCanvas != null)
        {
            RawImage[] masks = _vrCanvas.GetComponentsInChildren<RawImage>();
            foreach (var mask in masks)
            {
                Color c = mask.color;
                c.a = _maskAlpha;
                mask.color = c;

                RectTransform rt = mask.GetComponent<RectTransform>();
                rt.anchorMin = Vector2.zero;
                rt.anchorMax = Vector2.one;
                rt.offsetMin = Vector2.zero;
                rt.offsetMax = Vector2.zero;
                rt.localPosition = Vector3.zero;
                rt.localRotation = Quaternion.identity;
                rt.localScale = Vector3.one;
            }
        }
    }
}
