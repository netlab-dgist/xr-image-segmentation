using System;
using System.Collections;
using System.Collections.Generic;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Rendering;
using Meta.XR.EnvironmentDepth;
using PassthroughCameraSamples;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

public class IEExecutor : MonoBehaviour
{
    enum InferenceDownloadState
    {
        Running = 0,
        RequestingOutputs = 1,
        Success = 2,
        Error = 3,
        Cleanup = 4,
        Completed = 5
    }

    [Header("Model Settings")]
    [SerializeField] private Vector2Int _inputSize = new(640, 640);
    [SerializeField] private BackendType _backend = BackendType.GPUCompute;
    [SerializeField] private ModelAsset _sentisModel;
    [SerializeField] private int _layersPerFrame = 12;
    [SerializeField] private float _confidenceThreshold = 0.5f;
    [SerializeField] private Transform _displayLocation;

    [Header("Performance Mode")]
    [Tooltip("체크 해제 시: 무거운 UI(박스/마스크) 렌더링을 중지하여 깜빡임을 없앱니다. Point Cloud는 계속 나옵니다.")]
    public bool EnableUIRendering = false; // [기본값 False] 깜빡임 방지

    [Header("Depth Settings")]
    [SerializeField] private EnvironmentDepthManager _depthManager;
    [SerializeField] private int _maxPoints = 8000;
    [SerializeField] private Gradient _depthGradient;
    [Range(2, 8)]
    [SerializeField] private int _samplingStep = 4;

    public struct RGBDPoint
    {
        public Vector3 worldPos;
        public Color32 color;
    }

    // Burst Job용 데이터 구조체
    [BurstCompile]
    private struct DepthExtractionJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<ushort> depthData;
        [ReadOnly] public NativeArray<float> maskData;

        public int depthW;
        public int depthH;
        public int maskW;
        public int maskH;
        public int samplingStep;
        public float confidenceThreshold;

        // Bounding Box 정보
        public float rawBoxCenterX;
        public float rawBoxCenterY;
        public float rawBoxWidth;
        public float rawBoxHeight;

        // 카메라 정보 (Depth 촬영 시점 기준)
        public float3 depthCameraPose;
        public quaternion depthCameraRot;
        public float2 focalLength;
        public float2 principalPoint;
        public float2 sensorRes;

        // 출력
        [NativeDisableParallelForRestriction]
        public NativeArray<float3> outPositions;
        [NativeDisableParallelForRestriction]
        public NativeArray<float> outDepths;
        public NativeArray<int> outValid;

        public void Execute(int index)
        {
            int totalX = maskW / samplingStep;
            int localY = index / totalX;
            int localX = index % totalX;

            int y = localY * samplingStep;
            int x = localX * samplingStep;

            if (y >= maskH || x >= maskW)
            {
                outValid[index] = 0;
                return;
            }

            int maskIndex = y * maskW + x;
            if (maskIndex >= maskData.Length || maskData[maskIndex] <= confidenceThreshold)
            {
                outValid[index] = 0;
                return;
            }

            // 마스크 좌표 → 이미지 좌표
            float normX = (float)x / maskW;
            float normY = (float)y / maskH;

            float imgPixelX = rawBoxCenterX - (rawBoxWidth * 0.5f) + (normX * rawBoxWidth);
            float imgPixelY = rawBoxCenterY - (rawBoxHeight * 0.5f) + (normY * rawBoxHeight);

            float u = math.clamp(imgPixelX / 640f, 0f, 1f);
            float v = math.clamp(imgPixelY / 640f, 0f, 1f);

            // Depth 샘플링
            int dx = (int)(u * (depthW - 1));
            int dy = (int)((1.0f - v) * (depthH - 1));

            int depthIndex = dy * depthW + dx;
            if (depthIndex < 0 || depthIndex >= depthData.Length)
            {
                outValid[index] = 0;
                return;
            }

            float depthMeters = HalfToFloat(depthData[depthIndex]);

            // 유효 거리 필터링 (0.1m ~ 3.0m)
            if (depthMeters <= 0.1f || depthMeters >= 3.0f)
            {
                outValid[index] = 0;
                return;
            }

            // 카메라 좌표계에서 방향 계산 (Intrinsics 사용)
            float camPixelX = u * sensorRes.x;
            float camPixelY = (1.0f - v) * sensorRes.y;

            float3 dirInCamera = new float3(
                (camPixelX - principalPoint.x) / focalLength.x,
                (camPixelY - principalPoint.y) / focalLength.y,
                1.0f
            );
            dirInCamera = math.normalize(dirInCamera);

            // Depth 촬영 시점의 카메라 포즈로 월드 좌표 계산
            float3 dirInWorld = math.mul(depthCameraRot, dirInCamera);
            float3 worldPos = depthCameraPose + dirInWorld * depthMeters;

            outPositions[index] = worldPos;
            outDepths[index] = depthMeters;
            outValid[index] = 1;
        }

        // Half-precision float 변환 (Burst 호환)
        private static float HalfToFloat(ushort half)
        {
            int sign = (half >> 15) & 0x1;
            int exp = (half >> 10) & 0x1F;
            int mantissa = half & 0x3FF;

            if (exp == 0)
            {
                if (mantissa == 0) return sign == 0 ? 0f : -0f;
                float m = mantissa / 1024f;
                return (sign == 0 ? 1f : -1f) * m * math.pow(2f, -14f);
            }
            else if (exp == 31)
            {
                return mantissa == 0 ? (sign == 0 ? float.PositiveInfinity : float.NegativeInfinity) : float.NaN;
            }

            float fraction = 1f + mantissa / 1024f;
            return (sign == 0 ? 1f : -1f) * fraction * math.pow(2f, exp - 15);
        }
    }

    // 이중 버퍼 (데이터 끊김 방지)
    public RGBDPoint[] PointBuffer;
    public int CurrentPointCount { get; private set; } = 0;
    private RGBDPoint[] _backupBuffer;
    private int _backupCount = 0;

    [SerializeField] private IEBoxer _ieBoxer;
    [SerializeField] private IEMasker _ieMasker;
    private Worker _inferenceEngineWorker;
    private IEnumerator _schedule;
    private InferenceDownloadState _downloadState = InferenceDownloadState.Running;
    private Tensor<float> _input;
    private bool _started = false;

    // Readback
    private readonly Tensor[] _outputBuffers = new Tensor[4];
    private readonly bool[] _readbackComplete = new bool[4];
    private bool _readbacksInitiated = false;
    private Tensor<float> _output0BoxCoords;
    private Tensor<int> _output1LabelIds;
    private Tensor<float> _output2Masks;
    private Tensor<float> _output3MaskWeights;

    private Texture2D _cpuDepthTex;
    private bool _isDepthReadingBack = false;
    private PassthroughCameraIntrinsics _cachedIntrinsics;
    private bool _intrinsicsCached = false;

    // Job System용 NativeArray
    private NativeArray<float3> _jobPositions;
    private NativeArray<float> _jobDepths;
    private NativeArray<int> _jobValid;
    private NativeArray<ushort> _jobDepthData;
    private NativeArray<float> _jobMaskData;
    private JobHandle _currentJobHandle;
    private bool _jobInProgress = false;

    // Depth 촬영 시점의 카메라 포즈 (시간차 보정용)
    private Vector3 _depthCameraPose;
    private Quaternion _depthCameraRot;
    private bool _hasValidDepthPose = false;

    private const float DEPTH_LATENCY_SECONDS = 0.033f; // ~33ms (약 1-2프레임)
    private Vector3 _prevCameraPos;
    private Quaternion _prevCameraRot;
    private bool _hasPrevPose = false;

    private bool _isTracking = false;
    private BoundingBox? _lockedTargetBox = null;
    private List<BoundingBox> _currentFrameBoxes = new List<BoundingBox>();

    // UI 복구용 텍스처
    private Texture2D _transparentBackground;

    public bool IsModelLoaded { get; private set; } = false;
    public bool IsTracking => _isTracking;
    public BoundingBox? LockedTargetBox => _lockedTargetBox;
    public List<BoundingBox> CurrentFrameBoxes => _currentFrameBoxes;

    private void Awake()
    {
        PointBuffer = new RGBDPoint[_maxPoints];
        _backupBuffer = new RGBDPoint[_maxPoints];

        if (_depthGradient == null)
        {
            _depthGradient = new Gradient();
            _depthGradient.SetKeys(
                new GradientColorKey[] { new GradientColorKey(Color.red, 0.0f), new GradientColorKey(Color.blue, 1.0f) },
                new GradientAlphaKey[] { new GradientAlphaKey(1.0f, 0.0f), new GradientAlphaKey(1.0f, 1.0f) }
            );
        }
    }

    private IEnumerator Start()
    {
        yield return new WaitForSeconds(0.1f);

        // UI 셋업 (캔버스가 켜져 있어도 투명하게 처리)
        SetupDisplay();

        if (_ieMasker != null) _ieMasker.Initialize(_displayLocation, _confidenceThreshold);
        LoadModel();

        // [안전장치] Depth Manager 모드 확인
        if (_depthManager != null && _depthManager.OcclusionShadersMode != OcclusionShadersMode.SoftOcclusion)
        {
            Debug.LogWarning("[IEExecutor] Switching EnvironmentDepthManager to SoftOcclusion for Point Cloud generation.");
            _depthManager.OcclusionShadersMode = OcclusionShadersMode.SoftOcclusion;
        }
    }

    private void SetupDisplay()
    {
        if (_displayLocation == null) return;

        var rawImage = _displayLocation.GetComponent<RawImage>();
        if (rawImage != null)
        {
            _transparentBackground = new Texture2D(_inputSize.x, _inputSize.y, TextureFormat.RGBA32, false);
            Color32[] clearPixels = new Color32[_inputSize.x * _inputSize.y];
            // 완전 투명
            for (int i = 0; i < clearPixels.Length; i++) clearPixels[i] = new Color32(0, 0, 0, 0);
            _transparentBackground.SetPixels32(clearPixels);
            _transparentBackground.Apply();

            rawImage.texture = _transparentBackground;
            rawImage.color = Color.white;
            rawImage.enabled = true;
        }
    }

    private void Update()
    {
        UpdateInference();
        PrepareDepthData();
    }

    private void OnDestroy()
    {
        if (_schedule != null) StopCoroutine(_schedule);
        _input?.Dispose();
        _inferenceEngineWorker?.Dispose();
        if (_cpuDepthTex != null) Destroy(_cpuDepthTex);
        if (_transparentBackground != null) Destroy(_transparentBackground);

        // Job 완료 대기 및 NativeArray 정리
        if (_jobInProgress)
        {
            _currentJobHandle.Complete();
        }
        DisposeJobArrays();

        CleanupResources();
    }

    private void PrepareDepthData()
    {
        if (_depthManager == null || !_depthManager.IsDepthAvailable || _isDepthReadingBack) return;
        var depthRT = Shader.GetGlobalTexture("_PreprocessedEnvironmentDepthTexture") as RenderTexture;
        if (depthRT == null) return;

        if (_cpuDepthTex == null || _cpuDepthTex.width != depthRT.width || _cpuDepthTex.height != depthRT.height)
        {
            if (_cpuDepthTex != null) Destroy(_cpuDepthTex);
            _cpuDepthTex = new Texture2D(depthRT.width, depthRT.height, TextureFormat.RHalf, false, true);
        }

        // 현재 카메라 포즈 가져오기
        Pose currentCameraPose = PassthroughCameraUtils.GetCameraPoseInWorld(PassthroughCameraEye.Left);

        // 시간차 보정
        if (_hasPrevPose)
        {
            float lerpFactor = Mathf.Clamp01(DEPTH_LATENCY_SECONDS / Time.deltaTime);
            _depthCameraPose = Vector3.Lerp(currentCameraPose.position, _prevCameraPos, lerpFactor);
            _depthCameraRot = Quaternion.Slerp(currentCameraPose.rotation, _prevCameraRot, lerpFactor);
            _hasValidDepthPose = true;
        }
        else
        {
            _depthCameraPose = currentCameraPose.position;
            _depthCameraRot = currentCameraPose.rotation;
            _hasValidDepthPose = true;
        }

        _prevCameraPos = currentCameraPose.position;
        _prevCameraRot = currentCameraPose.rotation;
        _hasPrevPose = true;

        _isDepthReadingBack = true;
        AsyncGPUReadback.Request(depthRT, 0, request => {
            if (request.hasError) { _isDepthReadingBack = false; return; }
            if (_cpuDepthTex != null)
            {
                _cpuDepthTex.LoadRawTextureData(request.GetData<ushort>());
                _cpuDepthTex.Apply();
            }
            _isDepthReadingBack = false;
        });
    }

    public void RunInference(Texture inputTexture)
    {
        if (!_started)
        {
            _input?.Dispose();
            if (!inputTexture) return;
            _inputSize = new Vector2Int(inputTexture.width, inputTexture.height);
            _input = TextureConverter.ToTensor(inputTexture, 640, 640, 3);
            _schedule = _inferenceEngineWorker.ScheduleIterable(_input);
            _downloadState = InferenceDownloadState.Running;
            _started = true;
            _readbacksInitiated = false;
        }
    }

    public bool IsRunning() => _started;

    private void LoadModel()
    {
        Model model = ModelLoader.Load(_sentisModel);
        _inferenceEngineWorker = new Worker(model, _backend);
        Tensor input = TextureConverter.ToTensor(new Texture2D(_inputSize.x, _inputSize.y), _inputSize.x, _inputSize.y, 3);
        _inferenceEngineWorker.Schedule(input);
        IsModelLoaded = true;
    }

    private void UpdateInference()
    {
        if (!_started) return;

        switch (_downloadState)
        {
            case InferenceDownloadState.Running:
                int it = 0;
                while (_schedule.MoveNext()) if (++it % _layersPerFrame == 0) return;
                _downloadState = InferenceDownloadState.RequestingOutputs;
                break;

            case InferenceDownloadState.RequestingOutputs:
                UpdateParallelReadbacks();
                break;

            case InferenceDownloadState.Success:
                ProcessInferenceResult();
                _downloadState = InferenceDownloadState.Cleanup;
                break;

            case InferenceDownloadState.Error:
            case InferenceDownloadState.Cleanup:
                CleanupResources();
                _downloadState = InferenceDownloadState.Completed;
                _started = false;
                break;
        }
    }

    private void UpdateParallelReadbacks()
    {
        if (!_readbacksInitiated)
        {
            for (int i = 0; i < 4; i++)
            {
                _readbackComplete[i] = false;
                _outputBuffers[i] = _inferenceEngineWorker.PeekOutput(i);
                if (_outputBuffers[i].dataOnBackend != null) _outputBuffers[i].ReadbackRequest();
                else { _downloadState = InferenceDownloadState.Error; return; }
            }
            _readbacksInitiated = true;
            return;
        }

        bool allComplete = true;
        for (int i = 0; i < 4; i++)
        {
            if (!_readbackComplete[i])
            {
                if (_outputBuffers[i].IsReadbackRequestDone()) _readbackComplete[i] = true;
                else allComplete = false;
            }
        }

        if (allComplete)
        {
            _output0BoxCoords = _outputBuffers[0].ReadbackAndClone() as Tensor<float>;
            _output1LabelIds = _outputBuffers[1].ReadbackAndClone() as Tensor<int>;
            _output2Masks = _outputBuffers[2].ReadbackAndClone() as Tensor<float>;
            _output3MaskWeights = _outputBuffers[3].ReadbackAndClone() as Tensor<float>;

            for (int i = 0; i < 4; i++) { _outputBuffers[i]?.Dispose(); _outputBuffers[i] = null; }

            _downloadState = (_output0BoxCoords != null && _output0BoxCoords.shape[0] > 0)
                ? InferenceDownloadState.Success : InferenceDownloadState.Error;
        }
    }

    private void ProcessInferenceResult()
    {
        int boxesFound = _output0BoxCoords.shape[0];
        float screenW = Screen.width;
        float screenH = Screen.height;

        _currentFrameBoxes.Clear();
        if (boxesFound > 0)
        {
            _currentFrameBoxes = ParseBoxes(_output0BoxCoords, _output1LabelIds, screenW, screenH);
        }

        // [Case 1] 트래킹 모드가 아닐 때
        if (!_isTracking)
        {
            if (EnableUIRendering)
            {
                _ieBoxer.DrawBoxes(_output0BoxCoords, _output1LabelIds, screenW, screenH);
            }
            else
            {
                _ieBoxer.HideAllBoxes();
                _ieMasker.ClearAllMasks();
            }
            return;
        }

        // [Case 2] 트래킹 모드 (특정 타겟 고정)
        if (_lockedTargetBox.HasValue)
        {
            int bestIndex = -1;
            float minDist = float.MaxValue;
            BoundingBox bestBox = default;

            for (int i = 0; i < _currentFrameBoxes.Count; i++)
            {
                var box = _currentFrameBoxes[i];
                if (box.ClassName != _lockedTargetBox.Value.ClassName) continue;

                float dist = Vector2.Distance(
                    new Vector2(box.CenterX, box.CenterY),
                    new Vector2(_lockedTargetBox.Value.CenterX, _lockedTargetBox.Value.CenterY));

                if (dist < minDist)
                {
                    minDist = dist;
                    bestIndex = i;
                    bestBox = box;
                }
            }

            if (bestIndex != -1 && minDist < 300f)
            {
                _lockedTargetBox = bestBox;
                _ieBoxer.HideAllBoxes();

                if (EnableUIRendering)
                {
                    _ieMasker.DrawSingleMask(bestIndex, bestBox, _output3MaskWeights, _inputSize.x, _inputSize.y);
                }
                else
                {
                    _ieMasker.ClearAllMasks();
                }

                // * Point Cloud는 무조건 추출
                ExtractDepthData(bestIndex, bestBox);
            }
        }
    }

    private List<BoundingBox> ParseBoxes(Tensor<float> output, Tensor<int> labelIds, float screenW, float screenH)
    {
        List<BoundingBox> boxes = new List<BoundingBox>();
        var scaleX = screenW / 640f;
        var scaleY = screenH / 640f;
        int count = Mathf.Min(output.shape[0], 50);

        for (int i = 0; i < count; i++)
        {
            float rawCenterX = output[i, 0];
            float rawCenterY = output[i, 1];
            float rawWidth = output[i, 2];
            float rawHeight = output[i, 3];

            float offsetX = (rawCenterX - 320f) * scaleX;
            float offsetY = (320f - rawCenterY) * scaleY;

            var classname = _ieBoxer.GetClassName(labelIds[i]);

            boxes.Add(new BoundingBox
            {
                CenterX = offsetX,
                CenterY = offsetY,
                ClassName = classname,
                Width = rawWidth * scaleX,
                Height = rawHeight * scaleY,
                Label = classname,
            });
        }
        return boxes;
    }

    private void ExtractDepthData(int targetIndex, BoundingBox box)
    {
        if (_cpuDepthTex == null || !_hasValidDepthPose) return;

        if (_jobInProgress)
        {
            _currentJobHandle.Complete();
            CollectJobResults();
        }

        if (!_intrinsicsCached)
        {
            _cachedIntrinsics = PassthroughCameraUtils.GetCameraIntrinsics(PassthroughCameraEye.Left);
            _intrinsicsCached = true;
        }

        var depthData = _cpuDepthTex.GetRawTextureData<ushort>();
        int depthW = _cpuDepthTex.width;
        int depthH = _cpuDepthTex.height;
        Vector2 sensorRes = _cachedIntrinsics.Resolution;

        float screenW = Screen.width;
        float screenH = Screen.height;

        float rawBoxCenterX = (box.CenterX / (screenW / 640f)) + 320f;
        float rawBoxCenterY = 320f - (box.CenterY / (screenH / 640f));
        float rawBoxWidth = box.Width / (screenW / 640f);
        float rawBoxHeight = box.Height / (screenH / 640f);

        const int maskW = 160;
        const int maskH = 160;
        int totalJobs = (maskW / _samplingStep) * (maskH / _samplingStep);

        if (!_jobPositions.IsCreated || _jobPositions.Length != totalJobs)
        {
            DisposeJobArrays();
            _jobPositions = new NativeArray<float3>(totalJobs, Allocator.Persistent);
            _jobDepths = new NativeArray<float>(totalJobs, Allocator.Persistent);
            _jobValid = new NativeArray<int>(totalJobs, Allocator.Persistent);
        }

        if (!_jobDepthData.IsCreated || _jobDepthData.Length != depthData.Length)
        {
            if (_jobDepthData.IsCreated) _jobDepthData.Dispose();
            _jobDepthData = new NativeArray<ushort>(depthData.Length, Allocator.Persistent);
        }
        NativeArray<ushort>.Copy(depthData.ToArray(), _jobDepthData);

        int maskDataSize = maskW * maskH;
        if (!_jobMaskData.IsCreated || _jobMaskData.Length != maskDataSize)
        {
            if (_jobMaskData.IsCreated) _jobMaskData.Dispose();
            _jobMaskData = new NativeArray<float>(maskDataSize, Allocator.Persistent);
        }
        for (int y = 0; y < maskH; y++)
        {
            for (int x = 0; x < maskW; x++)
            {
                _jobMaskData[y * maskW + x] = _output3MaskWeights[targetIndex, y, x];
            }
        }

        var job = new DepthExtractionJob
        {
            depthData = _jobDepthData,
            maskData = _jobMaskData,
            depthW = depthW,
            depthH = depthH,
            maskW = maskW,
            maskH = maskH,
            samplingStep = _samplingStep,
            confidenceThreshold = _confidenceThreshold,
            rawBoxCenterX = rawBoxCenterX,
            rawBoxCenterY = rawBoxCenterY,
            rawBoxWidth = rawBoxWidth,
            rawBoxHeight = rawBoxHeight,
            depthCameraPose = _depthCameraPose,
            depthCameraRot = new quaternion(_depthCameraRot.x, _depthCameraRot.y, _depthCameraRot.z, _depthCameraRot.w),
            focalLength = new float2(_cachedIntrinsics.FocalLength.x, _cachedIntrinsics.FocalLength.y),
            principalPoint = new float2(_cachedIntrinsics.PrincipalPoint.x, _cachedIntrinsics.PrincipalPoint.y),
            sensorRes = new float2(sensorRes.x, sensorRes.y),
            outPositions = _jobPositions,
            outDepths = _jobDepths,
            outValid = _jobValid
        };

        _currentJobHandle = job.Schedule(totalJobs, 64);
        _jobInProgress = true;
        _currentJobHandle.Complete();
        CollectJobResults();
    }

    private void CollectJobResults()
    {
        if (!_jobInProgress) return;

        int newPointCount = 0;
        for (int i = 0; i < _jobValid.Length && newPointCount < _maxPoints; i++)
        {
            if (_jobValid[i] == 1)
            {
                PointBuffer[newPointCount].worldPos = _jobPositions[i];
                float normalizedDepth = Mathf.Clamp01((_jobDepths[i] - 0.2f) / 2.0f);
                PointBuffer[newPointCount].color = _depthGradient.Evaluate(normalizedDepth);
                newPointCount++;
            }
        }

        _jobInProgress = false;

        if (newPointCount > 0)
        {
            Array.Copy(PointBuffer, _backupBuffer, newPointCount);
            _backupCount = newPointCount;
            CurrentPointCount = newPointCount;
        }
        else if (_backupCount > 0)
        {
            Array.Copy(_backupBuffer, PointBuffer, _backupCount);
            CurrentPointCount = _backupCount;
        }
    }

    private void DisposeJobArrays()
    {
        if (_jobPositions.IsCreated) _jobPositions.Dispose();
        if (_jobDepths.IsCreated) _jobDepths.Dispose();
        if (_jobValid.IsCreated) _jobValid.Dispose();
        if (_jobDepthData.IsCreated) _jobDepthData.Dispose();
        if (_jobMaskData.IsCreated) _jobMaskData.Dispose();
    }

    private void CleanupResources()
    {
        _output0BoxCoords?.Dispose();
        _output1LabelIds?.Dispose();
        _output2Masks?.Dispose();
        _output3MaskWeights?.Dispose();
        if (_transparentBackground != null) Destroy(_transparentBackground);
        _readbacksInitiated = false;
    }

    public void ResetTracking()
    {
        _isTracking = false;
        _lockedTargetBox = null;
        CurrentPointCount = 0;
        _backupCount = 0;
        _ieMasker.ClearAllMasks();
        _ieBoxer.HideAllBoxes();
        Debug.Log("[IEExecutor] Tracking Reset");
    }

    public void ClearPointCloud()
    {
        CurrentPointCount = 0;
        _backupCount = 0;
    }

    // 레이저가 가리키는 위치의 물체에 대해 Point Cloud 생성
    public void ExtractPointCloudAtScreenPos(Vector2 screenPos)
    {
        if (_currentFrameBoxes == null || _currentFrameBoxes.Count == 0) return;
        if (_output3MaskWeights == null) return;

        float halfScreenW = Screen.width / 2f;
        float halfScreenH = Screen.height / 2f;
        Vector2 screenPosCentered = new Vector2(screenPos.x - halfScreenW, screenPos.y - halfScreenH);

        // [수정] 변수 선언 확인
        float minDistance = float.MaxValue; 
        int bestIndex = -1;
        BoundingBox bestBox = default;

        for (int i = 0; i < _currentFrameBoxes.Count; i++)
        {
            var box = _currentFrameBoxes[i];
            float margin = 50f; // 터치 허용 범위
            if (screenPosCentered.x >= box.CenterX - box.Width / 2f - margin &&
                screenPosCentered.x <= box.CenterX + box.Width / 2f + margin &&
                screenPosCentered.y >= box.CenterY - box.Height / 2f - margin &&
                screenPosCentered.y <= box.CenterY + box.Height / 2f + margin)
            {
                float dist = Vector2.Distance(screenPosCentered, new Vector2(box.CenterX, box.CenterY));
                if (dist < minDistance)
                {
                    minDistance = dist;
                    bestIndex = i;
                    bestBox = box;
                }
            }
        }

        if (bestIndex != -1)
        {
            ExtractDepthData(bestIndex, bestBox);
        }
        else
        {
            // 해당 위치에 물체가 없으면 Point Cloud 클리어
            ClearPointCloud();
        }
    }

    /// <summary>
    /// 트리거(또는 버튼) 입력 시 호출하여 특정 타겟을 추적 시작
    /// </summary>
    public void SelectTargetFromScreenPos(Vector2 screenPos)
    {
        if (_currentFrameBoxes == null || _currentFrameBoxes.Count == 0) return;

        // [수정] 변수 선언 확인 (이 줄이 없어서 에러가 났을 수 있습니다)
        float minDistance = float.MaxValue; 
        BoundingBox? bestBox = null;
        
        float halfScreenW = Screen.width / 2f;
        float halfScreenH = Screen.height / 2f;
        Vector2 screenPosCentered = new Vector2(screenPos.x - halfScreenW, screenPos.y - halfScreenH);

        foreach (var box in _currentFrameBoxes)
        {
            float margin = 50f;
            if (screenPosCentered.x >= box.CenterX - box.Width/2f - margin &&
                screenPosCentered.x <= box.CenterX + box.Width/2f + margin &&
                screenPosCentered.y >= box.CenterY - box.Height/2f - margin &&
                screenPosCentered.y <= box.CenterY + box.Height/2f + margin)
            {
                float dist = Vector2.Distance(screenPosCentered, new Vector2(box.CenterX, box.CenterY));
                
                // 여기서 minDistance를 사용합니다.
                if (dist < minDistance)
                {
                    minDistance = dist;
                    bestBox = box;
                }
            }
        }

        if (bestBox.HasValue)
        {
            _lockedTargetBox = bestBox.Value;
            _isTracking = true;
            Debug.Log($"[IEExecutor] Target Selected: {bestBox.Value.ClassName}");
        }
    }
}