using System;
using System.Collections;
using System.Collections.Generic;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Rendering; 
using Meta.XR.EnvironmentDepth; 

public class IEExecutor : MonoBehaviour
{
    enum InferenceDownloadState
    {
        Running = 0,
        RequestingOutput0 = 1,
        RequestingOutput1 = 2,
        RequestingOutput2 = 3,
        RequestingOutput3 = 4,
        Success = 5,
        Error = 6,
        Cleanup = 7,
        Completed = 8
    }

    [Header("Model Settings")]
    [SerializeField] private Vector2Int _inputSize = new(640, 640);
    [SerializeField] private BackendType _backend = BackendType.GPUCompute; 
    [SerializeField] private ModelAsset _sentisModel;
    [SerializeField] private int _layersPerFrame = 12; 
    [SerializeField] private float _confidenceThreshold = 0.5f;
    [SerializeField] private Transform _displayLocation;

    [Header("Natural Tracking Settings")]
    [SerializeField] private int _maxLostFrames = 15;
    [SerializeField] private float _minIoUThreshold = 0.3f;
    [SerializeField] private float _minSmoothTime = 0.03f;
    [SerializeField] private float _maxSmoothTime = 0.2f; 
    [SerializeField] private float _sizeSmoothTime = 0.3f; 

    [Header("RGB-D & Depth Settings")]
    [SerializeField] private EnvironmentDepthManager _depthManager;
    [SerializeField] private PassthroughCameraSamples.WebCamTextureManager _webCamManager;
    [SerializeField] private bool _captureRGBD = true;
    [SerializeField] private int _maxPoints = 8000; 

    [Header("Optimization Settings")]
    [Range(2, 8)]
    [SerializeField] private int _samplingStep = 4; 

    public struct RGBDPoint {
        public Vector3 worldPos;
        public Color32 color;
    }
    public RGBDPoint[] PointBuffer; 
    public int CurrentPointCount { get; private set; }

    public bool IsModelLoaded { get; private set; } = false;

    [SerializeField] private IEBoxer _ieBoxer;

    private IEMasker _ieMasker;
    private Worker _inferenceEngineWorker;
    private IEnumerator _schedule;
    private InferenceDownloadState _downloadState = InferenceDownloadState.Running;

    private Tensor<float> _input;
    private Tensor _buffer;
    private Tensor<float> _output0BoxCoords;
    private Tensor<int> _output1LabelIds;
    private Tensor<float> _output2Masks;
    private Tensor<float> _output3MaskWeights;
    
    private Texture2D _cpuDepthTex;
    private bool _isDepthReadingBack = false;
    private Color32[] _rgbPixelCache;

    // [최적화] 카메라 intrinsics 캐싱 (한 번만 로드)
    private PassthroughCameraSamples.PassthroughCameraIntrinsics _cachedIntrinsics;
    private bool _intrinsicsCached = false;

    private BoundingBox? _lockedTargetBox = null;
    private bool _isTracking = false;
    private int _consecutiveLostFrames = 0;
    
    private Vector2 _centerVelocity;
    private Vector2 _sizeVelocity;
    private float _currentSmoothTime;

    private bool _started = false;
    private bool _isWaitingForReadbackRequest = false;
    private List<BoundingBox> _currentFrameBoxes = new();

    private void Awake()
    {
        PointBuffer = new RGBDPoint[_maxPoints];
    }

    private IEnumerator Start()
    {
        yield return new WaitForSeconds(0.05f);
        _ieMasker = new IEMasker(_displayLocation, _confidenceThreshold);
        LoadModel();
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
    }

    private void PrepareDepthData()
    {
        if (_depthManager == null || !_depthManager.IsDepthAvailable || _isDepthReadingBack) return;
        var depthRT = Shader.GetGlobalTexture("_PreprocessedEnvironmentDepthTexture") as RenderTexture;
        if (depthRT == null) return;

        if (_cpuDepthTex == null || _cpuDepthTex.width != depthRT.width)
            _cpuDepthTex = new Texture2D(depthRT.width, depthRT.height, TextureFormat.RHalf, false, true);

        _isDepthReadingBack = true;
        AsyncGPUReadback.Request(depthRT, 0, request => {
            if (request.hasError) { _isDepthReadingBack = false; return; }
            var data = request.GetData<ushort>();
            _cpuDepthTex.LoadRawTextureData(data);
            _cpuDepthTex.Apply();
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
        }
    }

    // [추가] 외부 트리거 스크립트 에러 해결을 위한 함수
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
        if (_downloadState == InferenceDownloadState.Running)
        {
            int it = 0;
            while (_schedule.MoveNext()) if (++it % _layersPerFrame == 0) return;
            _downloadState = InferenceDownloadState.RequestingOutput0;
        }
        else UpdateProcessInferenceResults();
    }

    private void UpdateProcessInferenceResults()
    {
        switch (_downloadState)
        {
            case InferenceDownloadState.RequestingOutput0: HandleReadback(0, ref _output0BoxCoords, ref _downloadState, InferenceDownloadState.RequestingOutput1); break;
            case InferenceDownloadState.RequestingOutput1: HandleReadback(1, ref _output1LabelIds, ref _downloadState, InferenceDownloadState.RequestingOutput2); break;
            case InferenceDownloadState.RequestingOutput2: HandleReadback(2, ref _output2Masks, ref _downloadState, InferenceDownloadState.RequestingOutput3); break;
            case InferenceDownloadState.RequestingOutput3: HandleReadback(3, ref _output3MaskWeights, ref _downloadState, InferenceDownloadState.Success); break;
            case InferenceDownloadState.Success: ProcessSuccessState(); _downloadState = InferenceDownloadState.Cleanup; break;
            case InferenceDownloadState.Error: _downloadState = InferenceDownloadState.Cleanup; break;
            case InferenceDownloadState.Cleanup: CleanupResources(); _downloadState = InferenceDownloadState.Completed; _started = false; break;
        }
    }

    private void HandleReadback<T>(int outputIndex, ref Tensor<T> targetTensor, ref InferenceDownloadState currentState, InferenceDownloadState nextState) where T : unmanaged
    {
        if (!_isWaitingForReadbackRequest) { _buffer = GetOutputBuffer(outputIndex); InitiateReadbackRequest(_buffer); }
        else if (_buffer.IsReadbackRequestDone())
        {
            targetTensor = _buffer.ReadbackAndClone() as Tensor<T>;
            _isWaitingForReadbackRequest = false;
            currentState = targetTensor.shape[0] > 0 ? nextState : InferenceDownloadState.Error;
            _buffer?.Dispose();
        }
    }

    private void ProcessSuccessState()
    {
        // 1. IEBoxer로부터 현재 프레임의 모든 박스 데이터를 가져옵니다.
        List<BoundingBox> currentFrameBoxes = _ieBoxer.DrawBoxes(_output0BoxCoords, _output1LabelIds, _inputSize.x, _inputSize.y);
        _currentFrameBoxes = currentFrameBoxes;

        if (_isTracking && _lockedTargetBox.HasValue)
        {
            // [박스 증식 해결] 추적 중일 때는 IEBoxer가 그린 모든 중복 박스들을 즉시 숨깁니다.
            _ieBoxer.ClearBoxes(0);

            float bestScore = 0f;
            int bestIndex = -1;
            BoundingBox bestBox = default;
            Vector2 prevCenter = new Vector2(_lockedTargetBox.Value.CenterX, _lockedTargetBox.Value.CenterY);

            // 2. 여러 중복된 박스 중 현재 타겟과 가장 잘 맞는 '베스트 하나'를 선택합니다.
            for (int i = 0; i < currentFrameBoxes.Count; i++)
            {
                BoundingBox currBox = currentFrameBoxes[i];
                float iou = TrackingUtils.CalculateIoU(_lockedTargetBox.Value, currBox);
                
                if (iou < _minIoUThreshold) continue;

                float dist = Vector2.Distance(prevCenter, new Vector2(currBox.CenterX, currBox.CenterY));
                float distScore = 1.0f - Mathf.Clamp01(dist / (_inputSize.x * 0.5f));
                float totalScore = (iou * 0.7f) + (distScore * 0.3f);

                if (totalScore > bestScore)
                {
                    bestScore = totalScore;
                    bestIndex = i;
                    bestBox = currBox;
                }
            }

            if (bestIndex != -1)
            {
                _consecutiveLostFrames = 0;
                UpdateSmoothBox(bestBox);
                
                // [해결] 마스크와 시각적 데이터는 오직 베스트 인덱스 하나만 사용합니다.
                _ieMasker.DrawSingleMask(bestIndex, bestBox, _output3MaskWeights, _inputSize.x, _inputSize.y);
                
                if (_captureRGBD) ExtractRGBDData(bestIndex, bestBox);
            }
            else
            {
                _consecutiveLostFrames++;
                if (_consecutiveLostFrames <= _maxLostFrames) PredictBoxMovement();
                else ResetTracking();
            }
        }
        else
        {
            _ieMasker.DrawSingleMask(-1, default, null, _inputSize.x, _inputSize.y);
            CurrentPointCount = 0;
        }
    }

    private void ExtractRGBDData(int targetIndex, BoundingBox box)
    {
        if (_webCamManager == null || _cpuDepthTex == null) return;
        WebCamTexture rgbTex = _webCamManager.WebCamTexture;
        if (rgbTex == null || rgbTex.width < 100) return;

        // [최적화] 카메라 intrinsics 캐싱 (한 번만 로드)
        if (!_intrinsicsCached)
        {
            _cachedIntrinsics = PassthroughCameraSamples.PassthroughCameraUtils.GetCameraIntrinsics(
                PassthroughCameraSamples.PassthroughCameraEye.Left);
            _intrinsicsCached = true;
        }

        // [최적화] 카메라 pose는 프레임당 한 번만 가져오기
        Pose cameraPose = PassthroughCameraSamples.PassthroughCameraUtils.GetCameraPoseInWorld(
            PassthroughCameraSamples.PassthroughCameraEye.Left);
        Vector3 cameraPos = cameraPose.position;
        Quaternion cameraRot = cameraPose.rotation;

        // intrinsics 캐시에서 값 추출
        float fx = _cachedIntrinsics.FocalLength.x;
        float fy = _cachedIntrinsics.FocalLength.y;
        float cx = _cachedIntrinsics.PrincipalPoint.x;
        float cy = _cachedIntrinsics.PrincipalPoint.y;

        if (_rgbPixelCache == null || _rgbPixelCache.Length != rgbTex.width * rgbTex.height)
            _rgbPixelCache = new Color32[rgbTex.width * rgbTex.height];

        rgbTex.GetPixels32(_rgbPixelCache);

        var depthData = _cpuDepthTex.GetRawTextureData<ushort>();
        int rgbW = rgbTex.width; int rgbH = rgbTex.height;
        int depthW = _cpuDepthTex.width; int depthH = _cpuDepthTex.height;

        CurrentPointCount = 0;
        for (int y = 0; y < 160; y += _samplingStep)
        {
            for (int x = 0; x < 160; x += _samplingStep)
            {
                if (CurrentPointCount >= _maxPoints) break;
                float maskVal = _output3MaskWeights[targetIndex, y, x];
                if (maskVal > _confidenceThreshold)
                {
                    Vector2Int rgbPixel = MapMaskToRGBPixel(x, y, box, rgbW, rgbH);
                    int pixelIdx = (rgbH - 1 - rgbPixel.y) * rgbW + rgbPixel.x;
                    if (pixelIdx < 0 || pixelIdx >= _rgbPixelCache.Length) continue;

                    float u = (float)rgbPixel.x / rgbW;
                    float v = (float)rgbPixel.y / rgbH;
                    int dx = Mathf.FloorToInt(u * (depthW - 1));
                    int dy = Mathf.FloorToInt(v * (depthH - 1));
                    float depthMeters = Mathf.HalfToFloat(depthData[dy * depthW + dx]);

                    if (depthMeters > 0.1f && depthMeters < 3.0f)
                    {
                        // [최적화] 레이 방향 인라인 계산 (함수 호출 제거)
                        Vector3 dirInCamera = new Vector3(
                            (rgbPixel.x - cx) / fx,
                            (rgbPixel.y - cy) / fy,
                            1f
                        );
                        Vector3 dirInWorld = cameraRot * dirInCamera;
                        Vector3 worldPos = cameraPos + dirInWorld.normalized * depthMeters;

                        PointBuffer[CurrentPointCount].worldPos = worldPos;
                        PointBuffer[CurrentPointCount].color = _rgbPixelCache[pixelIdx];
                        CurrentPointCount++;
                    }
                }
            }
        }
    }

    private Vector2Int MapMaskToRGBPixel(int maskX, int maskY, BoundingBox box, int rgbW, int rgbH)
    {
        float normX = (float)maskX / 160f;
        float normY = (float)maskY / 160f;
        int pixelX = Mathf.RoundToInt((box.CenterX - box.Width/2f + normX * box.Width) + rgbW/2f);
        int pixelY = Mathf.RoundToInt(rgbH/2f - (box.CenterY - box.Height/2f + normY * box.Height));
        return new Vector2Int(Mathf.Clamp(pixelX, 0, rgbW - 1), Mathf.Clamp(pixelY, 0, rgbH - 1));
    }

    private void UpdateSmoothBox(BoundingBox bestBox)
    {
        float speed = _centerVelocity.magnitude;
        float targetSmoothTime = Mathf.Lerp(_maxSmoothTime, _minSmoothTime, speed / 500f);
        _currentSmoothTime = Mathf.Lerp(_currentSmoothTime, targetSmoothTime, 0.1f);
        Vector2 currentPos = new Vector2(_lockedTargetBox.Value.CenterX, _lockedTargetBox.Value.CenterY);
        Vector2 targetPos = new Vector2(bestBox.CenterX, bestBox.CenterY);
        Vector2 smoothedPos = Vector2.SmoothDamp(currentPos, targetPos, ref _centerVelocity, _currentSmoothTime);
        Vector2 currentSize = new Vector2(_lockedTargetBox.Value.Width, _lockedTargetBox.Value.Height);
        Vector2 targetSize = new Vector2(bestBox.Width, bestBox.Height);
        Vector2 smoothedSize = Vector2.SmoothDamp(currentSize, targetSize, ref _sizeVelocity, _sizeSmoothTime);
        _lockedTargetBox = new BoundingBox { CenterX = smoothedPos.x, CenterY = smoothedPos.y, Width = smoothedSize.x, Height = smoothedSize.y, ClassName = bestBox.ClassName, Label = bestBox.Label, WorldPos = bestBox.WorldPos };
    }

    private void PredictBoxMovement()
    {
        float deltaTime = Time.deltaTime; 
        float predX = _lockedTargetBox.Value.CenterX + (_centerVelocity.x * deltaTime);
        float predY = _lockedTargetBox.Value.CenterY + (_centerVelocity.y * deltaTime);
        _centerVelocity *= 0.9f; 
        _lockedTargetBox = new BoundingBox { CenterX = predX, CenterY = predY, Width = _lockedTargetBox.Value.Width, Height = _lockedTargetBox.Value.Height, ClassName = _lockedTargetBox.Value.ClassName, Label = _lockedTargetBox.Value.Label, WorldPos = _lockedTargetBox.Value.WorldPos };
    }

    private void CleanupResources() { _output0BoxCoords?.Dispose(); _output1LabelIds?.Dispose(); _output2Masks?.Dispose(); _output3MaskWeights?.Dispose(); }

    public void SelectTargetFromScreenPos(Vector2 screenPos)
    {
        if (_currentFrameBoxes == null || _currentFrameBoxes.Count == 0) return;
        float minDistance = float.MaxValue; BoundingBox? bestBox = null;
        float halfScreenW = Screen.width / 2f; float halfScreenH = Screen.height / 2f;
        foreach (var box in _currentFrameBoxes)
        {
            float boxScreenX = box.CenterX + halfScreenW; float boxScreenY = halfScreenH - box.CenterY; float margin = 30f; 
            if (screenPos.x >= boxScreenX - box.Width/2f - margin && screenPos.x <= boxScreenX + box.Width/2f + margin && screenPos.y >= boxScreenY - box.Height/2f - margin && screenPos.y <= boxScreenY + box.Height/2f + margin)
            {
                float dist = Vector2.Distance(screenPos, new Vector2(boxScreenX, boxScreenY));
                if (dist < minDistance) { minDistance = dist; bestBox = box; }
            }
        }
        if (bestBox.HasValue) { _lockedTargetBox = bestBox.Value; _isTracking = true; _consecutiveLostFrames = 0; _centerVelocity = Vector2.zero; _sizeVelocity = Vector2.zero; _currentSmoothTime = _maxSmoothTime; Debug.Log($"[IEExecutor] Target Selected: {bestBox.Value.ClassName}"); }
    }

    public void ResetTracking() { _isTracking = false; _lockedTargetBox = null; _consecutiveLostFrames = 0; if (_ieMasker != null) _ieMasker.DrawSingleMask(-1, default, null, _inputSize.x, _inputSize.y); CurrentPointCount = 0; Debug.Log("[IEExecutor] Tracking Reset"); }
    private Tensor GetOutputBuffer(int outputIndex) => _inferenceEngineWorker.PeekOutput(outputIndex);
    private void InitiateReadbackRequest(Tensor pullTensor) { if (pullTensor.dataOnBackend != null) { pullTensor.ReadbackRequest(); _isWaitingForReadbackRequest = true; } else _downloadState = InferenceDownloadState.Error; }
}