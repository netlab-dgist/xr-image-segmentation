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

    // 외부 공개용 프로퍼티 복원
    public IEMasker Masker => _ieMasker;
    public Vector2Int InputSize => _inputSize;
    public bool IsTracking => _isTracking;
    public BoundingBox? LockedTargetBox => _lockedTargetBox;

    [SerializeField] private IEBoxer _ieBoxer;

    // [수정] _ieMasker는 MonoBehaviour가 아니므로 SerializeField 제거하고 new로 생성
    private IEMasker _ieMasker;
    private Worker _inferenceEngineWorker;
    private IEnumerator _schedule;
    private InferenceDownloadState _downloadState = InferenceDownloadState.Running;

    private Tensor<float> _input;

    // [최적화] 병렬 Readback을 위한 버퍼 배열
    private readonly Tensor[] _outputBuffers = new Tensor[4];
    private readonly bool[] _readbackComplete = new bool[4];
    private Tensor<float> _output0BoxCoords;
    private Tensor<int> _output1LabelIds;
    private Tensor<float> _output2Masks;
    private Tensor<float> _output3MaskWeights;

    private Texture2D _cpuDepthTex;
    private bool _isDepthReadingBack = false;

    // [최적화] RGB 비동기 읽기를 위한 변수들
    private Color32[] _rgbPixelCache;
    private RenderTexture _rgbRenderTexture;
    private Texture2D _rgbCpuTexture;
    private bool _isRgbReadingBack = false;
    private bool _rgbDataReady = false;

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
    private bool _readbacksInitiated = false;
    private readonly List<BoundingBox> _currentFrameBoxes = new();

    private void Awake()
    {
        PointBuffer = new RGBDPoint[_maxPoints];
    }

    private IEnumerator Start()
    {
        yield return new WaitForSeconds(0.05f);
        // [수정] Initialize 대신 생성자 사용
        _ieMasker = new IEMasker(_displayLocation, _confidenceThreshold);
        
        LoadModel();
    }

    private void Update()
    {
        UpdateInference();
        PrepareDepthData();
        PrepareRgbData();
    }

    private void OnDestroy()
    {
        if (_schedule != null) StopCoroutine(_schedule);
        _input?.Dispose();
        _inferenceEngineWorker?.Dispose();
        if (_cpuDepthTex != null) Destroy(_cpuDepthTex);
        if (_rgbRenderTexture != null) _rgbRenderTexture.Release();
        if (_rgbCpuTexture != null) Destroy(_rgbCpuTexture);
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

    /// <summary>
    /// [최적화] RGB 텍스처를 비동기로 읽기 (GetPixels32 대체)
    /// </summary>
    private void PrepareRgbData()
    {
        if (_webCamManager == null || _isRgbReadingBack) return;
        WebCamTexture rgbTex = _webCamManager.WebCamTexture;
        if (rgbTex == null || rgbTex.width < 100) return;

        // RenderTexture 초기화 (필요시)
        if (_rgbRenderTexture == null || _rgbRenderTexture.width != rgbTex.width)
        {
            if (_rgbRenderTexture != null) _rgbRenderTexture.Release();
            _rgbRenderTexture = new RenderTexture(rgbTex.width, rgbTex.height, 0, RenderTextureFormat.ARGB32);
            _rgbCpuTexture = new Texture2D(rgbTex.width, rgbTex.height, TextureFormat.RGBA32, false);
            _rgbPixelCache = new Color32[rgbTex.width * rgbTex.height];
        }

        // GPU에서 복사 (빠름)
        Graphics.Blit(rgbTex, _rgbRenderTexture);

        _isRgbReadingBack = true;
        AsyncGPUReadback.Request(_rgbRenderTexture, 0, TextureFormat.RGBA32, request => {
            if (request.hasError) { _isRgbReadingBack = false; return; }
            var data = request.GetData<Color32>();
            data.CopyTo(_rgbPixelCache);
            _rgbDataReady = true;
            _isRgbReadingBack = false;
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
                ProcessSuccessState();
                _downloadState = InferenceDownloadState.Cleanup;
                break;

            case InferenceDownloadState.Error:
                _downloadState = InferenceDownloadState.Cleanup;
                break;

            case InferenceDownloadState.Cleanup:
                CleanupResources();
                _downloadState = InferenceDownloadState.Completed;
                _started = false;
                break;
        }
    }

    /// <summary>
    /// [최적화] 4개의 출력을 병렬로 Readback 요청
    /// </summary>
    private void UpdateParallelReadbacks()
    {
        // 첫 번째 호출: 모든 readback 요청을 동시에 시작
        if (!_readbacksInitiated)
        {
            for (int i = 0; i < 4; i++)
            {
                _readbackComplete[i] = false;
                _outputBuffers[i] = _inferenceEngineWorker.PeekOutput(i);

                if (_outputBuffers[i].dataOnBackend != null)
                {
                    _outputBuffers[i].ReadbackRequest();
                }
                else
                {
                    _downloadState = InferenceDownloadState.Error;
                    return;
                }
            }
            _readbacksInitiated = true;
            return;
        }

        // 이후 호출: 모든 readback이 완료되었는지 확인
        bool allComplete = true;
        for (int i = 0; i < 4; i++)
        {
            if (!_readbackComplete[i])
            {
                if (_outputBuffers[i].IsReadbackRequestDone())
                {
                    _readbackComplete[i] = true;
                }
                else
                {
                    allComplete = false;
                }
            }
        }

        // 모든 readback이 완료되면 텐서 복제
        if (allComplete)
        {
            _output0BoxCoords = _outputBuffers[0].ReadbackAndClone() as Tensor<float>;
            _output1LabelIds = _outputBuffers[1].ReadbackAndClone() as Tensor<int>;
            _output2Masks = _outputBuffers[2].ReadbackAndClone() as Tensor<float>;
            _output3MaskWeights = _outputBuffers[3].ReadbackAndClone() as Tensor<float>;

            // 버퍼 정리
            for (int i = 0; i < 4; i++)
            {
                _outputBuffers[i]?.Dispose();
                _outputBuffers[i] = null;
            }

            _downloadState = (_output0BoxCoords != null && _output0BoxCoords.shape[0] > 0)
                ? InferenceDownloadState.Success
                : InferenceDownloadState.Error;
        }
    }

    private void ProcessSuccessState()
    {
        List<BoundingBox> currentFrameBoxes;

        if (_isTracking && _lockedTargetBox.HasValue)
        {
            currentFrameBoxes = ParseBoxesWithoutDraw(_output0BoxCoords, _output1LabelIds, _inputSize.x, _inputSize.y);
            _currentFrameBoxes.Clear();
            _currentFrameBoxes.AddRange(currentFrameBoxes);

            _ieBoxer.HideAllBoxes();

            float bestScore = 0f;
            int bestIndex = -1;
            BoundingBox bestBox = default;
            Vector2 prevCenter = new Vector2(_lockedTargetBox.Value.CenterX, _lockedTargetBox.Value.CenterY);

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
                _ieMasker.DrawSingleMask(bestIndex, bestBox, _output3MaskWeights, _inputSize.x, _inputSize.y);

                if (_captureRGBD) ExtractRGBDData(bestIndex, bestBox);
            }
            else
            {
                _consecutiveLostFrames++;
                if (_consecutiveLostFrames <= _maxLostFrames)
                {
                    PredictBoxMovement();
                    _ieMasker.KeepCurrentMask();
                }
                else
                {
                    ResetTracking();
                }
            }
        }
        else
        {
            currentFrameBoxes = _ieBoxer.DrawBoxes(_output0BoxCoords, _output1LabelIds, _inputSize.x, _inputSize.y);
            _currentFrameBoxes.Clear();
            _currentFrameBoxes.AddRange(currentFrameBoxes);
            _ieMasker.ClearAllMasks();
            CurrentPointCount = 0;
        }
    }

    private List<BoundingBox> ParseBoxesWithoutDraw(Tensor<float> output, Tensor<int> labelIds, float imageWidth, float imageHeight)
    {
        List<BoundingBox> boundingBoxes = new();

        var scaleX = imageWidth / 640;
        var scaleY = imageHeight / 640;
        var halfWidth = imageWidth / 2;
        var halfHeight = imageHeight / 2;

        int boxesFound = output.shape[0];
        if (boxesFound <= 0) return boundingBoxes;

        var maxBoxes = Mathf.Min(boxesFound, 200);

        for (var n = 0; n < maxBoxes; n++)
        {
            var centerX = output[n, 0] * scaleX - halfWidth;
            var centerY = output[n, 1] * scaleY - halfHeight;
            var classname = _ieBoxer.GetClassName(labelIds[n]);

            var box = new BoundingBox
            {
                CenterX = centerX,
                CenterY = centerY,
                ClassName = classname,
                Width = output[n, 2] * scaleX,
                Height = output[n, 3] * scaleY,
                Label = classname,
            };
            boundingBoxes.Add(box);
        }
        return boundingBoxes;
    }

    /// <summary>
    /// [최적화] 비동기로 읽은 RGB 데이터 사용
    /// </summary>
    private void ExtractRGBDData(int targetIndex, BoundingBox box)
    {
        // 비동기 RGB 데이터가 준비되지 않았으면 스킵
        if (!_rgbDataReady || _cpuDepthTex == null) return;
        if (_rgbPixelCache == null || _rgbPixelCache.Length == 0) return;

        // [최적화] 카메라 intrinsics 캐싱 (한 번만 로드)
        if (!_intrinsicsCached)
        {
            _cachedIntrinsics = PassthroughCameraSamples.PassthroughCameraUtils.GetCameraIntrinsics(
                PassthroughCameraSamples.PassthroughCameraEye.Left);
            _intrinsicsCached = true;
        }

        Pose cameraPose = PassthroughCameraSamples.PassthroughCameraUtils.GetCameraPoseInWorld(
            PassthroughCameraSamples.PassthroughCameraEye.Left);
        Vector3 cameraPos = cameraPose.position;
        Quaternion cameraRot = cameraPose.rotation;

        float fx = _cachedIntrinsics.FocalLength.x;
        float fy = _cachedIntrinsics.FocalLength.y;
        float cx = _cachedIntrinsics.PrincipalPoint.x;
        float cy = _cachedIntrinsics.PrincipalPoint.y;

        var depthData = _cpuDepthTex.GetRawTextureData<ushort>();
        int rgbW = _rgbRenderTexture.width;
        int rgbH = _rgbRenderTexture.height;
        int depthW = _cpuDepthTex.width;
        int depthH = _cpuDepthTex.height;

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
                    
                    // [Fix] Removed (rgbH - 1 - ...) inversion to match Unity Bottom-Up coords
                    int pixelIdx = rgbPixel.y * rgbW + rgbPixel.x;
                    
                    if (pixelIdx < 0 || pixelIdx >= _rgbPixelCache.Length) continue;

                    float u = (float)rgbPixel.x / rgbW;
                    float v = (float)rgbPixel.y / rgbH;
                    int dx = Mathf.FloorToInt(u * (depthW - 1));
                    int dy = Mathf.FloorToInt(v * (depthH - 1));
                    float depthMeters = Mathf.HalfToFloat(depthData[dy * depthW + dx]);

                    if (depthMeters > 0.1f && depthMeters < 3.0f)
                    {
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

    private void CleanupResources()
    {
        _output0BoxCoords?.Dispose();
        _output1LabelIds?.Dispose();
        _output2Masks?.Dispose();
        _output3MaskWeights?.Dispose();
        _readbacksInitiated = false;
    }

    public void SelectTargetFromScreenPos(Vector2 screenPos)
    {
        if (_currentFrameBoxes == null || _currentFrameBoxes.Count == 0) return;
        float minDistance = float.MaxValue;
        BoundingBox? bestBox = null;
        float halfScreenW = Screen.width / 2f;
        float halfScreenH = Screen.height / 2f;

        foreach (var box in _currentFrameBoxes)
        {
            float boxScreenX = box.CenterX + halfScreenW;
            float boxScreenY = halfScreenH - box.CenterY;
            float margin = 30f;

            if (screenPos.x >= boxScreenX - box.Width/2f - margin &&
                screenPos.x <= boxScreenX + box.Width/2f + margin &&
                screenPos.y >= boxScreenY - box.Height/2f - margin &&
                screenPos.y <= boxScreenY + box.Height/2f + margin)
            {
                float dist = Vector2.Distance(screenPos, new Vector2(boxScreenX, boxScreenY));
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
            _consecutiveLostFrames = 0;
            _centerVelocity = Vector2.zero;
            _sizeVelocity = Vector2.zero;
            _currentSmoothTime = _maxSmoothTime;
            Debug.Log($"[IEExecutor] Target Selected: {bestBox.Value.ClassName}");
        }
    }

    public void ResetTracking()
    {
        _isTracking = false;
        _lockedTargetBox = null;
        _consecutiveLostFrames = 0;
        if (_ieMasker != null) _ieMasker.ClearAllMasks();
        CurrentPointCount = 0;
        Debug.Log("[IEExecutor] Tracking Reset");
    }
}