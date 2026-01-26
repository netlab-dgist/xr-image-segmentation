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
        Running = 0, RequestingOutputs = 1, Success = 2, Error = 3, Cleanup = 4, Completed = 5
    }

    [Header("Model Settings")]
    [SerializeField] private Vector2Int _inputSize = new(640, 640);
    [SerializeField] private BackendType _backend = BackendType.GPUCompute;
    [SerializeField] private ModelAsset _sentisModel;
    [SerializeField] private int _layersPerFrame = 12;
    [SerializeField] private float _confidenceThreshold = 0.3f;
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
    [SerializeField] private int _maxPoints = 8000;
    [SerializeField] private PointCloudRenderer _pointCloudRenderer;
    [SerializeField] private bool _flipDepthSampling = false;
    [Tooltip("Points closer than this (meters) are ignored to prevent spikes.")]
    [SerializeField] private float _minDepth = 0.3f; 
    [Tooltip("Points further than this (meters) are ignored.")]
    [SerializeField] private float _maxDepth = 5.0f;

    [Header("Optimization Settings")]
    [Range(1, 8)]
    [SerializeField] private int _samplingStep = 1;

    public bool IsModelLoaded { get; private set; } = false;
    public bool IsTracking => _isTracking;
    public BoundingBox? LockedTargetBox => _lockedTargetBox;
    public Vector2Int InputSize => _inputSize;
    public IEMasker Masker => _ieMasker;

    [SerializeField] private IEBoxer _ieBoxer;
    private IEMasker _ieMasker;
    private Worker _inferenceEngineWorker;
    private IEnumerator _schedule;
    private InferenceDownloadState _downloadState = InferenceDownloadState.Running;

    private Tensor<float> _input;
    private readonly Tensor[] _outputBuffers = new Tensor[4];
    private readonly bool[] _readbackComplete = new bool[4];
    
    private Tensor<float> _output0BoxCoords;
    private Tensor<int> _output1LabelIds;
    private Tensor<float> _output2Masks;
    private Tensor<float> _output3MaskWeights;

    private float[] _latestBoxData;
    private float[] _latestMaskWeights;
    private int _latestBoxCount;

    private Texture2D _cpuDepthTex;
    private Color32[] _rgbPixelCache;
    private RenderTexture _rgbRenderTexture;
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

    private void Start()
    {
        _ieMasker = new IEMasker(_displayLocation, _confidenceThreshold);
        LoadModel();
        if (_pointCloudRenderer == null) _pointCloudRenderer = gameObject.GetComponent<PointCloudRenderer>() ?? gameObject.AddComponent<PointCloudRenderer>();
    }

    private void Update()
    {
        UpdateInference();
    }

    private void OnDestroy()
    {
        if (_schedule != null) StopCoroutine(_schedule);
        _input?.Dispose();
        _inferenceEngineWorker?.Dispose();
        CleanupResources();
        if (_cpuDepthTex != null) Destroy(_cpuDepthTex);
        if (_rgbRenderTexture != null) _rgbRenderTexture.Release();
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
        Tensor input = TextureConverter.ToTensor(new Texture2D(640, 640), 640, 640, 3);
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
            if (!_readbackComplete[i] && _outputBuffers[i].IsReadbackRequestDone()) _readbackComplete[i] = true;
            if (!_readbackComplete[i]) allComplete = false;
        }

        if (allComplete)
        {
            _output0BoxCoords = _outputBuffers[0].ReadbackAndClone() as Tensor<float>;
            _output1LabelIds = _outputBuffers[1].ReadbackAndClone() as Tensor<int>;
            _output2Masks = _outputBuffers[2].ReadbackAndClone() as Tensor<float>;
            _output3MaskWeights = _outputBuffers[3].ReadbackAndClone() as Tensor<float>;

            for (int i = 0; i < 4; i++) { _outputBuffers[i]?.Dispose(); _outputBuffers[i] = null; }

            _downloadState = (_output0BoxCoords != null && _output0BoxCoords.shape[0] > 0) ? InferenceDownloadState.Success : InferenceDownloadState.Error;
        }
    }

    private void ProcessSuccessState()
    {
        try
        {
            if (_output0BoxCoords != null && _output3MaskWeights != null)
            {
                _latestBoxData = _output0BoxCoords.DownloadToArray();
                _latestMaskWeights = _output3MaskWeights.DownloadToArray();
                _latestBoxCount = _output0BoxCoords.shape[0];
            }
        }
        catch (Exception e) { Debug.LogWarning($"[IEExecutor] Failed to cache: {e.Message}"); }

        List<BoundingBox> currentFrameBoxes = _ieBoxer.DrawBoxes(_output0BoxCoords, _output1LabelIds, _inputSize.x, _inputSize.y);
        _currentFrameBoxes.Clear();
        _currentFrameBoxes.AddRange(currentFrameBoxes);

        if (_isTracking && _lockedTargetBox.HasValue)
        {
            _ieBoxer.HideAllBoxes();
            float bestScore = 0f; int bestIndex = -1; BoundingBox bestBox = default;
            Vector2 prevCenter = new Vector2(_lockedTargetBox.Value.CenterX, _lockedTargetBox.Value.CenterY);

            for (int i = 0; i < currentFrameBoxes.Count; i++)
            {
                BoundingBox currBox = currentFrameBoxes[i];
                float iou = TrackingUtils.CalculateIoU(_lockedTargetBox.Value, currBox);
                if (iou < _minIoUThreshold) continue;
                float dist = Vector2.Distance(prevCenter, new Vector2(currBox.CenterX, currBox.CenterY));
                float distScore = 1.0f - Mathf.Clamp01(dist / (_inputSize.x * 0.5f));
                float totalScore = (iou * 0.7f) + (distScore * 0.3f);
                if (totalScore > bestScore) { bestScore = totalScore; bestIndex = i; bestBox = currBox; }
            }

            if (bestIndex != -1)
            {
                _consecutiveLostFrames = 0;
                UpdateSmoothBox(bestBox);
                _ieMasker.DrawSingleMask(bestIndex, bestBox, _output3MaskWeights, _inputSize.x, _inputSize.y);
            }
            else
            {
                _consecutiveLostFrames++;
                if (_consecutiveLostFrames <= _maxLostFrames) { PredictBoxMovement(); _ieMasker.KeepCurrentMask(); }
                else ResetTracking();
            }
        }
        else
        {
            _ieMasker.ClearAllMasks();
        }
    }

    public void CaptureSnapshot()
    {
        if (!_isTracking || !_lockedTargetBox.HasValue) return;
        if (!gameObject.activeInHierarchy) gameObject.SetActive(true);
        if (!enabled) enabled = true;

        if (_latestBoxData == null || _latestMaskWeights == null)
        {
            Debug.LogWarning("[Snapshot] No cached inference data.");
            return;
        }
        Debug.Log($"[Snapshot] Cached BoxCount: {_latestBoxCount}");
        StartCoroutine(CaptureRoutine());
    }

    private IEnumerator CaptureRoutine()
    {
        Debug.Log("[Snapshot] Starting...");
        yield return PrepareDepthDataAsync();
        yield return PrepareRgbDataAsync();

        if (_cpuDepthTex == null || _rgbPixelCache == null) { Debug.LogError("[Snapshot] Data Missing."); yield break; }

        int targetIndex = FindBestMatchIndexFromCache();
        if (targetIndex == -1) { Debug.LogWarning("[Snapshot] No matching object found."); yield break; }

        Debug.Log($"[Snapshot] Extracting grid for Mesh...");
        
        int gridW = 160 / _samplingStep;
        int gridH = 160 / _samplingStep;
        Vector3[] worldGrid = new Vector3[gridW * gridH];
        Color32[] colorGrid = new Color32[gridW * gridH];
        bool[] validGrid = new bool[gridW * gridH];
        Vector3 centerPos = Vector3.zero;
        int validCount = 0;

        ExtractGridFromCache(targetIndex, _lockedTargetBox.Value, worldGrid, colorGrid, validGrid, ref centerPos, ref validCount);

        if (validCount > 0) 
        {
            centerPos /= validCount; 
            _pointCloudRenderer.GenerateTriangleMesh(worldGrid, colorGrid, validGrid, gridW, gridH, centerPos);
            Debug.Log($"[Snapshot] Mesh Generated! Verts: {validCount}");
        }
        else Debug.LogWarning("[Snapshot] No valid points extracted.");
    }

    private void ExtractGridFromCache(int targetIndex, BoundingBox box, Vector3[] worldGrid, Color32[] colorGrid, bool[] validGrid, ref Vector3 centerPosSum, ref int validCount)
    {
        var depthData = _cpuDepthTex.GetRawTextureData<ushort>();
        int rgbW = _rgbRenderTexture.width;
        int rgbH = _rgbRenderTexture.height;
        int depthW = _cpuDepthTex.width;
        int depthH = _cpuDepthTex.height;

        // Coordinate Scale Correction
        float scaleX = (float)rgbW / _inputSize.x;
        float scaleY = (float)rgbH / _inputSize.y;

        BoundingBox rgbBox = new BoundingBox
        {
            CenterX = box.CenterX * scaleX,
            CenterY = box.CenterY * scaleY,
            Width = box.Width * scaleX,
            Height = box.Height * scaleY
        };

        if (!_intrinsicsCached)
        {
            _cachedIntrinsics = PassthroughCameraSamples.PassthroughCameraUtils.GetCameraIntrinsics(PassthroughCameraSamples.PassthroughCameraEye.Left);
            _intrinsicsCached = true;
        }

        float fx = _cachedIntrinsics.FocalLength.x;
        float fy = _cachedIntrinsics.FocalLength.y;
        float cx = _cachedIntrinsics.PrincipalPoint.x;
        float cy = _cachedIntrinsics.PrincipalPoint.y;
        Vector2Int intrinsicRes = _cachedIntrinsics.Resolution;
        var cameraPose = PassthroughCameraSamples.PassthroughCameraUtils.GetCameraPoseInWorld(PassthroughCameraSamples.PassthroughCameraEye.Left);

        int maskSize = 160 * 160;
        int maskOffset = targetIndex * maskSize;
        
        // --- Pass 1: Raw Data Extraction to Local Grid ---
        // 160x160 그리드를 임시 저장할 2D 배열 (Depth & Color)
        int gridSize = 160;
        float[,] tempDepth = new float[gridSize, gridSize];
        Color32[,] tempColor = new Color32[gridSize, gridSize];
        bool[,] tempValid = new bool[gridSize, gridSize];

        for (int y = 0; y < gridSize; y += _samplingStep)
        {
            for (int x = 0; x < gridSize; x += _samplingStep)
            {
                float maskVal = _latestMaskWeights[maskOffset + (y * gridSize + x)];
                if (maskVal > _confidenceThreshold)
                {
                    Vector2Int rgbPixel = MapMaskToRGBPixel(x, y, rgbBox, rgbW, rgbH);
                    
                    // Depth Sampling
                    float rgbU = (float)rgbPixel.x / rgbW;
                    float rgbV = (float)rgbPixel.y / rgbH;
                    int dx = Mathf.Clamp(Mathf.FloorToInt(rgbU * (depthW - 1)), 0, depthW - 1);
                    int dy;
                    if (_flipDepthSampling) dy = Mathf.Clamp(Mathf.FloorToInt((1.0f - rgbV) * (depthH - 1)), 0, depthH - 1);
                    else dy = Mathf.Clamp(Mathf.FloorToInt(rgbV * (depthH - 1)), 0, depthH - 1);

                    float d = Mathf.HalfToFloat(depthData[dy * depthW + dx]);

                    // Color Sampling
                    int pixelIdx = rgbPixel.y * rgbW + rgbPixel.x;
                    if (pixelIdx >= 0 && pixelIdx < _rgbPixelCache.Length)
                    {
                        tempColor[x, y] = _rgbPixelCache[pixelIdx];
                    }

                    // Apply Min/Max Depth Filter
                    if (d > _minDepth && d < _maxDepth)
                    {
                        tempDepth[x, y] = d;
                        tempValid[x, y] = true;
                    }
                }
            }
        }

        // --- Pass 2: Hole Filling (Iterative Interpolation - STRICT MASK ONLY) ---
        // 마스크 내부인데 Depth가 없는 경우만 채움 (배경으로 확산 방지)
        // 반복 횟수를 64회로 대폭 늘려, 테두리에서 중앙까지 Depth가 전파되도록 함 (Inpainting)
        for (int iter = 0; iter < 64; iter++)
        {
            bool changed = false; // 최적화: 더 이상 채워지는 게 없으면 조기 종료

            for (int y = 1; y < gridSize - 1; y += _samplingStep)
            {
                for (int x = 1; x < gridSize - 1; x += _samplingStep)
                {
                    // 이미 유효하면 패스
                    if (tempValid[x, y]) continue;

                    // 마스크 값 확인: 마스크가 물체라고 한 곳인가?
                    float maskVal = _latestMaskWeights[maskOffset + (y * gridSize + x)];
                    if (maskVal <= _confidenceThreshold) continue; // 마스크 밖이면 절대 채우지 않음

                    // 여기서부터는 "마스크는 있는데 Depth가 구멍난 곳"
                    float sum = 0; int count = 0;
                    if (tempValid[x + 1, y]) { sum += tempDepth[x + 1, y]; count++; }
                    if (tempValid[x - 1, y]) { sum += tempDepth[x - 1, y]; count++; }
                    if (tempValid[x, y + 1]) { sum += tempDepth[x, y + 1]; count++; }
                    if (tempValid[x, y - 1]) { sum += tempDepth[x, y - 1]; count++; }

                    if (count >= 1)
                    {
                        tempDepth[x, y] = sum / count;
                        tempValid[x, y] = true;
                        changed = true;
                        
                        // 색상 보간
                        int r = 0, g = 0, b = 0, colorCount = 0;
                        if (tempValid[x + 1, y]) { Color32 c = tempColor[x + 1, y]; r += c.r; g += c.g; b += c.b; colorCount++; }
                        if (tempValid[x - 1, y]) { Color32 c = tempColor[x - 1, y]; r += c.r; g += c.g; b += c.b; colorCount++; }
                        if (tempValid[x, y + 1]) { Color32 c = tempColor[x, y + 1]; r += c.r; g += c.g; b += c.b; colorCount++; }
                        if (tempValid[x, y - 1]) { Color32 c = tempColor[x, y - 1]; r += c.r; g += c.g; b += c.b; colorCount++; }

                        if (colorCount > 0)
                            tempColor[x, y] = new Color32((byte)(r / colorCount), (byte)(g / colorCount), (byte)(b / colorCount), 255);
                    }
                }
            }

            if (!changed) break; // 더 이상 채울 구멍이 없으면 종료
        }

        // --- Pass 3: Smoothing (3x3 Blur) ---
        // 표면을 부드럽게 만들기 위해 Depth Blur 적용
        float[,] smoothedDepth = new float[gridSize, gridSize];
        for (int y = 1; y < gridSize - 1; y += _samplingStep)
        {
            for (int x = 1; x < gridSize - 1; x += _samplingStep)
            {
                if (tempValid[x, y])
                {
                    float sum = tempDepth[x, y];
                    int count = 1;
                    
                    // 3x3 kernel
                    if (tempValid[x+1, y]) { sum += tempDepth[x+1, y]; count++; }
                    if (tempValid[x-1, y]) { sum += tempDepth[x-1, y]; count++; }
                    if (tempValid[x, y+1]) { sum += tempDepth[x, y+1]; count++; }
                    if (tempValid[x, y-1]) { sum += tempDepth[x, y-1]; count++; }

                    smoothedDepth[x, y] = sum / count;
                }
            }
        }

        // --- Pass 4: Generate World Mesh ---
        int gridIndex = 0;
        for (int y = 0; y < gridSize; y += _samplingStep)
        {
            for (int x = 0; x < gridSize; x += _samplingStep)
            {
                // Smoothing된 Depth 사용
                if (tempValid[x, y])
                {
                    float d = smoothedDepth[x, y];
                    if (d == 0) d = tempDepth[x, y]; // Fallback if smoothing skipped edge

                    Vector2Int rgbPixel = MapMaskToRGBPixel(x, y, rgbBox, rgbW, rgbH);
                    
                    float scaledPixelX = (float)rgbPixel.x / rgbW * intrinsicRes.x;
                    float scaledPixelY = (float)rgbPixel.y / rgbH * intrinsicRes.y;

                    float localX = (scaledPixelX - cx) / fx * d;
                    float localY = (scaledPixelY - cy) / fy * d;
                    float localZ = d;

                    Vector3 localPoint = new Vector3(localX, -localY, localZ);
                    Vector3 worldPos = cameraPose.position + cameraPose.rotation * localPoint;

                    worldGrid[gridIndex] = worldPos;
                    colorGrid[gridIndex] = tempColor[x, y];
                    validGrid[gridIndex] = true;
                    centerPosSum += worldPos;
                    validCount++;
                }
                gridIndex++;
            }
        }

        Debug.Log($"[Snapshot] Valid points: {validCount} (Smoothed & Filled)");
    }

    private int FindBestMatchIndexFromCache()
    {
        if (_latestBoxData == null || _latestBoxCount == 0) return -1;
        int bestIdx = -1; float bestIoU = 0f;
        float scaleX = _inputSize.x / 640f; float scaleY = _inputSize.y / 640f;
        float halfW = _inputSize.x / 2f; float halfH = _inputSize.y / 2f;

        for (int i = 0; i < Mathf.Min(_latestBoxCount, 200); i++)
        {
            float cx = _latestBoxData[i * 4 + 0] * scaleX - halfW;
            float cy = _latestBoxData[i * 4 + 1] * scaleY - halfH;
            float w = _latestBoxData[i * 4 + 2] * scaleX;
            float h = _latestBoxData[i * 4 + 3] * scaleY;
            BoundingBox box = new BoundingBox { CenterX = cx, CenterY = cy, Width = w, Height = h };
            float iou = TrackingUtils.CalculateIoU(_lockedTargetBox.Value, box);
            if (iou > bestIoU) { bestIoU = iou; bestIdx = i; }
        }
        return (bestIoU > 0.3f) ? bestIdx : -1;
    }

    private IEnumerator PrepareDepthDataAsync()
    {
        if (_depthManager == null || !_depthManager.IsDepthAvailable) yield break;
        var depthRT = Shader.GetGlobalTexture("_PreprocessedEnvironmentDepthTexture") as RenderTexture;
        if (depthRT == null) yield break;
        if (_cpuDepthTex == null || _cpuDepthTex.width != depthRT.width) _cpuDepthTex = new Texture2D(depthRT.width, depthRT.height, TextureFormat.RHalf, false, true);
        bool done = false;
        AsyncGPUReadback.Request(depthRT, 0, request => { if (!request.hasError) { var data = request.GetData<ushort>(); _cpuDepthTex.LoadRawTextureData(data); _cpuDepthTex.Apply(); } done = true; });
        while (!done) yield return null;
    }

    private IEnumerator PrepareRgbDataAsync()
    {
        if (_webCamManager == null || _webCamManager.WebCamTexture == null) yield break;
        var rgbTex = _webCamManager.WebCamTexture;
        if (_rgbRenderTexture == null || _rgbRenderTexture.width != rgbTex.width) { _rgbRenderTexture = new RenderTexture(rgbTex.width, rgbTex.height, 0, RenderTextureFormat.ARGB32); _rgbPixelCache = new Color32[rgbTex.width * rgbTex.height]; }
        Graphics.Blit(rgbTex, _rgbRenderTexture);
        bool done = false;
        AsyncGPUReadback.Request(_rgbRenderTexture, 0, TextureFormat.RGBA32, request => { if (!request.hasError) { var data = request.GetData<Color32>(); data.CopyTo(_rgbPixelCache); } done = true; });
        while (!done) yield return null;
    }

    private Ray GetRayFromScreenBox(BoundingBox box)
    {
        float screenX = box.CenterX + (_inputSize.x / 2f);
        float screenY = (_inputSize.y / 2f) - box.CenterY;
        var rgbW = _webCamManager != null ? _webCamManager.WebCamTexture.width : 1280;
        var rgbH = _webCamManager != null ? _webCamManager.WebCamTexture.height : 1280;
        float u = screenX / (float)_inputSize.x;
        float v = screenY / (float)_inputSize.y;
        return PassthroughCameraSamples.PassthroughCameraUtils.ScreenPointToRayInWorld(PassthroughCameraSamples.PassthroughCameraEye.Left, new Vector2Int((int)(u * rgbW), (int)(v * rgbH)));
    }

    public void SelectTargetFromScreenPos(Vector2 screenPos)
    {
        if (_currentFrameBoxes == null || _currentFrameBoxes.Count == 0) return;
        float minDistance = float.MaxValue; BoundingBox? bestBox = null;
        float halfScreenW = Screen.width / 2f; float halfScreenH = Screen.height / 2f;
        foreach (var box in _currentFrameBoxes)
        {
            float boxScreenX = box.CenterX + halfScreenW; float boxScreenY = halfScreenH - box.CenterY;
            if (screenPos.x >= boxScreenX - box.Width/2f - 30f && screenPos.x <= boxScreenX + box.Width/2f + 30f && screenPos.y >= boxScreenY - box.Height/2f - 30f && screenPos.y <= boxScreenY + box.Height/2f + 30f)
            {
                float dist = Vector2.Distance(screenPos, new Vector2(boxScreenX, boxScreenY));
                if (dist < minDistance) { minDistance = dist; bestBox = box; }
            }
        }
        if (bestBox.HasValue) { _lockedTargetBox = bestBox.Value; _isTracking = true; _consecutiveLostFrames = 0; _centerVelocity = Vector2.zero; _sizeVelocity = Vector2.zero; _currentSmoothTime = _maxSmoothTime; if (_pointCloudRenderer != null) _pointCloudRenderer.ClearMesh(); Debug.Log($"[IEExecutor] Target Selected: {bestBox.Value.ClassName}"); }
    }

    public void ResetTracking() { _isTracking = false; _lockedTargetBox = null; _consecutiveLostFrames = 0; if (_ieMasker != null) _ieMasker.ClearAllMasks(); if (_pointCloudRenderer != null) _pointCloudRenderer.ClearMesh(); if (_ieBoxer != null) _ieBoxer.ClearBoxes(0); Debug.Log("[IEExecutor] Tracking Reset"); }

    private void CleanupResources() { _output0BoxCoords?.Dispose(); _output1LabelIds?.Dispose(); _output2Masks?.Dispose(); _output3MaskWeights?.Dispose(); _readbacksInitiated = false; }


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

    private Vector2Int MapMaskToRGBPixel(int maskX, int maskY, BoundingBox box, int rgbW, int rgbH)
    {
        float normX = (float)maskX / 160f;
        float normY = (float)maskY / 160f;

        // Reverting to the original logic which handles Y-axis correctly for Unity (Bottom-Left 0)
        // box.CenterY is relative to the image center.
        // If box.CenterY is positive (Down in UI), (H/2) - CenterY moves it DOWN in texture space (towards 0).
        float texCenterX = box.CenterX + (rgbW * 0.5f);
        float texCenterY = (rgbH * 0.5f) - box.CenterY;

        float boxLeft = texCenterX - (box.Width * 0.5f);
        float boxTop = texCenterY + (box.Height * 0.5f);

        float pixelX = boxLeft + (normX * box.Width);
        // Go DOWN from Top
        float pixelY = boxTop - (normY * box.Height);

        return new Vector2Int(Mathf.Clamp(Mathf.RoundToInt(pixelX), 0, rgbW - 1), Mathf.Clamp(Mathf.RoundToInt(pixelY), 0, rgbH - 1));
    }
}
