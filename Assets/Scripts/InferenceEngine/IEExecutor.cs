using System;
using System.Collections;
using System.Collections.Generic;
using System.Reflection;
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

        // EnvironmentDepthManager에서 frameDescriptors 가져오기 (internal이라 Reflection 사용)
        float tanLeft, tanRight, tanTop, tanBottom;
        Vector3 depthCamPos;
        Quaternion depthCamRot;
        Matrix4x4 trackingToWorld = Matrix4x4.identity;

        try
        {
            var frameDescField = typeof(EnvironmentDepthManager).GetField("frameDescriptors", BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Public);
            if (frameDescField == null)
            {
                Debug.LogError("[Snapshot] Cannot find frameDescriptors field! Using fallback.");
                UseFallbackProjection(out tanLeft, out tanRight, out tanTop, out tanBottom, out depthCamPos, out depthCamRot);
            }
            else
            {
                var frameDescriptors = frameDescField.GetValue(_depthManager) as Array;
                if (frameDescriptors == null || frameDescriptors.Length == 0)
                {
                    Debug.LogError("[Snapshot] frameDescriptors is null or empty! Using fallback.");
                    UseFallbackProjection(out tanLeft, out tanRight, out tanTop, out tanBottom, out depthCamPos, out depthCamRot);
                }
                else
                {
                    object depthFrameDesc = frameDescriptors.GetValue(0);
                    Type descType = depthFrameDesc.GetType();

                    Debug.Log($"[Snapshot] DepthFrameDesc type: {descType.FullName}");

                    // internal 필드 접근 - NonPublic 플래그 사용
                    var leftField = descType.GetField("fovLeftAngleTangent", BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Public);
                    var rightField = descType.GetField("fovRightAngleTangent", BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Public);
                    var topField = descType.GetField("fovTopAngleTangent", BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Public);
                    var bottomField = descType.GetField("fovDownAngleTangent", BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Public);
                    var posField = descType.GetField("createPoseLocation", BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Public);
                    var rotField = descType.GetField("createPoseRotation", BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Public);

                    if (leftField == null || rightField == null || topField == null || bottomField == null)
                    {
                        Debug.LogError($"[Snapshot] Cannot find FOV fields! Left:{leftField != null}, Right:{rightField != null}, Top:{topField != null}, Bottom:{bottomField != null}");
                        // 모든 필드 이름 출력
                        Debug.Log($"[Snapshot] Available fields in {descType.Name}:");
                        foreach (var f in descType.GetFields(BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Public))
                        {
                            Debug.Log($"  - {f.Name} ({f.FieldType.Name})");
                        }
                        UseFallbackProjection(out tanLeft, out tanRight, out tanTop, out tanBottom, out depthCamPos, out depthCamRot);
                    }
                    else
                    {
                        tanLeft = (float)leftField.GetValue(depthFrameDesc);
                        tanRight = (float)rightField.GetValue(depthFrameDesc);
                        tanTop = (float)topField.GetValue(depthFrameDesc);
                        tanBottom = (float)bottomField.GetValue(depthFrameDesc);
                        depthCamPos = posField != null ? (Vector3)posField.GetValue(depthFrameDesc) : Vector3.zero;
                        depthCamRot = rotField != null ? (Quaternion)rotField.GetValue(depthFrameDesc) : Quaternion.identity;
                    }
                }
            }

            // Tracking space -> World 변환
            var trackingSpaceMethod = typeof(EnvironmentDepthManager).GetMethod("GetTrackingSpaceWorldToLocalMatrix", BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Public);
            if (trackingSpaceMethod != null)
            {
                Matrix4x4 worldToTracking = (Matrix4x4)trackingSpaceMethod.Invoke(_depthManager, null);
                trackingToWorld = worldToTracking.inverse;
            }
        }
        catch (Exception ex)
        {
            Debug.LogError($"[Snapshot] Reflection failed: {ex.Message}\n{ex.StackTrace}");
            UseFallbackProjection(out tanLeft, out tanRight, out tanTop, out tanBottom, out depthCamPos, out depthCamRot);
        }

        Debug.Log($"[Snapshot] Depth FOV tangents: L={tanLeft:F3}, R={tanRight:F3}, T={tanTop:F3}, B={tanBottom:F3}");
        Debug.Log($"[Snapshot] Depth Cam Pos: {depthCamPos}, Rot: {depthCamRot.eulerAngles}");
        Debug.Log($"[Snapshot] RGB: {rgbW}x{rgbH}, Depth: {depthW}x{depthH}");

        int maskSize = 160 * 160;
        int maskOffset = targetIndex * maskSize;
        int gridIndex = 0;

        for (int y = 0; y < 160; y += _samplingStep)
        {
            for (int x = 0; x < 160; x += _samplingStep)
            {
                float maskVal = _latestMaskWeights[maskOffset + (y * 160 + x)];

                if (maskVal > _confidenceThreshold)
                {
                    // Mask -> RGB 픽셀 좌표
                    Vector2Int rgbPixel = MapMaskToRGBPixel(x, y, box, rgbW, rgbH);
                    int pixelIdx = rgbPixel.y * rgbW + rgbPixel.x;

                    if (pixelIdx < 0 || pixelIdx >= _rgbPixelCache.Length) { gridIndex++; continue; }

                    // RGB -> Depth UV (0~1)
                    float u = (float)rgbPixel.x / rgbW;
                    float v = (float)rgbPixel.y / rgbH;

                    int dx = Mathf.FloorToInt(u * (depthW - 1));
                    int dy = Mathf.FloorToInt(v * (depthH - 1));
                    float depthMeters = Mathf.HalfToFloat(depthData[dy * depthW + dx]);

                    if (depthMeters > 0.1f && depthMeters < 5.0f)
                    {
                        // Depth UV를 사용해 카메라 로컬 방향 계산 (FOV tangent 기반)
                        // UV (0,0) = 좌상단, (1,1) = 우하단
                        // X: u=0 -> -tanLeft, u=1 -> tanRight
                        // Y: v=0 -> tanTop, v=1 -> -tanBottom
                        float dirX = Mathf.Lerp(-tanLeft, tanRight, u);
                        float dirY = Mathf.Lerp(tanTop, -tanBottom, v);
                        float dirZ = 1.0f;

                        // 로컬 3D 좌표 (depth 카메라 공간)
                        Vector3 localPoint = new Vector3(dirX, dirY, dirZ) * depthMeters;

                        // Meta SDK 방식: scale(1,1,-1) 적용 후 rotation, translation
                        Vector3 scaledLocal = new Vector3(localPoint.x, localPoint.y, -localPoint.z);
                        Vector3 trackingPos = depthCamPos + depthCamRot * scaledLocal;

                        // Tracking Space -> World Space
                        Vector3 worldPos = trackingToWorld.MultiplyPoint3x4(trackingPos);

                        worldGrid[gridIndex] = worldPos;
                        colorGrid[gridIndex] = _rgbPixelCache[pixelIdx];
                        validGrid[gridIndex] = true;
                        centerPosSum += worldPos;
                        validCount++;
                    }
                }
                gridIndex++;
            }
        }

        Debug.Log($"[Snapshot] Valid points: {validCount} / {gridIndex}");
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

    private void UseFallbackProjection(out float tanLeft, out float tanRight, out float tanTop, out float tanBottom, out Vector3 depthCamPos, out Quaternion depthCamRot)
    {
        // Quest 3 Depth 센서 대략적인 FOV (약 90도 수평/수직)
        // tan(45도) = 1.0
        tanLeft = 1.0f;
        tanRight = 1.0f;
        tanTop = 1.0f;
        tanBottom = 1.0f;

        // Passthrough 카메라 Pose를 대신 사용
        Pose camPose = PassthroughCameraSamples.PassthroughCameraUtils.GetCameraPoseInWorld(PassthroughCameraSamples.PassthroughCameraEye.Left);
        depthCamPos = camPose.position;
        depthCamRot = camPose.rotation;

        Debug.LogWarning("[Snapshot] Using FALLBACK depth projection - results may be inaccurate!");
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

    private Vector2Int MapMaskToRGBPixel(int maskX, int maskY, BoundingBox box, int rgbW, int rgbH)
    {
        float normX = (float)maskX / 160f;
        float normY = (float)maskY / 160f;

        float texCenterX = box.CenterX + (rgbW * 0.5f);
        float texCenterY = (rgbH * 0.5f) - box.CenterY;

        float boxLeft = texCenterX - (box.Width * 0.5f);
        float boxTop = texCenterY + (box.Height * 0.5f);

        float pixelX = boxLeft + (normX * box.Width);
        float pixelY = boxTop - (normY * box.Height);

        return new Vector2Int(Mathf.Clamp(Mathf.RoundToInt(pixelX), 0, rgbW - 1), Mathf.Clamp(Mathf.RoundToInt(pixelY), 0, rgbH - 1));
    }
}
