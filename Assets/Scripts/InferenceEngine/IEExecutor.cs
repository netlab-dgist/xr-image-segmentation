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
    [SerializeField] private int _maxPoints = 8000;
    
    // [신규] Point Cloud Renderer 참조
    [SerializeField] private PointCloudRenderer _pointCloudRenderer;

    [Header("Optimization Settings")]
    [Range(2, 8)]
    [SerializeField] private int _samplingStep = 4;

    public bool IsModelLoaded { get; private set; } = false;

    // 외부 공개용 프로퍼티
    public IEMasker Masker => _ieMasker;
    public Vector2Int InputSize => _inputSize;
    public bool IsTracking => _isTracking;
    public BoundingBox? LockedTargetBox => _lockedTargetBox;

    [SerializeField] private IEBoxer _ieBoxer;

    private IEMasker _ieMasker;
    private Worker _inferenceEngineWorker;
    private IEnumerator _schedule;
    private InferenceDownloadState _downloadState = InferenceDownloadState.Running;

    private Tensor<float> _input;

    // 병렬 Readback 버퍼
    private readonly Tensor[] _outputBuffers = new Tensor[4];
    private readonly bool[] _readbackComplete = new bool[4];
    private Tensor<float> _output0BoxCoords;
    private Tensor<int> _output1LabelIds;
    private Tensor<float> _output2Masks;
    private Tensor<float> _output3MaskWeights;

    // Depth/RGB 처리를 위한 임시 버퍼 (Snapshot 시에만 사용)
    private Texture2D _cpuDepthTex;
    private Color32[] _rgbPixelCache;
    private RenderTexture _rgbRenderTexture;
    
    // 카메라 Intrinsics
    private PassthroughCameraSamples.PassthroughCameraIntrinsics _cachedIntrinsics;
    private bool _intrinsicsCached = false;

    // Tracking State
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
        
        // PointCloudRenderer가 없다면 자동으로 추가
        if (_pointCloudRenderer == null)
        {
            _pointCloudRenderer = gameObject.GetComponent<PointCloudRenderer>();
            if (_pointCloudRenderer == null)
            {
                _pointCloudRenderer = gameObject.AddComponent<PointCloudRenderer>();
            }
        }
    }

    private void Update()
    {
        UpdateInference();
        
        // [핵심 변경] 매 프레임 Depth/RGB 추출하지 않음.
        // 대신 캡처된 Point Cloud가 있다면, 현재 YOLO Box 위치에 맞춰 이동시킴
        if (_isTracking && _pointCloudRenderer.IsMeshGenerated && _lockedTargetBox.HasValue)
        {
            // 현재 박스 중심에서 쏘는 Ray 생성 (World Space)
            Ray ray = GetRayFromScreenBox(_lockedTargetBox.Value);
            _pointCloudRenderer.UpdateTransform(_lockedTargetBox.Value, ray);
        }
    }

    private void OnDestroy()
    {
        if (_schedule != null) StopCoroutine(_schedule);
        _input?.Dispose();
        _inferenceEngineWorker?.Dispose();
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

    private void UpdateParallelReadbacks()
    {
        if (!_readbacksInitiated)
        {
            for (int i = 0; i < 4; i++)
            {
                _readbackComplete[i] = false;
                _outputBuffers[i] = _inferenceEngineWorker.PeekOutput(i);

                if (_outputBuffers[i].dataOnBackend != null)
                    _outputBuffers[i].ReadbackRequest();
                else
                {
                    _downloadState = InferenceDownloadState.Error;
                    return;
                }
            }
            _readbacksInitiated = true;
            return;
        }

        bool allComplete = true;
        for (int i = 0; i < 4; i++)
        {
            if (!_readbackComplete[i])
            {
                if (_outputBuffers[i].IsReadbackRequestDone())
                    _readbackComplete[i] = true;
                else
                    allComplete = false;
            }
        }

        if (allComplete)
        {
            _output0BoxCoords = _outputBuffers[0].ReadbackAndClone() as Tensor<float>;
            _output1LabelIds = _outputBuffers[1].ReadbackAndClone() as Tensor<int>;
            _output2Masks = _outputBuffers[2].ReadbackAndClone() as Tensor<float>;
            _output3MaskWeights = _outputBuffers[3].ReadbackAndClone() as Tensor<float>;

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
                
                // [변경] 매 프레임 마스크만 업데이트 (Point Cloud 재생성 X)
                // 필요하다면 시각적 피드백을 위해 마스크를 그릴 수 있지만, 
                // PointCloud가 생성된 상태라면 마스크를 꺼도 됩니다. 여기서는 유지.
                _ieMasker.DrawSingleMask(bestIndex, bestBox, _output3MaskWeights, _inputSize.x, _inputSize.y);
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
        }
    }

    // ------------------------------------------------------------------------
    // [Snapshot Logic] Trigger 누를 때 1회 실행
    // ------------------------------------------------------------------------
    public void CaptureSnapshot()
    {
        if (!_isTracking || !_lockedTargetBox.HasValue) return;
        
        // 현재 추적 중인 대상의 인덱스를 찾아서 Point Cloud 생성 요청
        StartCoroutine(CaptureRoutine());
    }

    private IEnumerator CaptureRoutine()
    {
        Debug.Log("[Snapshot] Starting Capture Routine...");
        
        // 1. 필요한 리소스(Depth, RGB) 준비 요청 (Async)
        yield return PrepareDepthDataAsync();
        yield return PrepareRgbDataAsync();

        if (_cpuDepthTex == null || _rgbPixelCache == null)
        {
            Debug.LogError("[Snapshot] Failed to get Depth or RGB data.");
            yield break;
        }

        // 2. 현재 추론 결과(Mask)가 유효할 때까지 대기 (타이밍 맞추기)
        // _output3MaskWeights는 가장 최근 프레임의 결과임.
        if (_output3MaskWeights == null) 
        {
             Debug.LogWarning("[Snapshot] No mask weights available yet.");
             yield break;
        }

        // 3. Point Cloud 데이터 추출
        // 현재 추적 중인 박스 정보를 기준으로 마스크 찾기 (ProcessSuccessState에서 이미 bestIndex를 찾았어야 함... 
        // 구조상 bestIndex를 저장해두는 게 좋지만, 여기서는 가장 score 높은 놈을 다시 찾거나
        // 현재 _lockedTargetBox와 가장 가까운 놈을 찾아서 처리)
        
        // 간단히: 현재 화면 중앙(또는 박스 중심)에 있는 마스크를 추출
        // 하지만 _output3MaskWeights는 여러 객체를 포함함.
        // 현재 Tracking 중인 객체의 Index를 정확히 알기 위해, 가장 최근 추론 결과를 다시 순회
        
        int targetIndex = FindBestMatchIndex();
        if (targetIndex == -1) 
        {
            Debug.LogWarning("[Snapshot] Could not find matching object index in current frame.");
            yield break;
        }

        Debug.Log($"[Snapshot] Converting to Mesh (Index: {targetIndex})...");
        
        // 4. 데이터 추출 및 Mesh 생성
        var points = new List<Vector3>();
        var colors = new List<Color32>();
        float avgDepth = ExtractPoints(targetIndex, _lockedTargetBox.Value, points, colors);

        if (points.Count > 0)
        {
            _pointCloudRenderer.GenerateMesh(points, colors, _lockedTargetBox.Value, avgDepth);
            // 캡처 성공 후 마스크 끄기? (선택사항, 일단 유지)
        }
        else
        {
            Debug.LogWarning("[Snapshot] No valid points found (too far or invalid depth).");
        }
    }

    private int FindBestMatchIndex()
    {
        if (_output0BoxCoords == null || _output0BoxCoords.shape[0] == 0) return -1;
        
        // 현재 lockedBox와 가장 IoU가 높은 인덱스 찾기
        int bestIdx = -1;
        float bestIoU = 0f;
        int count = Mathf.Min(_output0BoxCoords.shape[0], 200);
        var scaleX = _inputSize.x / 640f; 
        var scaleY = _inputSize.y / 640f;
        var halfW = _inputSize.x / 2f;
        var halfH = _inputSize.y / 2f;

        for (int i = 0; i < count; i++)
        {
             // Tensor에서 박스 복원
             float cx = _output0BoxCoords[i, 0] * scaleX - halfW;
             float cy = _output0BoxCoords[i, 1] * scaleY - halfH;
             float w = _output0BoxCoords[i, 2] * scaleX;
             float h = _output0BoxCoords[i, 3] * scaleY;
             
             BoundingBox box = new BoundingBox { CenterX = cx, CenterY = cy, Width = w, Height = h };
             float iou = TrackingUtils.CalculateIoU(_lockedTargetBox.Value, box);
             if (iou > bestIoU)
             {
                 bestIoU = iou;
                 bestIdx = i;
             }
        }
        return (bestIoU > 0.3f) ? bestIdx : -1;
    }

    // ------------------------------------------------------------------------
    // [Helpers] Async Data Fetching
    // ------------------------------------------------------------------------
    private IEnumerator PrepareDepthDataAsync()
    {
        if (_depthManager == null || !_depthManager.IsDepthAvailable) yield break;
        var depthRT = Shader.GetGlobalTexture("_PreprocessedEnvironmentDepthTexture") as RenderTexture;
        if (depthRT == null) yield break;

        if (_cpuDepthTex == null || _cpuDepthTex.width != depthRT.width)
            _cpuDepthTex = new Texture2D(depthRT.width, depthRT.height, TextureFormat.RHalf, false, true);

        bool done = false;
        AsyncGPUReadback.Request(depthRT, 0, request => {
            if (!request.hasError)
            {
                var data = request.GetData<ushort>();
                _cpuDepthTex.LoadRawTextureData(data);
                _cpuDepthTex.Apply();
            }
            done = true;
        });

        while (!done) yield return null;
    }

    private IEnumerator PrepareRgbDataAsync()
    {
        if (_webCamManager == null) yield break;
        var rgbTex = _webCamManager.WebCamTexture;
        if (rgbTex == null) yield break;

        if (_rgbRenderTexture == null || _rgbRenderTexture.width != rgbTex.width)
        {
            if (_rgbRenderTexture != null) _rgbRenderTexture.Release();
            _rgbRenderTexture = new RenderTexture(rgbTex.width, rgbTex.height, 0, RenderTextureFormat.ARGB32);
            _rgbPixelCache = new Color32[rgbTex.width * rgbTex.height];
        }

        Graphics.Blit(rgbTex, _rgbRenderTexture);
        
        bool done = false;
        AsyncGPUReadback.Request(_rgbRenderTexture, 0, TextureFormat.RGBA32, request => {
            if (!request.hasError)
            {
                var data = request.GetData<Color32>();
                data.CopyTo(_rgbPixelCache);
            }
            done = true;
        });
        
        while (!done) yield return null;
    }

    // ------------------------------------------------------------------------
    // [Extraction] Points Extraction Logic
    // ------------------------------------------------------------------------
    private float ExtractPoints(int targetIndex, BoundingBox box, List<Vector3> outPoints, List<Color32> outColors)
    {
        if (!_intrinsicsCached)
        {
            _cachedIntrinsics = PassthroughCameraSamples.PassthroughCameraUtils.GetCameraIntrinsics(
                PassthroughCameraSamples.PassthroughCameraEye.Left);
            _intrinsicsCached = true;
        }

        Pose cameraPose = PassthroughCameraSamples.PassthroughCameraUtils.GetCameraPoseInWorld(
            PassthroughCameraSamples.PassthroughCameraEye.Left);
        
        float fx = _cachedIntrinsics.FocalLength.x;
        float fy = _cachedIntrinsics.FocalLength.y;
        float cx = _cachedIntrinsics.PrincipalPoint.x;
        float cy = _cachedIntrinsics.PrincipalPoint.y;

        var depthData = _cpuDepthTex.GetRawTextureData<ushort>();
        int rgbW = _rgbRenderTexture.width;
        int rgbH = _rgbRenderTexture.height;
        int depthW = _cpuDepthTex.width;
        int depthH = _cpuDepthTex.height;

        float totalDepth = 0f;
        int depthCount = 0;

        for (int y = 0; y < 160; y += _samplingStep)
        {
            for (int x = 0; x < 160; x += _samplingStep)
            {
                if (outPoints.Count >= _maxPoints) break;
                
                float maskVal = _output3MaskWeights[targetIndex, y, x];
                if (maskVal > _confidenceThreshold)
                {
                    Vector2Int rgbPixel = MapMaskToRGBPixel(x, y, box, rgbW, rgbH);
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
                        Vector3 dirInWorld = cameraPose.rotation * dirInCamera;
                        Vector3 worldPos = cameraPose.position + dirInWorld.normalized * depthMeters;

                        outPoints.Add(worldPos);
                        outColors.Add(_rgbPixelCache[pixelIdx]);
                        
                        totalDepth += depthMeters;
                        depthCount++;
                    }
                }
            }
        }
        
        return depthCount > 0 ? (totalDepth / depthCount) : 1.0f;
    }

    private Ray GetRayFromScreenBox(BoundingBox box)
    {
        // Screen Point to Ray
        // PassthroughCameraUtils 사용
        float screenX = box.CenterX + (_inputSize.x / 2f);
        float screenY = (_inputSize.y / 2f) - box.CenterY;
        
        // 640x640 좌표 -> 실제 RGB 해상도로 변환 필요할 수 있음 (Utils가 내부적으로 처리하는지 확인 필요)
        // Utils.ScreenPointToRayInWorld는 RGB 텍스처 기준 좌표를 원함.
        // 하지만 여기서는 간단히 Unity Camera.main을 쓸 수도 있고, Utils를 쓸 수도 있음.
        // 정확도를 위해 PassthroughCameraUtils 사용 권장하지만, 입력 좌표계 변환이 필요함.
        
        // IEExecutor가 사용하는 640x640 좌표를 RGB 텍스처 좌표로 매핑
        var rgbW = _webCamManager != null ? _webCamManager.WebCamTexture.width : 1280;
        var rgbH = _webCamManager != null ? _webCamManager.WebCamTexture.height : 720;
        
        float u = screenX / 640f;
        float v = screenY / 640f;
        
        return PassthroughCameraSamples.PassthroughCameraUtils.ScreenPointToRayInWorld(
            PassthroughCameraSamples.PassthroughCameraEye.Left, 
            new Vector2Int((int)(u * rgbW), (int)(v * rgbH)));
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
            
            // PointCloudRenderer 리셋
            if (_pointCloudRenderer != null) _pointCloudRenderer.ClearMesh();
            
            Debug.Log($"[IEExecutor] Target Selected: {bestBox.Value.ClassName}");
        }
    }

    public void ResetTracking()
    {
        _isTracking = false;
        _lockedTargetBox = null;
        _consecutiveLostFrames = 0;
        if (_ieMasker != null) _ieMasker.ClearAllMasks();
        if (_pointCloudRenderer != null) _pointCloudRenderer.ClearMesh();
        
        Debug.Log("[IEExecutor] Tracking Reset");
    }

    private Vector2Int MapMaskToRGBPixel(int maskX, int maskY, BoundingBox box, int rgbW, int rgbH)
    {
        float normX = (float)maskX / 160f;
        float normY = (float)maskY / 160f;
        int pixelX = Mathf.RoundToInt((box.CenterX - box.Width/2f + normX * box.Width) + rgbW/2f);
        int pixelY = Mathf.RoundToInt(rgbH/2f - (box.CenterY - box.Height/2f + normY * box.Height));
        return new Vector2Int(Mathf.Clamp(pixelX, 0, rgbW - 1), Mathf.Clamp(pixelY, 0, rgbH - 1));
    }

}