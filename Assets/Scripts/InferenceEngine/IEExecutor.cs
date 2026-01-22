using System;
using System.Collections;
using System.Collections.Generic;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Rendering;
using Meta.XR.EnvironmentDepth;
using PassthroughCameraSamples;

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

    public struct RGBDPoint {
        public Vector3 worldPos;
        public Color32 color;
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

    private bool _isTracking = false;
    private BoundingBox? _lockedTargetBox = null;
    private List<BoundingBox> _currentFrameBoxes = new List<BoundingBox>();

    // UI 복구용 텍스처
    private Texture2D _transparentBackground;

    public bool IsModelLoaded { get; private set; } = false;
    public bool IsTracking => _isTracking; 
    public BoundingBox? LockedTargetBox => _lockedTargetBox; 

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
        CleanupResources();
    }

    private void PrepareDepthData()
    {
        if (_depthManager == null || !_depthManager.IsDepthAvailable || _isDepthReadingBack) return;
        var depthRT = Shader.GetGlobalTexture("_PreprocessedEnvironmentDepthTexture") as RenderTexture;
        if (depthRT == null) return;

        if (_cpuDepthTex == null || _cpuDepthTex.width != depthRT.width || _cpuDepthTex.height != depthRT.height)
        {
            if(_cpuDepthTex != null) Destroy(_cpuDepthTex);
            _cpuDepthTex = new Texture2D(depthRT.width, depthRT.height, TextureFormat.RHalf, false, true);
        }

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

        // [Case 1] 트래킹 모드가 아닐 때 (자동 표시 모드)
        if (!_isTracking)
        {
            // UI 렌더링이 켜져 있을 때만 그리기 (깜빡임 원인 차단)
            if (EnableUIRendering)
            {
                _ieBoxer.DrawBoxes(_output0BoxCoords, _output1LabelIds, screenW, screenH);
            }
            else
            {
                _ieBoxer.HideAllBoxes(); // UI 끄기
                _ieMasker.ClearAllMasks();
            }

            // * 중요: UI를 끄더라도 가장 자신감 있는 물체 하나는 포인트 클라우드로 보여줌
            if (boxesFound > 0)
            {
                // 자동으로 첫 번째 물체를 타겟으로 Point Cloud 생성
                BoundingBox bestBox = _currentFrameBoxes[0];
                ExtractDepthData(0, bestBox);
            }
            else
            {
                CurrentPointCount = 0;
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

                // UI 렌더링이 켜진 경우에만 마스크 그리기
                if (EnableUIRendering)
                {
                    _ieMasker.DrawSingleMask(bestIndex, bestBox, _output3MaskWeights, _inputSize.x, _inputSize.y);
                }
                else
                {
                    _ieMasker.ClearAllMasks();
                }
                
                // * Point Cloud는 무조건 추출 (UI 설정과 무관)
                ExtractDepthData(bestIndex, bestBox);
            }
            else
            {
                // 놓쳤을 때: Point Cloud 유지 (Masker 호출 안 함)
                // _ieMasker.KeepCurrentMask() 호출도 생략하여 부하 줄임
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
        if (_cpuDepthTex == null) return;

        if (!_intrinsicsCached)
        {
            _cachedIntrinsics = PassthroughCameraUtils.GetCameraIntrinsics(PassthroughCameraEye.Left);
            _intrinsicsCached = true;
        }

        var depthData = _cpuDepthTex.GetRawTextureData<ushort>();
        int depthW = _cpuDepthTex.width;
        int depthH = _cpuDepthTex.height;
        Vector2 sensorRes = _cachedIntrinsics.Resolution;

        int newPointCount = 0;

        float screenW = Screen.width;
        float screenH = Screen.height;
        
        float rawBoxCenterX = (box.CenterX / (screenW / 640f)) + 320f;
        float rawBoxCenterY = 320f - (box.CenterY / (screenH / 640f));
        float rawBoxWidth = box.Width / (screenW / 640f);
        float rawBoxHeight = box.Height / (screenH / 640f);

        for (int y = 0; y < 160; y += _samplingStep)
        {
            for (int x = 0; x < 160; x += _samplingStep)
            {
                if (newPointCount >= _maxPoints) break;

                float maskVal = _output3MaskWeights[targetIndex, y, x];
                if (maskVal > _confidenceThreshold)
                {
                    float normX = (float)x / 160f;
                    float normY = (float)y / 160f;
                    
                    float imgPixelX = rawBoxCenterX - (rawBoxWidth * 0.5f) + (normX * rawBoxWidth);
                    float imgPixelY = rawBoxCenterY - (rawBoxHeight * 0.5f) + (normY * rawBoxHeight);

                    float u = Mathf.Clamp01(imgPixelX / 640f);
                    float v = Mathf.Clamp01(imgPixelY / 640f); 

                    int dx = Mathf.FloorToInt(u * (depthW - 1));
                    int dy = Mathf.FloorToInt((1.0f - v) * (depthH - 1));

                    int depthIndex = dy * depthW + dx;
                    if (depthIndex < 0 || depthIndex >= depthData.Length) continue;
                    float depthMeters = Mathf.HalfToFloat(depthData[depthIndex]);

                    if (depthMeters > 0.1f && depthMeters < 3.0f)
                    {
                        Vector2Int cameraPixel = new Vector2Int(
                            Mathf.FloorToInt(u * sensorRes.x), 
                            Mathf.FloorToInt((1.0f - v) * sensorRes.y)
                        );

                        Ray worldRay = PassthroughCameraUtils.ScreenPointToRayInWorld(PassthroughCameraEye.Left, cameraPixel);
                        Vector3 worldPos = worldRay.GetPoint(depthMeters);

                        PointBuffer[newPointCount].worldPos = worldPos;
                        
                        float normalizedDepth = Mathf.Clamp01((depthMeters - 0.2f) / 2.0f);
                        PointBuffer[newPointCount].color = _depthGradient.Evaluate(normalizedDepth);

                        newPointCount++;
                    }
                }
            }
        }

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

    public void SelectTargetFromScreenPos(Vector2 screenPos)
    {
        if (_currentFrameBoxes == null || _currentFrameBoxes.Count == 0) return;

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