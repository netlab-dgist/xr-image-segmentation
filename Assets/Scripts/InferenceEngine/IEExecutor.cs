using System;
using System.Collections;
using System.Collections.Generic;
using Unity.InferenceEngine;
using UnityEngine;
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
    [SerializeField] private float _confidenceThreshold = 0.4f;
    [SerializeField] private Transform _displayLocation;

    [Header("Stabilization Settings")]
    [Tooltip("트래킹 실패 시에도 이전 형상을 유지할 시간 (초)")]
    [SerializeField] private float _keepAliveDuration = 1.0f;
    private float _lastValidTime = 0f;

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

    // 이중 버퍼: 현재 프레임용 / 렌더링용(안전한 데이터)
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

    // Readback 관련
    private readonly Tensor[] _outputBuffers = new Tensor[4];
    private readonly bool[] _readbackComplete = new bool[4];
    private bool _readbacksInitiated = false;
    private Tensor<float> _output0BoxCoords;
    private Tensor<int> _output1LabelIds;
    private Tensor<float> _output2Masks;
    private Tensor<float> _output3MaskWeights;

    // Depth 관련
    private Texture2D _cpuDepthTex;
    private bool _isDepthReadingBack = false;
    private PassthroughCameraIntrinsics _cachedIntrinsics;
    private bool _intrinsicsCached = false;

    // [에러 해결 1] 외부 스크립트 호환용 프로퍼티 복구
    public bool IsModelLoaded { get; private set; } = false;
    
    // [에러 해결 2] Trigger 스크립트가 참조하는 IsTracking
    // 자동 모드이므로 항상 true로 두거나, 모델이 실행 중이면 true로 반환
    public bool IsTracking => _started; 

    // [에러 해결 3] 외부에서 접근 가능한 LockedTargetBox (자동 모드에서는 null일 수 있으나 형식 유지)
    public BoundingBox? LockedTargetBox => null; 

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
        if (_ieMasker != null) _ieMasker.Initialize(_displayLocation, _confidenceThreshold);
        LoadModel();
    }

    private void Update()
    {
        UpdateInference();
        PrepareDepthData();

        // 타임아웃 체크
        if (Time.time - _lastValidTime > _keepAliveDuration && CurrentPointCount > 0)
        {
             // 필요시 잔상 제거 로직 (선택 사항)
             // CurrentPointCount = 0; 
        }
    }

    private void OnDestroy()
    {
        if (_schedule != null) StopCoroutine(_schedule);
        _input?.Dispose();
        _inferenceEngineWorker?.Dispose();
        if (_cpuDepthTex != null) Destroy(_cpuDepthTex);
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
        
        // [에러 해결 4] 모델 로드 완료 플래그 설정
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
        
        if (boxesFound > 0)
        {
            BoundingBox bestBox = ParseSingleBox(0);
            _ieBoxer.HideAllBoxes(); 
            _ieMasker.DrawSingleMask(0, bestBox, _output3MaskWeights, _inputSize.x, _inputSize.y);
            ExtractDepthData(0, bestBox);
        }
        else
        {
            _ieMasker.KeepCurrentMask(); 
        }
    }

    private BoundingBox ParseSingleBox(int index)
    {
        var scaleX = _inputSize.x / 640f;
        var scaleY = _inputSize.y / 640f;
        var halfWidth = _inputSize.x / 2f;
        var halfHeight = _inputSize.y / 2f;

        var centerX = _output0BoxCoords[index, 0] * scaleX - halfWidth;
        var centerY = _output0BoxCoords[index, 1] * scaleY - halfHeight;
        var classname = _ieBoxer.GetClassName(_output1LabelIds[index]);

        return new BoundingBox
        {
            CenterX = centerX,
            CenterY = centerY,
            ClassName = classname,
            Width = _output0BoxCoords[index, 2] * scaleX,
            Height = _output0BoxCoords[index, 3] * scaleY,
            Label = classname,
        };
    }

    private void ExtractDepthData(int targetIndex, BoundingBox box)
    {
        if (_cpuDepthTex == null) return;

        if (!_intrinsicsCached)
        {
            _cachedIntrinsics = PassthroughCameraUtils.GetCameraIntrinsics(PassthroughCameraEye.Left);
            _intrinsicsCached = true;
        }

        Pose cameraPose = PassthroughCameraUtils.GetCameraPoseInWorld(PassthroughCameraEye.Left);
        Vector3 cameraPos = cameraPose.position;
        Quaternion cameraRot = cameraPose.rotation;

        float fx = _cachedIntrinsics.FocalLength.x;
        float fy = _cachedIntrinsics.FocalLength.y;
        float cx = _cachedIntrinsics.PrincipalPoint.x;
        float cy = _cachedIntrinsics.PrincipalPoint.y;

        var depthData = _cpuDepthTex.GetRawTextureData<ushort>();
        int depthW = _cpuDepthTex.width;
        int depthH = _cpuDepthTex.height;

        int newPointCount = 0;

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
                    
                    float imgPixelX = (box.CenterX - box.Width * 0.5f + normX * box.Width) + 320f;
                    float imgPixelY = 320f - (box.CenterY - box.Height * 0.5f + normY * box.Height); 

                    float u = Mathf.Clamp01(imgPixelX / 640f);
                    float v = Mathf.Clamp01(imgPixelY / 640f);

                    int dx = Mathf.FloorToInt(u * (depthW - 1));
                    int dy = Mathf.FloorToInt(v * (depthH - 1));

                    int depthIndex = dy * depthW + dx;
                    if (depthIndex < 0 || depthIndex >= depthData.Length) continue;

                    float depthMeters = Mathf.HalfToFloat(depthData[depthIndex]);

                    if (depthMeters > 0.1f && depthMeters < 3.0f)
                    {
                        Vector3 dirInCamera = new Vector3((dx - cx) / fx, (dy - cy) / fy, 1f);
                        Vector3 worldPos = cameraPos + (cameraRot * dirInCamera.normalized) * depthMeters;

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
            _lastValidTime = Time.time;
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
        _readbacksInitiated = false;
    }

    // [에러 해결 5] 외부 호출용 함수 복구 (ResetTracking)
    // 자동 모드이므로 화면 초기화(버퍼 클리어) 용도로 사용
    public void ResetTracking()
    {
        CurrentPointCount = 0;
        _backupCount = 0;
        Debug.Log("[IEExecutor] Buffer Cleared by Trigger");
    }

    // [에러 해결 6] 외부 호출용 함수 복구 (SelectTargetFromScreenPos)
    // 자동 모드(Auto Visualization)이므로 별도의 선택 로직 없이 로그만 출력
    public void SelectTargetFromScreenPos(Vector2 screenPos)
    {
        Debug.Log("[IEExecutor] Auto Mode Active: Selection Ignored (Showing best result automatically)");
    }
}