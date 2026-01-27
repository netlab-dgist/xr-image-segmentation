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
    [SerializeField] private int _maxPoints = 102400;

    [Header("Optimization Settings")]
    [Range(1, 8)]
    [SerializeField] private int _samplingStep = 1;

    [Tooltip("각 마스크 픽셀 내 서브샘플링 (2 = 2x2 = 4배 밀도)")]
    [Range(1, 4)]
    [SerializeField] private int _subSamplingFactor = 2;

    [Tooltip("포인트클라우드용 confidence (낮을수록 더 많은 포인트)")]
    [Range(0.01f, 0.9f)]
    [SerializeField] private float _pointCloudConfidence = 0.1f;

    [Tooltip("depth가 없는 픽셀도 평균 depth로 포함")]
    [SerializeField] private bool _includeInvalidDepth = true;

    [Tooltip("유효하지 않은 depth 대신 사용할 기본값 (0이면 평균 사용)")]
    [SerializeField] private float _fallbackDepth = 0f;

    [Header("Compute Shader Settings")]
    [SerializeField] private ComputeShader _pointCloudShader;
    [SerializeField] private float _minDepthRange = 0.1f;
    [SerializeField] private float _maxDepthRange = 3.0f;

    // Compute Shader 버퍼
    private ComputeBuffer _maskBuffer;
    private ComputeBuffer _rgbBuffer;
    private ComputeBuffer _positionBuffer;
    private ComputeBuffer _colorBuffer;
    private ComputeBuffer _counterBuffer;
    private int _kernelId = -1;
    private int _debugKernelId = -1;
    // [0]=pointCount, [1]=maskPassCount, [2]=uvFailCount, [3]=depthFailCount
    private readonly int[] _counterData = new int[4];
    private bool _computeBuffersInitialized = false;

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
    private float[] _cpuDepthData;  // Linear depth values
    private bool _isDepthReadingBack = false;
    private bool _depthDataReady = false;

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

        // Compute Shader 버퍼 해제
        _maskBuffer?.Release();
        _rgbBuffer?.Release();
        _positionBuffer?.Release();
        _colorBuffer?.Release();
        _counterBuffer?.Release();
    }

    private void PrepareDepthData()
    {
        if (_depthManager == null || !_depthManager.IsDepthAvailable || _isDepthReadingBack) return;

        // _PreprocessedEnvironmentDepthTexture는 R16G16B16A16_SFloat format의 Texture2DArray
        // SoftOcclusion 모드에서만 생성됨. 없으면 raw depth 텍스처 사용
        var depthRT = Shader.GetGlobalTexture("_PreprocessedEnvironmentDepthTexture") as RenderTexture;
        bool usePreprocessed = depthRT != null;

        if (!usePreprocessed)
        {
            depthRT = Shader.GetGlobalTexture("_EnvironmentDepthTexture") as RenderTexture;
            if (depthRT == null) return;
        }

        int w = depthRT.width;
        int h = depthRT.height;

        if (_cpuDepthData == null || _cpuDepthData.Length != w * h)
        {
            _cpuDepthData = new float[w * h];
            _cpuDepthTex = new Texture2D(w, h, TextureFormat.RFloat, false, true);
        }

        _isDepthReadingBack = true;

        if (usePreprocessed)
        {
            // _PreprocessedEnvironmentDepthTexture: R16G16B16A16_SFloat (4 channels)
            // 첫 번째 채널(R)이 depth 정보. Texture2DArray의 첫 번째 슬라이스(z=0) 요청
            AsyncGPUReadback.Request(depthRT, 0, 0, w, 0, h, 0, 1, TextureFormat.RGBAHalf, request => {
                if (request.hasError) { _isDepthReadingBack = false; return; }

                var data = request.GetData<ushort>();
                
                // RGBAHalf = 4 ushort per pixel, R채널만 추출
                for (int i = 0; i < w * h; i++)
                {
                    float rawDepth = Mathf.HalfToFloat(data[i * 4]);  // R channel only
                    _cpuDepthData[i] = rawDepth;
                }

                // [수정] _cpuDepthTex에 데이터 업데이트하여 ExtractRGBDData에서 사용 가능하게 함
                _cpuDepthTex.SetPixelData(_cpuDepthData, 0);
                _cpuDepthTex.Apply();

                _depthDataReady = true;
                _isDepthReadingBack = false;
            });
        }
        else
        {
            // _EnvironmentDepthTexture: R16_UNorm
            // Texture2DArray의 첫 번째 슬라이스(z=0) 요청
            AsyncGPUReadback.Request(depthRT, 0, 0, w, 0, h, 0, 1, TextureFormat.R16, request => {
                if (request.hasError) { _isDepthReadingBack = false; return; }

                var data = request.GetData<ushort>();
                
                // R16_UNorm = 1 ushort per pixel [0, 65535] -> [0, 1]
                for (int i = 0; i < w * h; i++)
                {
                    _cpuDepthData[i] = data[i] / 65535.0f;
                }

                _cpuDepthTex.SetPixelData(_cpuDepthData, 0);
                _cpuDepthTex.Apply();

                _depthDataReady = true;
                _isDepthReadingBack = false;
            });
        }
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

        InitializeComputeBuffers();
    }

    private void InitializeComputeBuffers()
    {
        if (_pointCloudShader == null)
        {
            Debug.LogWarning("[IEExecutor] PointCloud ComputeShader is not assigned. GPU extraction disabled.");
            return;
        }

        _kernelId = _pointCloudShader.FindKernel("ExtractPointCloud");
        _debugKernelId = _pointCloudShader.FindKernel("DebugMask");
        _maskBuffer = new ComputeBuffer(160 * 160, sizeof(float));
        _rgbBuffer = new ComputeBuffer(_maxPoints, sizeof(uint));
        _positionBuffer = new ComputeBuffer(_maxPoints, sizeof(float) * 4);
        _colorBuffer = new ComputeBuffer(_maxPoints, sizeof(uint));
        // 4 counters: pointCount, maskPassCount, uvFailCount, depthFailCount
        _counterBuffer = new ComputeBuffer(4, sizeof(int));
        _computeBuffersInitialized = true;

        Debug.Log("[IEExecutor] Compute buffers initialized for GPU PointCloud extraction.");
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

                if (_captureRGBD)
                {
                    if (_computeBuffersInitialized)
                        ExtractPointCloudGPU(bestIndex, bestBox);
                    else
                        ExtractRGBDData(bestIndex, bestBox);  // 폴백
                }
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
    /// GPU Compute Shader를 사용하여 PointCloud 추출 (카메라 Intrinsics 기반)
    /// </summary>
    private void ExtractPointCloudGPU(int targetIndex, BoundingBox box)
    {
        if (!_computeBuffersInitialized || _pointCloudShader == null) return;
        if (_depthManager == null || !_depthManager.IsDepthAvailable) return;

        // 1. Depth 텍스처 가져오기 (Texture2DArray)
        RenderTexture depthRT = Shader.GetGlobalTexture("_EnvironmentDepthTexture") as RenderTexture;
        if (depthRT == null) return;

        // 2. ZBuffer params 가져오기
        Vector4 zParams = Shader.GetGlobalVector("_EnvironmentDepthZBufferParams");

        // 3. 카메라 intrinsics 및 pose 가져오기
        if (!_intrinsicsCached)
        {
            _cachedIntrinsics = PassthroughCameraSamples.PassthroughCameraUtils.GetCameraIntrinsics(
                PassthroughCameraSamples.PassthroughCameraEye.Left);
            _intrinsicsCached = true;
        }

        Pose cameraPose = PassthroughCameraSamples.PassthroughCameraUtils.GetCameraPoseInWorld(
            PassthroughCameraSamples.PassthroughCameraEye.Left);

        // [추가] 환경 깊이 재투영 행렬 가져오기 (Must read!!!.txt 참고)
        Matrix4x4[] reprojMatrices = Shader.GetGlobalMatrixArray("_EnvironmentDepthReprojectionMatrices");
        if (reprojMatrices != null && reprojMatrices.Length > 0)
        {
            _pointCloudShader.SetMatrix("_DepthReprojMatrix", reprojMatrices[0]);
        }

        // Intrinsics를 현재 RGB 해상도에 맞게 스케일링
        int rgbW = _rgbRenderTexture != null ? _rgbRenderTexture.width : 1280;
        int rgbH = _rgbRenderTexture != null ? _rgbRenderTexture.height : 1280;
        float intrinsicsScale = (float)rgbW / _cachedIntrinsics.Resolution.x;

        Vector4 scaledIntrinsics = new Vector4(
            _cachedIntrinsics.FocalLength.x * intrinsicsScale,
            _cachedIntrinsics.FocalLength.y * intrinsicsScale,
            _cachedIntrinsics.PrincipalPoint.x * intrinsicsScale,
            _cachedIntrinsics.PrincipalPoint.y * intrinsicsScale
        );

        // 4. 마스크 데이터 추출 및 업로드 (Y축 뒤집기 없이 원본 그대로)
        float[] maskData = new float[160 * 160];
        for (int y = 0; y < 160; y++)
        {
            for (int x = 0; x < 160; x++)
            {
                maskData[y * 160 + x] = _output3MaskWeights[targetIndex, y, x];
            }
        }
        _maskBuffer.SetData(maskData);

        // 5. RGB 데이터 업로드
        if (_rgbDataReady && _rgbPixelCache != null && _rgbPixelCache.Length > 0)
        {
            uint[] rgbPacked = new uint[_rgbPixelCache.Length];
            for (int i = 0; i < _rgbPixelCache.Length; i++)
            {
                Color32 c = _rgbPixelCache[i];
                rgbPacked[i] = ((uint)c.r) | ((uint)c.g << 8) | ((uint)c.b << 16) | ((uint)c.a << 24);
            }
            _rgbBuffer.SetData(rgbPacked);
        }

        // 6. Counter 리셋 (4개 모두 0으로)
        for (int i = 0; i < 4; i++) _counterData[i] = 0;
        _counterBuffer.SetData(_counterData);

        // 7. Compute Shader 유니폼 설정
        _pointCloudShader.SetVector("_CameraIntrinsics", scaledIntrinsics);
        _pointCloudShader.SetVector("_CameraPosition", new Vector4(cameraPose.position.x, cameraPose.position.y, cameraPose.position.z, 1));
        _pointCloudShader.SetMatrix("_CameraRotation", Matrix4x4.Rotate(cameraPose.rotation));
        _pointCloudShader.SetVector("_BBoxParams", new Vector4(box.CenterX, box.CenterY, box.Width, box.Height));
        _pointCloudShader.SetVector("_DepthZBufferParams", zParams);
        _pointCloudShader.SetFloat("_ConfidenceThreshold", _pointCloudConfidence);
        _pointCloudShader.SetInt("_RGBWidth", rgbW);
        _pointCloudShader.SetInt("_RGBHeight", rgbH);
        _pointCloudShader.SetInt("_DepthWidth", depthRT.width);
        _pointCloudShader.SetInt("_DepthHeight", depthRT.height);
        _pointCloudShader.SetInt("_MaxPoints", _maxPoints);
        _pointCloudShader.SetFloat("_MinDepth", _minDepthRange);
        _pointCloudShader.SetFloat("_MaxDepth", _maxDepthRange);

        // 8. 버퍼 바인딩
        _pointCloudShader.SetTexture(_kernelId, "_DepthTexture", depthRT);
        _pointCloudShader.SetBuffer(_kernelId, "_MaskBuffer", _maskBuffer);
        _pointCloudShader.SetBuffer(_kernelId, "_RGBBuffer", _rgbBuffer);
        _pointCloudShader.SetBuffer(_kernelId, "_PositionBuffer", _positionBuffer);
        _pointCloudShader.SetBuffer(_kernelId, "_ColorBuffer", _colorBuffer);
        _pointCloudShader.SetBuffer(_kernelId, "_CounterBuffer", _counterBuffer);

        // 9. Dispatch (160x160 / 8x8 = 20x20 thread groups)
        _pointCloudShader.Dispatch(_kernelId, 20, 20, 1);

        // 10. 결과 읽기 (동기)
        _counterBuffer.GetData(_counterData);
        CurrentPointCount = Mathf.Min(_counterData[0], _maxPoints);
        int maskPassCount = _counterData[1];
        int depthFailCount = _counterData[3];

        // 11. PointBuffer로 복사
        if (CurrentPointCount > 0)
        {
            Vector4[] positions = new Vector4[CurrentPointCount];
            uint[] colors = new uint[CurrentPointCount];
            _positionBuffer.GetData(positions, 0, 0, CurrentPointCount);
            _colorBuffer.GetData(colors, 0, 0, CurrentPointCount);

            for (int i = 0; i < CurrentPointCount; i++)
            {
                PointBuffer[i].worldPos = new Vector3(positions[i].x, positions[i].y, positions[i].z);
                uint c = colors[i];
                PointBuffer[i].color = new Color32((byte)(c & 0xFF), (byte)((c >> 8) & 0xFF), (byte)((c >> 16) & 0xFF), (byte)((c >> 24) & 0xFF));
            }
        }

        // 디버그 로그 (30프레임마다)
        if (Time.frameCount % 30 == 0)
        {
            float passRate = maskPassCount > 0 ? (float)CurrentPointCount / maskPassCount * 100f : 0f;
            Debug.Log($"[IEExecutor] GPU PointCloud: mask passed={maskPassCount}, depth fail={depthFailCount}, final points={CurrentPointCount} ({passRate:F1}%)");
            Debug.Log($"[IEExecutor] BBox: center=({box.CenterX:F1}, {box.CenterY:F1}), size=({box.Width:F1}x{box.Height:F1})");
        }
    }

    /// <summary>
    /// [레거시] CPU 기반 RGB 데이터 사용 - GPU 방식 실패시 폴백용
    /// IEMasker와 동일한 Y축 처리 방식 사용
    /// </summary>
    private void ExtractRGBDData(int targetIndex, BoundingBox box)
    {
        // 비동기 RGB 데이터가 준비되지 않았으면 스킵
        if (!_rgbDataReady || _cpuDepthTex == null) return;
        if (_rgbPixelCache == null || _rgbPixelCache.Length == 0) return;

        // 카메라 intrinsics 캐싱
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

        int rgbW = _rgbRenderTexture.width;
        int rgbH = _rgbRenderTexture.height;
        int depthW = _cpuDepthTex.width;
        int depthH = _cpuDepthTex.height;

        // Intrinsics를 현재 RGB 해상도에 맞게 스케일링
        float intrinsicsScale = (float)rgbW / _cachedIntrinsics.Resolution.x;
        float fx = _cachedIntrinsics.FocalLength.x * intrinsicsScale;
        float fy = _cachedIntrinsics.FocalLength.y * intrinsicsScale;
        float cx = _cachedIntrinsics.PrincipalPoint.x * intrinsicsScale;
        float cy = _cachedIntrinsics.PrincipalPoint.y * intrinsicsScale;

        // [수정] ushort -> float (RFloat 텍스처 대응)
        var depthData = _cpuDepthTex.GetRawTextureData<float>();

        // [추가] 전처리된 텍스처를 사용 중인지 확인 (이미 선형화되어 있음)
        bool isAlreadyLinear = Shader.GetGlobalTexture("_PreprocessedEnvironmentDepthTexture") != null;

        CurrentPointCount = 0;
        int maskPixelCount = 0;
        int depthFilteredCount = 0;
        int fallbackUsedCount = 0;
        float minDepth = float.MaxValue;
        float maxDepth = float.MinValue;
        float sumDepth = 0f;
        int depthSampleCount = 0;

        // ZBuffer params for linear depth conversion
        Vector4 zParams = Shader.GetGlobalVector("_EnvironmentDepthZBufferParams");

        // 1단계: 평균 depth 계산 (빠른 패스)
        float avgDepthForFallback = _fallbackDepth;
        if (_includeInvalidDepth && _fallbackDepth <= 0f)
        {
            float tempSum = 0f;
            int tempCount = 0;
            for (int y = 0; y < 160; y += _samplingStep)
            {
                for (int x = 0; x < 160; x += _samplingStep)
                {
                    // IEMasker와 동일한 Y축 뒤집기
                    int flippedY = 159 - y;
                    float maskVal = _output3MaskWeights[targetIndex, flippedY, x];
                    if (maskVal > _pointCloudConfidence)
                    {
                        Vector2Int rgbPixel = MapMaskToRGBPixelFloat(x, y, box, rgbW, rgbH);
                        float u = (float)rgbPixel.x / rgbW;
                        float v = (float)rgbPixel.y / rgbH;
                        int dx = Mathf.FloorToInt(u * (depthW - 1));
                        int dy = Mathf.FloorToInt(v * (depthH - 1));
                        
                        float rawDepth = depthData[dy * depthW + dx];
                        float d;
                        if (isAlreadyLinear)
                        {
                            d = rawDepth;
                        }
                        else
                        {
                            float depthNdc = rawDepth * 2.0f - 1.0f;
                            d = zParams.x / (depthNdc + zParams.y);
                        }

                        if (d > _minDepthRange && d < _maxDepthRange)
                        {
                            tempSum += d;
                            tempCount++;
                        }
                    }
                }
            }
            if (tempCount > 0) avgDepthForFallback = tempSum / tempCount;
        }

        // 2단계: 포인트 생성
        float subStep = 1.0f / _subSamplingFactor;

        for (int y = 0; y < 160; y += _samplingStep)
        {
            for (int x = 0; x < 160; x += _samplingStep)
            {
                if (CurrentPointCount >= _maxPoints) break;

                // IEMasker와 동일한 Y축 뒤집기
                int flippedY = 159 - y;
                float maskVal = _output3MaskWeights[targetIndex, flippedY, x];

                if (maskVal > _pointCloudConfidence)
                {
                    maskPixelCount++;

                    // 서브샘플링
                    for (int sy = 0; sy < _subSamplingFactor; sy++)
                    {
                        for (int sx = 0; sx < _subSamplingFactor; sx++)
                        {
                            if (CurrentPointCount >= _maxPoints) break;

                            float subMaskX = x + sx * subStep;
                            float subMaskY = y + sy * subStep;

                            Vector2Int rgbPixel = MapMaskToRGBPixelFloat(subMaskX, subMaskY, box, rgbW, rgbH);

                            int pixelIdx = rgbPixel.y * rgbW + rgbPixel.x;
                            if (pixelIdx < 0 || pixelIdx >= _rgbPixelCache.Length) continue;

                            float u = (float)rgbPixel.x / rgbW;
                            float v = (float)rgbPixel.y / rgbH;
                            int dx = Mathf.FloorToInt(u * (depthW - 1));
                            int dy = Mathf.FloorToInt(v * (depthH - 1));
                            
                            float rawDepth = depthData[dy * depthW + dx];
                            float depthMeters;

                            if (isAlreadyLinear)
                            {
                                depthMeters = rawDepth;
                            }
                            else
                            {
                                float depthNdc = rawDepth * 2.0f - 1.0f;
                                depthMeters = zParams.x / (depthNdc + zParams.y);
                            }

                            // 디버그용 depth 통계
                            if (sx == 0 && sy == 0 && depthMeters > _minDepthRange && depthMeters < _maxDepthRange)
                            {
                                minDepth = Mathf.Min(minDepth, depthMeters);
                                maxDepth = Mathf.Max(maxDepth, depthMeters);
                                sumDepth += depthMeters;
                                depthSampleCount++;
                            }

                            bool validDepth = depthMeters > _minDepthRange && depthMeters < _maxDepthRange;

                            if (!validDepth)
                            {
                                if (_includeInvalidDepth && avgDepthForFallback > _minDepthRange)
                                {
                                    depthMeters = avgDepthForFallback;
                                    if (sx == 0 && sy == 0) fallbackUsedCount++;
                                }
                                else
                                {
                                    if (sx == 0 && sy == 0) depthFilteredCount++;
                                    continue;
                                }
                            }

                            Vector3 dirInCamera = new Vector3(
                                (rgbPixel.x - cx) / fx,
                                (rgbPixel.y - cy) / fy,
                                1f
                            );
                            Vector3 dirInWorld = cameraRot * dirInCamera.normalized;
                            Vector3 worldPos = cameraPos + dirInWorld * depthMeters;

                            PointBuffer[CurrentPointCount].worldPos = worldPos;
                            PointBuffer[CurrentPointCount].color = _rgbPixelCache[pixelIdx];
                            CurrentPointCount++;
                        }
                    }
                }
            }
        }

        // 디버그 로그 (30프레임마다)
        if (Time.frameCount % 30 == 0)
        {
            float avgDepth = depthSampleCount > 0 ? sumDepth / depthSampleCount : 0f;
            float passRate = maskPixelCount > 0 ? (float)depthSampleCount / maskPixelCount * 100f : 0f;
            Debug.Log($"[IEExecutor] CPU PointCloud: mask={maskPixelCount}, fallback={fallbackUsedCount}, filtered={depthFilteredCount}, valid={CurrentPointCount} ({passRate:F1}%)");
            Debug.Log($"[IEExecutor] Depth range: min={minDepth:F3}m, max={maxDepth:F3}m, avg={avgDepth:F3}m");
        }
    }

    private Vector2Int MapMaskToRGBPixel(int maskX, int maskY, BoundingBox box, int rgbW, int rgbH)
    {
        // BBox는 640x640 YOLO 공간 기준, RGB는 1280x1280이므로 스케일 변환 필요
        float scaleYoloToRgb = rgbW / 640f;

        float normX = (float)maskX / 160f;
        float normY = (float)maskY / 160f;

        // YOLO 공간에서의 위치 계산 (CenterY: Top-Left origin 기준 오프셋, 아래로 갈수록 증가)
        float posInYoloX = box.CenterX - box.Width / 2f + normX * box.Width;
        float posInYoloY = box.CenterY - box.Height / 2f + normY * box.Height;

        // [수정] Y축 매핑: Unity Texture는 Bottom-Left origin.
        // YOLO Y 증가(아래로) -> Texture Y 감소(아래로)
        int pixelX = Mathf.RoundToInt(posInYoloX * scaleYoloToRgb + rgbW / 2f);
        int pixelY = Mathf.RoundToInt(rgbH / 2f - posInYoloY * scaleYoloToRgb);

        return new Vector2Int(Mathf.Clamp(pixelX, 0, rgbW - 1), Mathf.Clamp(pixelY, 0, rgbH - 1));
    }

    /// <summary>
    /// 서브픽셀 좌표를 지원하는 마스크→RGB 변환 (보간용)
    /// </summary>
    private Vector2Int MapMaskToRGBPixelFloat(float maskX, float maskY, BoundingBox box, int rgbW, int rgbH)
    {
        float scaleYoloToRgb = rgbW / 640f;

        float normX = maskX / 160f;
        float normY = maskY / 160f;

        float posInYoloX = box.CenterX - box.Width / 2f + normX * box.Width;
        float posInYoloY = box.CenterY - box.Height / 2f + normY * box.Height;

        // [수정] Y축 매핑: Unity Texture는 Bottom-Left origin.
        int pixelX = Mathf.RoundToInt(posInYoloX * scaleYoloToRgb + rgbW / 2f);
        int pixelY = Mathf.RoundToInt(rgbH / 2f - posInYoloY * scaleYoloToRgb);

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