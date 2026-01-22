using System;
using System.Collections;
using System.Collections.Generic;
using Unity.InferenceEngine;
using UnityEngine;

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
    [SerializeField] private BackendType _backend = BackendType.CPU;
    [SerializeField] private ModelAsset _sentisModel;
    [SerializeField] private int _layersPerFrame = 25;
    [SerializeField] private float _confidenceThreshold = 0.5f;
    [SerializeField] private Transform _displayLocation;

    [Header("Natural Tracking Settings")]
    [SerializeField] private int _maxLostFrames = 15; // [상향] 가려졌을 때 더 오래 버티도록 10 -> 15 프레임으로 늘림
    [SerializeField] private float _minIoUThreshold = 0.3f;
    
    // 적응형 스무딩 범위
    [SerializeField] private float _minSmoothTime = 0.03f;
    [SerializeField] private float _maxSmoothTime = 0.2f; 
    [SerializeField] private float _sizeSmoothTime = 0.3f; 
    
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
    
    // 추적 관련 변수
    private BoundingBox? _lockedTargetBox = null;
    private bool _isTracking = false;
    private int _consecutiveLostFrames = 0;
    private int _lockedTargetIndex = -1;
    // 물리적 움직임 계산용
    private Vector2 _centerVelocity;
    private Vector2 _sizeVelocity;
    private float _currentSmoothTime;

    private bool _started = false;
    private bool _isWaitingForReadbackRequest = false;
    private List<BoundingBox> _currentFrameBoxes = new();
    public bool IsTracking => _isTracking;
    public BoundingBox? LockedTargetBox => _lockedTargetBox;
    public int LockedTargetIndex => _lockedTargetIndex;
    public Tensor<float> OutputMasks => _output2Masks;
    public Tensor<float> OutputMaskWeights => _output3MaskWeights;
    public Vector2Int InputSize => _inputSize;    
    private IEnumerator Start()
    {
        yield return new WaitForSeconds(0.05f);
        _ieMasker = new IEMasker(_displayLocation, _confidenceThreshold);
        LoadModel();
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
        else
        {
            UpdateProcessInferenceResults();
        }
    }

    private void UpdateProcessInferenceResults()
    {
        switch (_downloadState)
        {
            case InferenceDownloadState.RequestingOutput0:
                HandleReadback(0, ref _output0BoxCoords, ref _downloadState, InferenceDownloadState.RequestingOutput1);
                break;
            case InferenceDownloadState.RequestingOutput1:
                HandleReadback(1, ref _output1LabelIds, ref _downloadState, InferenceDownloadState.RequestingOutput2);
                break;
            case InferenceDownloadState.RequestingOutput2:
                HandleReadback(2, ref _output2Masks, ref _downloadState, InferenceDownloadState.RequestingOutput3);
                break;
            case InferenceDownloadState.RequestingOutput3:
                HandleReadback(3, ref _output3MaskWeights, ref _downloadState, InferenceDownloadState.Success);
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

    private void HandleReadback<T>(int outputIndex, ref Tensor<T> targetTensor, ref InferenceDownloadState currentState, InferenceDownloadState nextState) where T : unmanaged
    {
        if (!_isWaitingForReadbackRequest)
        {
            _buffer = GetOutputBuffer(outputIndex);
            InitiateReadbackRequest(_buffer);
        }
        else
        {
            if (_buffer.IsReadbackRequestDone())
            {
                targetTensor = _buffer.ReadbackAndClone() as Tensor<T>;
                _isWaitingForReadbackRequest = false;

                if (targetTensor.shape[0] > 0)
                    currentState = nextState;
                else
                    currentState = InferenceDownloadState.Error;

                _buffer?.Dispose();
            }
        }
    }

    private void ProcessSuccessState()
    {
        List<BoundingBox> currentFrameBoxes = _ieBoxer.DrawBoxes(_output0BoxCoords, _output1LabelIds, _inputSize.x, _inputSize.y);
        _currentFrameBoxes = currentFrameBoxes;

        if (_isTracking && _lockedTargetBox.HasValue)
        {
            float bestScore = 0f;
            int bestIndex = -1;
            BoundingBox bestBox = default;

            Vector2 prevCenter = new Vector2(_lockedTargetBox.Value.CenterX, _lockedTargetBox.Value.CenterY);

            for (int i = 0; i < currentFrameBoxes.Count; i++)
            {
                BoundingBox currBox = currentFrameBoxes[i];
                
                float iou = TrackingUtils.CalculateIoU(_lockedTargetBox.Value, currBox);
                float dist = Vector2.Distance(prevCenter, new Vector2(currBox.CenterX, currBox.CenterY));
                float maxDist = _inputSize.x * 0.5f; 
                float distScore = 1.0f - Mathf.Clamp01(dist / maxDist);

                float totalScore = (iou * 0.7f) + (distScore * 0.3f);

                if (iou > _minIoUThreshold && totalScore > bestScore)
                {
                    bestScore = totalScore;
                    bestIndex = i;
                    bestBox = currBox;
                }
            }

            if (bestIndex != -1)
            {
                // [성공] 물체를 찾음 -> 마스크 갱신
                _consecutiveLostFrames = 0;
                _lockedTargetIndex = bestIndex; // [추가] 인덱스 저장
                float speed = _centerVelocity.magnitude;
                float targetSmoothTime = Mathf.Lerp(_maxSmoothTime, _minSmoothTime, speed / 500f);
                _currentSmoothTime = Mathf.Lerp(_currentSmoothTime, targetSmoothTime, 0.1f);

                Vector2 currentPos = new Vector2(_lockedTargetBox.Value.CenterX, _lockedTargetBox.Value.CenterY);
                Vector2 targetPos = new Vector2(bestBox.CenterX, bestBox.CenterY);
                Vector2 smoothedPos = Vector2.SmoothDamp(currentPos, targetPos, ref _centerVelocity, _currentSmoothTime);

                Vector2 currentSize = new Vector2(_lockedTargetBox.Value.Width, _lockedTargetBox.Value.Height);
                Vector2 targetSize = new Vector2(bestBox.Width, bestBox.Height);
                Vector2 smoothedSize = Vector2.SmoothDamp(currentSize, targetSize, ref _sizeVelocity, _sizeSmoothTime);

                _lockedTargetBox = new BoundingBox
                {
                    CenterX = smoothedPos.x,
                    CenterY = smoothedPos.y,
                    Width = smoothedSize.x,
                    Height = smoothedSize.y,
                    ClassName = bestBox.ClassName,
                    Label = bestBox.Label,
                    WorldPos = bestBox.WorldPos
                };

                // 새로운 마스크 그리기
                _ieMasker.DrawSingleMask(bestIndex, bestBox, _output3MaskWeights, _inputSize.x, _inputSize.y);
            }
            else
            {
                // [실패] 물체를 놓침 (가려짐 등)
                _consecutiveLostFrames++;
                _lockedTargetIndex = -1; // 놓쳤을 땐 인덱스 무효화
                if (_consecutiveLostFrames <= _maxLostFrames)
                {
                    // [핵심 변경 사항]
                    // 놓쳤더라도 허용 범위 내라면, 마스크를 지우지 않습니다.
                    // 이전에 그려진 마스크(RawImage Texture)가 그대로 화면에 남아서 "유지"되는 효과를 줍니다.
                    // _ieMasker.DrawSingleMask(-1, ...) <--- 이 코드를 제거했습니다.

                    // 박스 위치는 관성에 따라 계속 예측 이동 (내부 데이터만 갱신)
                    float deltaTime = Time.deltaTime; 
                    float predX = _lockedTargetBox.Value.CenterX + (_centerVelocity.x * deltaTime);
                    float predY = _lockedTargetBox.Value.CenterY + (_centerVelocity.y * deltaTime);
                    _centerVelocity *= 0.9f; 

                    _lockedTargetBox = new BoundingBox
                    {
                        CenterX = predX,
                        CenterY = predY,
                        Width = _lockedTargetBox.Value.Width,
                        Height = _lockedTargetBox.Value.Height,
                        ClassName = _lockedTargetBox.Value.ClassName,
                        Label = _lockedTargetBox.Value.Label,
                        WorldPos = _lockedTargetBox.Value.WorldPos
                    };
                }
                else
                {
                    // 너무 오래 놓치면 그때 리셋 (마스크 사라짐)
                    Debug.Log("Tracking Lost (Timeout)");
                    ResetTracking();
                }
            }
        }
        else
        {
            // 애초에 추적 중이 아니면 마스크 지우기
            _ieMasker.DrawSingleMask(-1, default, null, _inputSize.x, _inputSize.y);
        }
    }

    private void CleanupResources()
    {
        _output0BoxCoords?.Dispose();
        _output1LabelIds?.Dispose();
        _output2Masks?.Dispose();
        _output3MaskWeights?.Dispose();
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

            float halfW = box.Width / 2f;
            float halfH = box.Height / 2f;

            float margin = 30f; 
            if (screenPos.x >= boxScreenX - halfW - margin && screenPos.x <= boxScreenX + halfW + margin &&
                screenPos.y >= boxScreenY - halfH - margin && screenPos.y <= boxScreenY + halfH + margin)
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
        
        if (_ieMasker != null) 
             _ieMasker.DrawSingleMask(-1, default, null, _inputSize.x, _inputSize.y);
        
        Debug.Log("[IEExecutor] Tracking Reset");
    }

    private Tensor GetOutputBuffer(int outputIndex)
    {
        return _inferenceEngineWorker.PeekOutput(outputIndex);
    }

    private void InitiateReadbackRequest(Tensor pullTensor)
    {
        if (pullTensor.dataOnBackend != null)
        {
            pullTensor.ReadbackRequest();
            _isWaitingForReadbackRequest = true;
        }
        else
        {
            _downloadState = InferenceDownloadState.Error;
        }
    }
}