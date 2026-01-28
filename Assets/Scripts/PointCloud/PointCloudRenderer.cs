using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// IEExecutor의 PointBuffer 데이터를 GPU로 전송하여 PointCloud를 렌더링합니다.
/// Quest 3 최적화: 단일 드로우콜, GraphicsBuffer 사용, Unlit 셰이더
/// </summary>
public class PointCloudRenderer : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private IEExecutor _ieExecutor;
    [SerializeField] private Material _pointMaterial;

    [Header("Rendering Settings")]
    [SerializeField] private float _pointSize = 0.03f;
    [SerializeField] private bool _sizeAttenuation = true;
    [SerializeField] private float _attenuationMinDist = 0.5f;
    [SerializeField] private float _attenuationMaxDist = 3.0f;

    [Header("Debug")]
    [SerializeField] private bool _showDebugInfo = false;

    private GraphicsBuffer _positionBuffer;
    private GraphicsBuffer _colorBuffer;
    private int _maxPoints = 102400;
    private int _lastPointCount;

    private Mesh _pointMesh;
    private MaterialPropertyBlock _propertyBlock;

    private static readonly int PositionBufferID = Shader.PropertyToID("_PositionBuffer");
    private static readonly int ColorBufferID = Shader.PropertyToID("_ColorBuffer");
    private static readonly int PointSizeID = Shader.PropertyToID("_PointSize");
    private static readonly int SizeAttenuationID = Shader.PropertyToID("_SizeAttenuation");
    private static readonly int AttenuationMinDistID = Shader.PropertyToID("_AttenuationMinDist");
    private static readonly int AttenuationMaxDistID = Shader.PropertyToID("_AttenuationMaxDist");

    private Vector3[] _positionArray;
    private uint[] _colorArray;

    private void Start()
    {
        if (_ieExecutor == null)
        {
            Debug.LogError("[PointCloudRenderer] IEExecutor reference is missing!");
            enabled = false;
            return;
        }

        if (_pointMaterial == null)
        {
            Debug.LogError("[PointCloudRenderer] Point Material is missing!");
            enabled = false;
            return;
        }

        InitializeBuffers();
        InitializeMesh();

        _propertyBlock = new MaterialPropertyBlock();
        _positionArray = new Vector3[_maxPoints];
        _colorArray = new uint[_maxPoints];
    }

    private void InitializeBuffers()
    {
        _positionBuffer = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured,
            _maxPoints,
            sizeof(float) * 3
        );

        _colorBuffer = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured,
            _maxPoints,
            sizeof(uint)
        );

        Debug.Log($"[PointCloudRenderer] Initialized buffers for {_maxPoints} points");
    }

    private void InitializeMesh()
    {
        _pointMesh = new Mesh();
        _pointMesh.name = "PointCloudMesh";

        Vector3[] vertices = new Vector3[_maxPoints];
        int[] indices = new int[_maxPoints];

        for (int i = 0; i < _maxPoints; i++)
        {
            vertices[i] = Vector3.zero;
            indices[i] = i;
        }

        _pointMesh.vertices = vertices;
        _pointMesh.SetIndices(indices, MeshTopology.Points, 0);
        _pointMesh.bounds = new Bounds(Vector3.zero, Vector3.one * 100f);

        Debug.Log("[PointCloudRenderer] Initialized point mesh");
    }

    private void LateUpdate()
    {
        if (_ieExecutor == null || !_ieExecutor.IsTracking)
        {
            _lastPointCount = 0;
            return;
        }

        int currentCount = _ieExecutor.CurrentPointCount;
        if (currentCount <= 0) return;

        UpdateBuffers(currentCount);
        RenderPoints(currentCount);

        if (_showDebugInfo && Time.frameCount % 60 == 0)
        {
            Debug.Log($"[PointCloudRenderer] Rendering {currentCount} points");
        }
    }

    private void UpdateBuffers(int pointCount)
    {
        int safeCount = Mathf.Min(pointCount, _maxPoints);

        Vector3 min = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
        Vector3 max = new Vector3(float.MinValue, float.MinValue, float.MinValue);

        for (int i = 0; i < safeCount; i++)
        {
            _positionArray[i] = _ieExecutor.PointBuffer[i].worldPos;

            Color32 c = _ieExecutor.PointBuffer[i].color;
            _colorArray[i] = (uint)(c.r | (c.g << 8) | (c.b << 16) | (c.a << 24));

            // Bounds 계산용 min/max 갱신
            min = Vector3.Min(min, _positionArray[i]);
            max = Vector3.Max(max, _positionArray[i]);
        }

        _positionBuffer.SetData(_positionArray, 0, 0, safeCount);
        _colorBuffer.SetData(_colorArray, 0, 0, safeCount);

        _lastPointCount = safeCount;

        // 동적 bounds 업데이트
        if (safeCount > 0)
        {
            Vector3 center = (min + max) * 0.5f;
            Vector3 size = (max - min) + Vector3.one * 0.1f;
            _pointMesh.bounds = new Bounds(center, size);
        }

        // 디버그 로그 (30프레임마다)
        if (_showDebugInfo && Time.frameCount % 30 == 0)
        {
            Debug.Log($"[PointCloudRenderer] Points={safeCount}, BoundsCenter={_pointMesh.bounds.center:F3}, BoundsSize={_pointMesh.bounds.size:F3}");
            Debug.Log($"[PointCloudRenderer] PositionRange: min={min:F3}, max={max:F3}");
        }
    }

    private void RenderPoints(int pointCount)
    {
        _propertyBlock.SetBuffer(PositionBufferID, _positionBuffer);
        _propertyBlock.SetBuffer(ColorBufferID, _colorBuffer);
        _propertyBlock.SetFloat(PointSizeID, _pointSize);
        _propertyBlock.SetFloat(SizeAttenuationID, _sizeAttenuation ? 1.0f : 0.0f);
        _propertyBlock.SetFloat(AttenuationMinDistID, _attenuationMinDist);
        _propertyBlock.SetFloat(AttenuationMaxDistID, _attenuationMaxDist);

        Graphics.DrawMesh(
            _pointMesh,
            Matrix4x4.identity,
            _pointMaterial,
            0,
            null,
            0,
            _propertyBlock,
            ShadowCastingMode.Off,
            false
        );
    }

    private void OnDestroy()
    {
        _positionBuffer?.Dispose();
        _colorBuffer?.Dispose();

        if (_pointMesh != null)
            Destroy(_pointMesh);

        Debug.Log("[PointCloudRenderer] Cleaned up resources");
    }

    public void SetPointSize(float size)
    {
        _pointSize = Mathf.Max(0.001f, size);
    }

    public void SetSizeAttenuation(bool enabled)
    {
        _sizeAttenuation = enabled;
    }

    public int GetCurrentPointCount()
    {
        return _lastPointCount;
    }
}
