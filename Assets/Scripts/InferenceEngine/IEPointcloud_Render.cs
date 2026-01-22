using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
public class IEPointcloud_Render : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private IEExecutor _executor;

    [Header("Settings")]
    [SerializeField] private bool _isRendering = true;

    private Mesh _pointMesh;
    private MeshFilter _meshFilter;
    
    // 월드 좌표를 저장해두는 리스트 (매 프레임 재계산용)
    private List<Vector3> _worldPositions = new List<Vector3>();
    private List<Color32> _colors = new List<Color32>();
    
    // 렌더링용 버퍼 (로컬 좌표 변환용)
    private List<Vector3> _renderVertices = new List<Vector3>();
    private List<int> _indices = new List<int>();

    void Start()
    {
        _meshFilter = GetComponent<MeshFilter>();
        _pointMesh = new Mesh();
        _pointMesh.MarkDynamic();
        _meshFilter.mesh = _pointMesh;
    }

    void LateUpdate()
    {
        if (!_isRendering || _executor == null) return;

        var pointBuffer = _executor.PointBuffer;
        int count = _executor.CurrentPointCount;

        if (count == 0)
        {
            if (_pointMesh.vertexCount > 0) _pointMesh.Clear();
            return;
        }

        // [핵심 로직] Executor의 버퍼 내용이 바뀔 때만 _worldPositions 업데이트
        // (간단히 매 프레임 동기화하되, 중요한 건 아래의 로컬 변환)
        if (_worldPositions.Count != count) 
        {
            _worldPositions.Clear();
            _colors.Clear();
            _indices.Clear();
            _renderVertices.Clear();
            
            for (int i = 0; i < count; i++)
            {
                _worldPositions.Add(pointBuffer[i].worldPos);
                _colors.Add(pointBuffer[i].color);
                _indices.Add(i);
                _renderVertices.Add(Vector3.zero); // 자리 확보
            }
        }
        else
        {
            // 내용 갱신
            for (int i = 0; i < count; i++)
            {
                _worldPositions[i] = pointBuffer[i].worldPos;
                _colors[i] = pointBuffer[i].color;
            }
        }

        // [Drift 방지] "카메라가 움직여도 점은 그 자리에 있어야 한다"
        // 렌더러(GameObject)가 카메라 자식으로 붙어있다면 transform.position이 계속 변함.
        // 따라서 고정된 월드 좌표(_worldPositions)를 현재의 transform 기준 로컬 좌표로 매번 변환해야 함.
        for (int i = 0; i < count; i++)
        {
            _renderVertices[i] = transform.InverseTransformPoint(_worldPositions[i]);
        }

        // 메쉬 업데이트
        _pointMesh.Clear();
        _pointMesh.SetVertices(_renderVertices);
        _pointMesh.SetColors(_colors);
        _pointMesh.SetIndices(_indices, MeshTopology.Points, 0);
        _pointMesh.RecalculateBounds();
    }
}