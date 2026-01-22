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
    
    // GC(Garbage Collector) 방지를 위해 리스트 버퍼를 재사용합니다.
    private List<Vector3> _vertices = new List<Vector3>();
    private List<Color32> _colors = new List<Color32>();
    private List<int> _indices = new List<int>();

    void Start()
    {
        _meshFilter = GetComponent<MeshFilter>();
        _pointMesh = new Mesh();
        _pointMesh.MarkDynamic(); // 매 프레임 업데이트되는 메쉬 성능 최적화
        _meshFilter.mesh = _pointMesh;
    }

    void LateUpdate()
    {
        if (!_isRendering || _executor == null) return;

        UpdatePointCloud();
    }

    private void UpdatePointCloud()
    {
        // 에러 원인 해결: LastTrackedObjectPoints 대신 PointBuffer와 CurrentPointCount 사용
        var pointBuffer = _executor.PointBuffer;
        int count = _executor.CurrentPointCount;

        // 데이터가 없으면 메쉬를 비우고 리턴
        if (pointBuffer == null || count == 0)
        {
            _pointMesh.Clear();
            return;
        }

        _vertices.Clear();
        _colors.Clear();
        _indices.Clear();

        // [최적화] 전체 배열이 아니라 현재 유효한 데이터 개수(count)만큼만 루프를 돕니다.
        for (int i = 0; i < count; i++)
        {
            // 월드 좌표를 렌더러의 로컬 좌표로 변환
            _vertices.Add(transform.InverseTransformPoint(pointBuffer[i].worldPos));
            _colors.Add(pointBuffer[i].color);
            _indices.Add(i);
        }

        // 메쉬 데이터 갱신
        _pointMesh.Clear();
        if (_vertices.Count > 0)
        {
            _pointMesh.SetVertices(_vertices);
            _pointMesh.SetColors(_colors);
            _pointMesh.SetIndices(_indices, MeshTopology.Points, 0);
            _pointMesh.RecalculateBounds();
        }
    }
}