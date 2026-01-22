using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
public class IEPointcloud_Render : MonoBehaviour
{
    [SerializeField] private IEExecutor _executor;
    [SerializeField] private bool _isRendering = true;

    private Mesh _pointMesh;
    private MeshFilter _meshFilter;
    
    // 리스트 재사용 (GC 방지)
    private List<Vector3> _vertices = new List<Vector3>();
    private List<Color32> _colors = new List<Color32>();
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
        UpdatePointCloud();
    }

    private void UpdatePointCloud()
    {
        var pointBuffer = _executor.PointBuffer;
        int count = _executor.CurrentPointCount;

        // [완벽 방어] 
        // 0개라면 아예 렌더링 파이프라인을 건드리지 않고 리턴해버립니다. 
        // 이렇게 하면 GPU에 올라가 있는 이전 프레임의 메쉬가 그대로 유지됩니다.
        if (count == 0) return;

        _vertices.Clear();
        _colors.Clear();
        _indices.Clear();

        // 버퍼에서 데이터 옮겨담기
        for (int i = 0; i < count; i++)
        {
            // 로컬 좌표 변환이 필요하다면 사용 (여기서는 월드 좌표를 그대로 쓰거나 변환)
            // 보통 PointCloud는 월드 좌표로 계산되므로 InverseTransformPoint 필요
            _vertices.Add(transform.InverseTransformPoint(pointBuffer[i].worldPos));
            _colors.Add(pointBuffer[i].color);
            _indices.Add(i);
        }

        // 메쉬 갱신
        _pointMesh.Clear();
        _pointMesh.SetVertices(_vertices);
        _pointMesh.SetColors(_colors);
        _pointMesh.SetIndices(_indices, MeshTopology.Points, 0);
        _pointMesh.RecalculateBounds(); // 바운드 재계산 필수
    }
}