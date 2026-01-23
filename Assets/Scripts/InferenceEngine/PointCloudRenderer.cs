using System.Collections.Generic;
using UnityEngine;

public class PointCloudRenderer : MonoBehaviour
{
    [Header("Rendering Settings")]
    [SerializeField] private Material _pointCloudMaterial;
    [SerializeField] private float _pointSize = 0.01f;

    private Mesh _mesh;
    private MeshFilter _meshFilter;
    private MeshRenderer _meshRenderer;

    // 캡처 당시의 기준 데이터
    private float _initialDepth;
    private float _initialBoxSize; // 대각선 길이 또는 면적 기준

    public bool IsMeshGenerated { get; private set; } = false;

    private void Awake()
    {
        _meshFilter = gameObject.AddComponent<MeshFilter>();
        _meshRenderer = gameObject.AddComponent<MeshRenderer>();
        
        if (_pointCloudMaterial == null)
        {
            // 1. 커스텀 셰이더 시도
            var shader = Shader.Find("Custom/PointcloudShader");
            
            // 2. 실패 시 기본 Unlit 시도
            if (shader == null) shader = Shader.Find("Unlit/Color");
            
            // 3. 그래도 없으면 스탠다드 (최후의 수단)
            if (shader == null) shader = Shader.Find("Standard");

            if (shader != null)
            {
                _pointCloudMaterial = new Material(shader);
            }
            else
            {
                Debug.LogError("[PointCloudRenderer] Failed to find any shader for Point Cloud!");
            }
        }

        if (_pointCloudMaterial != null)
        {
            _meshRenderer.material = _pointCloudMaterial;
        }

        _mesh = new Mesh();
        _mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
        _meshFilter.mesh = _mesh;
    }

    /// <summary>
    /// [Snapshot] 1회 실행: 수집된 포인트 데이터로 메쉬 생성
    /// </summary>
    public void GenerateMesh(List<Vector3> worldPoints, List<Color32> colors, BoundingBox initialBox, float centerDepth)
    {
        if (worldPoints.Count == 0) return;

        // 1. Mesh 데이터 채우기
        _mesh.Clear();
        
        // 좌표계를 로컬로 변환 (Mesh의 피벗을 물체 중심으로 맞추기 위함)
        Vector3 centerPos = initialBox.WorldPos.HasValue ? initialBox.WorldPos.Value : worldPoints[0];
        
        Vector3[] vertices = new Vector3[worldPoints.Count];
        int[] indices = new int[worldPoints.Count];
        Color32[] meshColors = new Color32[worldPoints.Count];

        for (int i = 0; i < worldPoints.Count; i++)
        {
            // 월드 좌표 -> 로컬 좌표 (중심점 기준)
            vertices[i] = worldPoints[i] - centerPos; 
            indices[i] = i;
            meshColors[i] = colors[i];
        }

        _mesh.vertices = vertices;
        _mesh.colors32 = meshColors;
        _mesh.SetIndices(indices, MeshTopology.Points, 0);
        
        // 2. 초기 상태 저장 (Tracking을 위해)
        transform.position = centerPos;
        transform.rotation = Quaternion.identity; // 포인트 클라우드는 월드 정렬됨
        
        _initialDepth = centerDepth;
        // 박스 크기 척도: 가로/세로의 평균을 사용
        _initialBoxSize = (initialBox.Width + initialBox.Height) * 0.5f;

        IsMeshGenerated = true;
        gameObject.SetActive(true);
        
        Debug.Log($"[PointCloud] Mesh Generated. Points: {worldPoints.Count}, InitDepth: {_initialDepth:F2}m, InitSize: {_initialBoxSize:F1}");
    }

    /// <summary>
    /// [Update] 매 프레임 실행: 2D 박스 변화에 맞춰 3D 위치/스케일 동기화
    /// </summary>
    public void UpdateTransform(BoundingBox currentBox, Ray cameraRay)
    {
        if (!IsMeshGenerated) return;

        float currentBoxSize = (currentBox.Width + currentBox.Height) * 0.5f;
        
        // 0으로 나누기 방지
        if (currentBoxSize < 1f || _initialBoxSize < 1f) return;

        // 1. 거리(Depth) 추정
        // 원리: 물체가 멀어지면 2D 박스가 작아짐. (Scale 비율의 역수)
        // NewDepth = InitialDepth * (InitialSize / CurrentSize)
        float scaleRatio = _initialBoxSize / currentBoxSize;
        float newDepth = _initialDepth * scaleRatio;

        // 2. 3D 위치 계산
        // 카메라에서 2D 박스 중심을 향해 쏘는 Ray 상에서, 추정된 Depth만큼 떨어진 위치
        Vector3 newWorldPos = cameraRay.GetPoint(newDepth);

        // 3. Transform 적용
        transform.position = newWorldPos;

        // (선택사항) 스케일 보정?
        // 포인트 클라우드 자체의 크기는 '실제 크기'이므로 스케일을 조절할 필요는 없음.
        // 다만, 줌인/줌아웃 효과를 원한다면 transform.localScale을 건드릴 수 있음.
        // 여기서는 위치만 이동시킵니다.
    }

    public void ClearMesh()
    {
        if (_mesh != null)
        {
            _mesh.Clear();
        }
        IsMeshGenerated = false;
        
        if (gameObject != null)
        {
            gameObject.SetActive(false);
        }
    }
}
