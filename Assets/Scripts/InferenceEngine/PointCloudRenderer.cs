using System.Collections.Generic;
using UnityEngine;

public class PointCloudRenderer : MonoBehaviour
{
    [Header("Rendering Settings")]
    [SerializeField] private Material _pointCloudMaterial;
    [SerializeField] private float _pointSize = 0.01f;
    [SerializeField] private float _meshConnectionDistance = 0.15f; // 15cm 이상 차이나면 연결 안함

    private Mesh _mesh;
    private MeshFilter _meshFilter;
    private MeshRenderer _meshRenderer;

    public bool IsMeshGenerated { get; private set; } = false;

    private void Awake()
    {
        _meshFilter = gameObject.AddComponent<MeshFilter>();
        _meshRenderer = gameObject.AddComponent<MeshRenderer>();
        
        if (_pointCloudMaterial == null)
        {
            var shader = Shader.Find("Custom/PointcloudShader");
            if (shader == null) shader = Shader.Find("Unlit/Color");
            if (shader == null) shader = Shader.Find("Standard");

            if (shader != null)
            {
                _pointCloudMaterial = new Material(shader);
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
    /// [Points Mode] 점만 찍기 (기존 방식)
    /// </summary>
    public void GeneratePointMesh(List<Vector3> worldPoints, List<Color32> colors)
    {
        if (worldPoints.Count == 0) return;

        _mesh.Clear();
        
        Vector3 centerPos = worldPoints[0];
        
        Vector3[] vertices = new Vector3[worldPoints.Count];
        int[] indices = new int[worldPoints.Count];
        Color32[] meshColors = new Color32[worldPoints.Count];

        for (int i = 0; i < worldPoints.Count; i++)
        {
            vertices[i] = worldPoints[i] - centerPos; 
            indices[i] = i;
            meshColors[i] = colors[i];
        }

        _mesh.vertices = vertices;
        _mesh.colors32 = meshColors;
        _mesh.SetIndices(indices, MeshTopology.Points, 0);
        
        transform.position = centerPos;
        transform.rotation = Quaternion.identity;

        IsMeshGenerated = true;
        gameObject.SetActive(true);
    }

    /// <summary>
    /// [Triangles Mode] 격자 기반 메쉬 생성 (면 만들기)
    /// </summary>
    public void GenerateTriangleMesh(Vector3[] worldGrid, Color32[] colorGrid, bool[] validGrid, int width, int height, Vector3 centerPos)
    {
        _mesh.Clear();

        List<Vector3> vertices = new List<Vector3>();
        List<Color32> colors = new List<Color32>();
        List<int> triangles = new List<int>();

        // 1. 버텍스 생성 (월드 -> 로컬 변환)
        // gridIndex -> vertexIndex 매핑 테이블
        int[] gridToVertIndex = new int[worldGrid.Length]; 
        for (int i = 0; i < gridToVertIndex.Length; i++) gridToVertIndex[i] = -1;

        for (int i = 0; i < worldGrid.Length; i++)
        {
            if (validGrid[i])
            {
                gridToVertIndex[i] = vertices.Count;
                vertices.Add(worldGrid[i] - centerPos);
                colors.Add(colorGrid[i]);
            }
        }

        // 2. 삼각형 연결 (Quad Topology)
        for (int y = 0; y < height - 1; y++)
        {
            for (int x = 0; x < width - 1; x++)
            {
                // 현재 픽셀과 오른쪽, 아래, 대각선 픽셀 인덱스
                int idxTL = y * width + x;       // Top-Left
                int idxTR = y * width + (x + 1); // Top-Right
                int idxBL = (y + 1) * width + x; // Bottom-Left
                int idxBR = (y + 1) * width + (x + 1); // Bottom-Right

                // 4점이 모두 유효한지 확인
                if (validGrid[idxTL] && validGrid[idxTR] && validGrid[idxBL] && validGrid[idxBR])
                {
                    // 깊이 차이가 너무 크면 연결 안 함 (경계면 처리)
                    float d1 = Vector3.Distance(worldGrid[idxTL], worldGrid[idxBR]);
                    float d2 = Vector3.Distance(worldGrid[idxTR], worldGrid[idxBL]);

                    if (d1 < _meshConnectionDistance && d2 < _meshConnectionDistance)
                    {
                        int vTL = gridToVertIndex[idxTL];
                        int vTR = gridToVertIndex[idxTR];
                        int vBL = gridToVertIndex[idxBL];
                        int vBR = gridToVertIndex[idxBR];

                        // Triangle 1 (TL - TR - BL)
                        triangles.Add(vTL); triangles.Add(vTR); triangles.Add(vBL);
                        // Triangle 2 (TR - BR - BL)
                        triangles.Add(vTR); triangles.Add(vBR); triangles.Add(vBL);
                    }
                }
            }
        }

        _mesh.SetVertices(vertices);
        _mesh.SetColors(colors);
        _mesh.SetTriangles(triangles, 0);
        _mesh.RecalculateNormals(); 

        transform.position = centerPos;
        transform.rotation = Quaternion.identity;

        // Vertex Color를 지원하는 shader 사용 (우선순위: Custom > Particles > Standard)
        var surfShader = Shader.Find("Custom/MeshSurfaceShader");
        if (surfShader == null) surfShader = Shader.Find("Particles/Standard Unlit");
        if (surfShader == null) surfShader = Shader.Find("Unlit/Color");

        if (surfShader != null)
        {
            var mat = new Material(surfShader);

            // Particles/Standard Unlit의 경우 Vertex Color 모드 설정
            if (surfShader.name.Contains("Particles"))
            {
                mat.SetFloat("_ColorMode", 1); // Vertex Color
            }

            _meshRenderer.material = mat;
            Debug.Log($"[PointCloudRenderer] Using shader: {surfShader.name}");
        }
        else
        {
            Debug.LogError("[PointCloudRenderer] Failed to find any compatible shader!");
        }

        _mesh.RecalculateBounds();

        IsMeshGenerated = true;
        gameObject.SetActive(true);

        // 디버그: 위치 및 바운드 정보 출력
        Debug.Log($"[PointCloudRenderer] Mesh Surface Generated. Verts: {vertices.Count}, Tris: {triangles.Count / 3}");
        Debug.Log($"[PointCloudRenderer] Center Position: {centerPos}");
        Debug.Log($"[PointCloudRenderer] Mesh Bounds: center={_mesh.bounds.center}, size={_mesh.bounds.size}");
        Debug.Log($"[PointCloudRenderer] World Bounds: center={transform.TransformPoint(_mesh.bounds.center)}, size={_mesh.bounds.size}");

        // 카메라와의 거리 체크
        if (Camera.main != null)
        {
            float distToCamera = Vector3.Distance(Camera.main.transform.position, centerPos);
            Debug.Log($"[PointCloudRenderer] Distance to camera: {distToCamera:F2}m");
        }
    }

    public void UpdateTransform(BoundingBox currentBox, Ray cameraRay)
    {
        // 현재는 고정 모드를 사용하므로 내용 비움 (필요시 구현)
    }

    public void ClearMesh()
    {
        if (_mesh != null) _mesh.Clear();
        IsMeshGenerated = false;
        if (gameObject != null) gameObject.SetActive(false);
    }
}
