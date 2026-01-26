using System.Collections.Generic;
using UnityEngine;

public class PointCloudRenderer : MonoBehaviour
{
    [Header("Rendering Settings")]
    [SerializeField] private Material _pointCloudMaterial;
    [SerializeField] private Material _meshSurfaceMaterial;  // Inspector에서 MeshSurfaceShader 머티리얼 설정
    [SerializeField] private float _pointSize = 0.01f;
    [SerializeField] private float _meshConnectionDistance = 0.05f; // 5cm 이상 차이나면 연결 안함 (기둥 방지)

    [Header("Debug")]
    [SerializeField] private bool _createDebugCube = true;
    private GameObject _debugCube;

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

        // 2. 삼각형 연결 - 개선된 알고리즘 (기둥 방지)
        for (int y = 0; y < height - 1; y++)
        {
            for (int x = 0; x < width - 1; x++)
            {
                int idxTL = y * width + x;
                int idxTR = y * width + (x + 1);
                int idxBL = (y + 1) * width + x;
                int idxBR = (y + 1) * width + (x + 1);

                bool vTL = validGrid[idxTL];
                bool vTR = validGrid[idxTR];
                bool vBL = validGrid[idxBL];
                bool vBR = validGrid[idxBR];

                // 삼각형 1: TL-TR-BL (3점이 유효하고 모든 엣지가 연결 가능할 때)
                if (vTL && vTR && vBL)
                {
                    if (IsEdgeValid(worldGrid[idxTL], worldGrid[idxTR]) &&
                        IsEdgeValid(worldGrid[idxTR], worldGrid[idxBL]) &&
                        IsEdgeValid(worldGrid[idxBL], worldGrid[idxTL]))
                    {
                        triangles.Add(gridToVertIndex[idxTL]);
                        triangles.Add(gridToVertIndex[idxTR]);
                        triangles.Add(gridToVertIndex[idxBL]);
                    }
                }

                // 삼각형 2: TR-BR-BL (3점이 유효하고 모든 엣지가 연결 가능할 때)
                if (vTR && vBR && vBL)
                {
                    if (IsEdgeValid(worldGrid[idxTR], worldGrid[idxBR]) &&
                        IsEdgeValid(worldGrid[idxBR], worldGrid[idxBL]) &&
                        IsEdgeValid(worldGrid[idxBL], worldGrid[idxTR]))
                    {
                        triangles.Add(gridToVertIndex[idxTR]);
                        triangles.Add(gridToVertIndex[idxBR]);
                        triangles.Add(gridToVertIndex[idxBL]);
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

        // Material 설정: Inspector에서 설정된 것 사용, 없으면 fallback
        if (_meshSurfaceMaterial != null)
        {
            _meshRenderer.material = _meshSurfaceMaterial;
            Debug.Log($"[PointCloudRenderer] Using assigned material: {_meshSurfaceMaterial.name}");
        }
        else
        {
            // Fallback: shader 찾아서 생성
            var surfShader = Shader.Find("Custom/MeshSurfaceShader");
            if (surfShader == null) surfShader = Shader.Find("Particles/Standard Unlit");
            if (surfShader == null) surfShader = Shader.Find("Unlit/Color");

            if (surfShader != null)
            {
                var mat = new Material(surfShader);
                if (surfShader.name.Contains("Particles"))
                {
                    mat.SetFloat("_ColorMode", 1);
                }
                _meshRenderer.material = mat;
                Debug.Log($"[PointCloudRenderer] Using fallback shader: {surfShader.name}");
            }
            else
            {
                Debug.LogError("[PointCloudRenderer] No material and no shader found!");
            }
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

        // 렌더링 상태 디버그
        Debug.Log($"[PointCloudRenderer] === Rendering Debug ===");
        Debug.Log($"[PointCloudRenderer] GameObject active: {gameObject.activeSelf}, activeInHierarchy: {gameObject.activeInHierarchy}");
        Debug.Log($"[PointCloudRenderer] Layer: {gameObject.layer} ({LayerMask.LayerToName(gameObject.layer)})");
        Debug.Log($"[PointCloudRenderer] MeshRenderer enabled: {_meshRenderer.enabled}");
        Debug.Log($"[PointCloudRenderer] MeshRenderer isVisible: {_meshRenderer.isVisible}");
        Debug.Log($"[PointCloudRenderer] Material: {_meshRenderer.material?.name}, Shader: {_meshRenderer.material?.shader?.name}");
        Debug.Log($"[PointCloudRenderer] MeshFilter mesh: {_meshFilter.mesh?.name}, vertexCount: {_meshFilter.mesh?.vertexCount}");
        if (Camera.main != null)
        {
            Debug.Log($"[PointCloudRenderer] Camera cullingMask includes layer: {((Camera.main.cullingMask & (1 << gameObject.layer)) != 0)}");
        }

        // 디버그: 같은 위치에 테스트 큐브 생성
        if (_createDebugCube)
        {
            CreateDebugCube(centerPos);
        }
    }

    private void CreateDebugCube(Vector3 position)
    {
        if (_debugCube != null)
        {
            Destroy(_debugCube);
        }

        _debugCube = GameObject.CreatePrimitive(PrimitiveType.Cube);
        _debugCube.name = "DebugCube_MeshPosition";
        _debugCube.transform.position = position;
        _debugCube.transform.localScale = Vector3.one * 0.1f; // 10cm 큐브

        // 밝은 녹색 머티리얼 적용
        var renderer = _debugCube.GetComponent<Renderer>();
        var mat = new Material(Shader.Find("Unlit/Color"));
        mat.color = Color.green;
        renderer.material = mat;

        Debug.Log($"[PointCloudRenderer] Debug cube created at {position}");
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

    /// <summary>
    /// 두 점 사이의 거리가 연결 가능한 범위인지 확인 (기둥 방지)
    /// </summary>
    private bool IsEdgeValid(Vector3 a, Vector3 b)
    {
        return Vector3.Distance(a, b) < _meshConnectionDistance;
    }
}
