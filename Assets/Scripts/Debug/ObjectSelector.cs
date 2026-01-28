using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// Selects an object at the CENTER of the screen (Head Gaze).
/// Eliminates calibration errors from controller raycasting.
/// </summary>
public class ObjectSelector : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private IEExecutor _ieExecutor;

    [Header("Settings")]
    [SerializeField] private bool _showReticle = true;

    private GameObject _reticleCanvas;

    private void Start()
    {
        if (_ieExecutor == null)
            _ieExecutor = FindObjectOfType<IEExecutor>();

        if (_showReticle)
        {
            CreateReticle();
        }
    }

    private void Update()
    {
        // Handle Input (A Button) - Changed from Trigger
        if (OVRInput.GetDown(OVRInput.Button.One, OVRInput.Controller.RTouch))
        {
            SelectCenterObject();
        }
    }

    private void SelectCenterObject()
    {
        if (_ieExecutor == null) return;

        // 화면 정중앙 좌표 계산
        // Screen.width/height는 전체 해상도이므로, 
        // 렌더링 파이프라인에 따라 다를 수 있지만 일반적으로 Center는 (0.5, 0.5)입니다.
        
        float centerX = Screen.width * 0.5f;
        float centerY = Screen.height * 0.5f;

        Debug.Log($"[ObjectSelector] Selecting Center: {centerX}, {centerY}");
            
        // 중앙 좌표로 타겟 선택 요청
        _ieExecutor.SelectTargetFromScreenPos(new Vector2(centerX, centerY));
    }

    private void CreateReticle()
    {
        // 화면 중앙에 항상 떠 있는 UI 생성 (Screen Space Overlay는 VR에서 안 될 수 있으므로 Camera 앞 고정)
        _reticleCanvas = new GameObject("ReticleCanvas");
        Canvas c = _reticleCanvas.AddComponent<Canvas>();
        c.renderMode = RenderMode.WorldSpace;
        _reticleCanvas.AddComponent<CanvasScaler>();

        // 카메라 찾기
        Camera cam = Camera.main;
        if (cam == null) cam = FindObjectOfType<Camera>();
        if (cam == null) return;

        // 캔버스를 카메라 자식으로 넣어서 따라다니게 함
        _reticleCanvas.transform.SetParent(cam.transform);
        _reticleCanvas.transform.localPosition = Vector3.forward * 1.0f; // 1m 앞
        _reticleCanvas.transform.localRotation = Quaternion.identity;
        _reticleCanvas.transform.localScale = Vector3.one * 0.001f; // 아주 작게

        // 조준점 이미지 생성
        GameObject imageObj = new GameObject("ReticleImage");
        imageObj.transform.SetParent(_reticleCanvas.transform, false);
        
        Image img = imageObj.AddComponent<Image>();
        img.color = new Color(1, 0, 0, 0.8f); // 빨간색 반투명

        // 둥근 점 만들기 (스프라이트 없으면 네모로 나옴, 상관없음)
        RectTransform rt = imageObj.GetComponent<RectTransform>();
        rt.sizeDelta = new Vector2(20, 20); // 크기 조절
    }
}