using System.Collections;
using UnityEngine;
using UnityEngine.UI;
using PassthroughCameraSamples;

public class IEPassthroughTrigger : MonoBehaviour
{
    [SerializeField] private WebCamTextureManager _webCamTextureManager;
    [SerializeField] private RawImage _outputImage;
    [SerializeField] private IEExecutor _ieExecutor;

    [Header("Controller Setup")]
    // [중요] 인스펙터에서 OVRCameraRig를 꼭 연결해야 합니다!
    [SerializeField] private OVRCameraRig _cameraRig; 
    
    // 레이저 시각화 (선택 사항)
    [SerializeField] private LineRenderer _laserLineRenderer; 
    [SerializeField] private float _laserLength = 50.0f; 

    private IEnumerator Start()
    {
        // LineRenderer가 없으면 자동으로 추가해서 빨간 선으로 설정
        if (_laserLineRenderer == null)
        {
            _laserLineRenderer = gameObject.AddComponent<LineRenderer>();
            _laserLineRenderer.startWidth = 0.005f;
            _laserLineRenderer.endWidth = 0.002f;
            _laserLineRenderer.material = new Material(Shader.Find("Sprites/Default"));
            _laserLineRenderer.startColor = new Color(1, 0, 0, 0.5f);
            _laserLineRenderer.endColor = new Color(1, 0, 0, 0);
            _laserLineRenderer.positionCount = 2;
        }

        // Wait until Sentis model is loaded
        while (!_ieExecutor.IsModelLoaded) yield return null;
        Debug.Log("IEPassthroughTrigger: Sentis model is loaded");
    }

    private void Update()
    {
        // 1. 웹캠 텍스처 관리 및 추론 실행 (기존 로직)
        var hasWebCamTextureData = _webCamTextureManager.WebCamTexture != null;
        if (!hasWebCamTextureData) return;

        if (!_ieExecutor.IsRunning())
        {
            _outputImage.texture = _webCamTextureManager.WebCamTexture;
            _outputImage.SetNativeSize();
            _ieExecutor.RunInference(_webCamTextureManager.WebCamTexture);
        }

        // ---------------------------------------------------------------------
        // [수정됨] 레이저 포인팅 및 선택 로직
        // ---------------------------------------------------------------------
        
        if (_cameraRig != null)
        {
            // 오른쪽 컨트롤러(RightHandAnchor) 기준
            Transform rightHand = _cameraRig.rightHandAnchor;

            // A. 레이저 시각화 (컨트롤러에서 뻗어나가는 선 그리기)
            _laserLineRenderer.SetPosition(0, rightHand.position);
            _laserLineRenderer.SetPosition(1, rightHand.position + rightHand.forward * 3.0f); // 시각적으로는 3m만 표시

            // B. 트리거 입력 감지 (선택)
            // OVRInput.Button.SecondaryIndexTrigger -> 오른쪽 검지 트리거
            if (OVRInput.GetDown(OVRInput.Button.SecondaryIndexTrigger) || 
                OVRInput.GetDown(OVRInput.Button.PrimaryIndexTrigger, OVRInput.Controller.RTouch))
            {
                Debug.Log("Right Index Trigger Pressed -> Raycasting...");

                // 1. 컨트롤러가 가리키는 먼 지점(World Pos) 계산
                Vector3 targetWorldPoint = rightHand.position + rightHand.forward * _laserLength;

                // 2. 그 지점을 2D 화면 좌표(Screen Pos)로 변환
                // (카메라가 보고 있는 화면상의 어디에 해당하는지 찾음)
                Vector3 screenPoint = Camera.main.WorldToScreenPoint(targetWorldPoint);

                // 3. Executor에게 "이 화면 좌표에 있는 물체 찾아줘" 요청
                _ieExecutor.SelectTargetFromScreenPos(screenPoint);
            }

            // C. B 버튼 입력 감지 (추적 해제)
            if (OVRInput.GetDown(OVRInput.Button.Two))
            {
                Debug.Log("B Button Pressed -> Reset Tracking");
                _ieExecutor.ResetTracking();
            }
        }
    }
}