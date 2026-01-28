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

    [Header("RGBD Debug Viewer")]
    [SerializeField] private RGBDDebugViewer _rgbdDebugViewer; 

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

            // B. 트리거 입력 감지 (물체 선택 - Preview 모드 시작)
            // OVRInput.Button.SecondaryIndexTrigger -> 오른쪽 검지 트리거
            if (OVRInput.GetDown(OVRInput.Button.SecondaryIndexTrigger) ||
                OVRInput.GetDown(OVRInput.Button.PrimaryIndexTrigger, OVRInput.Controller.RTouch))
            {
                Debug.Log("Right Index Trigger Pressed -> Select Target (Preview Mode)");

                // 1. 컨트롤러가 가리키는 먼 지점(World Pos) 계산
                Vector3 targetWorldPoint = rightHand.position + rightHand.forward * _laserLength;

                // 2. 그 지점을 2D 화면 좌표(Screen Pos)로 변환
                Vector3 screenPoint = Camera.main.WorldToScreenPoint(targetWorldPoint);

                // 3. Executor에게 "이 화면 좌표에 있는 물체 찾아줘" 요청
                // [Step 1] 이제 마스크만 보여주고 PointCloud는 생성하지 않음 (Preview 모드)
                _ieExecutor.SelectTargetFromScreenPos(screenPoint);
            }

            // C. A 버튼 입력 감지 (PointCloud 캡처)
            // [Step 1] Preview에서 마스크를 조절한 후 A 버튼으로 캡처
            if (OVRInput.GetDown(OVRInput.Button.One))
            {
                if (_ieExecutor.IsReadyToCapture())
                {
                    Debug.Log("A Button Pressed -> Capture PointCloud");
                    _ieExecutor.CapturePointCloud();

                    // RGBD 디버그 캡처 (선택된 물체의 RGB/Depth 표시)
                    if (_rgbdDebugViewer != null)
                    {
                        StartCoroutine(CaptureRGBDDelayed());
                    }
                }
                else
                {
                    Debug.Log("A Button Pressed -> No target to capture (select with trigger first)");
                }
            }

            // D. B 버튼 입력 감지 (추적 해제)
            if (OVRInput.GetDown(OVRInput.Button.Two))
            {
                Debug.Log("B Button Pressed -> Reset Tracking");
                _ieExecutor.ResetTracking();

                // 디버그 이미지도 초기화
                if (_rgbdDebugViewer != null)
                {
                    _rgbdDebugViewer.ClearDebugImages();
                }
            }
        }
    }

    /// <summary>
    /// 마스크가 적용된 후 RGBD 캡처 (약간의 딜레이)
    /// </summary>
    private IEnumerator CaptureRGBDDelayed()
    {
        // 다음 프레임까지 대기 (마스크가 그려진 후)
        yield return null;
        yield return null;

        if (_rgbdDebugViewer != null)
        {
            _rgbdDebugViewer.CaptureRGBDForTrackedObject();
        }
    }
}