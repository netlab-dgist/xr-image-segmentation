using System.Collections;
using UnityEngine;
using UnityEngine.UI;
using PassthroughCameraSamples;

public class IEPassthroughTrigger : MonoBehaviour
{
    [SerializeField] private WebCamTextureManager _webCamTextureManager;
    [SerializeField] private RawImage _outputImage;
    [SerializeField] private IEExecutor _ieExecutor;

    [Header("Laser Settings")]
    [SerializeField] private Transform _rightController;
    [SerializeField] private LineRenderer _laserLine;
    [SerializeField] private Material _laserMaterial;
    [SerializeField] private float _laserLength = 10f;
    [SerializeField] private Color _laserColor = Color.cyan;

    private bool _isTriggerHeld = false;

    private IEnumerator Start()
    {
        // 컨트롤러 자동 찾기
        if (_rightController == null)
        {
            var rig = FindObjectOfType<OVRCameraRig>();
            if (rig != null) _rightController = rig.rightHandAnchor;
        }

        SetupLaserLine();

        while (!_ieExecutor.IsModelLoaded) yield return null;
    }

    private void SetupLaserLine()
    {
        if (_laserLine == null && _rightController != null)
        {
            GameObject laserObj = new GameObject("LaserPointer");
            laserObj.transform.SetParent(_rightController);
            laserObj.transform.localPosition = Vector3.zero;
            laserObj.transform.localRotation = Quaternion.identity;

            _laserLine = laserObj.AddComponent<LineRenderer>();
            _laserLine.startWidth = 0.01f;
            _laserLine.endWidth = 0.005f;
            _laserLine.positionCount = 2;
            _laserLine.useWorldSpace = true;

            var shader = Shader.Find("Unlit/Color"); // 혹은 Universal Render Pipeline/Unlit
            if (shader != null) _laserLine.material = new Material(shader);
            _laserLine.startColor = _laserColor;
            _laserLine.endColor = _laserColor;
            _laserLine.enabled = false;
        }
    }

    private void Update()
    {
        // 1. 컨트롤러 입력 처리 (웹캠 상태와 무관하게 동작해야 함)
        HandleControllerInput();

        // 2. 추론 루프
        var hasWebCamTextureData = _webCamTextureManager.WebCamTexture != null;
        if (!hasWebCamTextureData) return;

        if (!_ieExecutor.IsRunning())
        {
            _outputImage.texture = _webCamTextureManager.WebCamTexture;
            _outputImage.SetNativeSize();
            _ieExecutor.RunInference(_webCamTextureManager.WebCamTexture);
        }
    }

    private void HandleControllerInput()
    {
        if (_rightController == null) return;

        // B 버튼: 리셋
        if (OVRInput.GetDown(OVRInput.Button.Two))
        {
            _ieExecutor.ResetTracking();
            Debug.Log("Reset Tracking");
        }

        // 트리거 누르는 중
        bool triggerHeld = OVRInput.Get(OVRInput.Button.SecondaryIndexTrigger);
        bool triggerDown = OVRInput.GetDown(OVRInput.Button.SecondaryIndexTrigger);

        if (triggerHeld)
        {
            if (!_isTriggerHeld) _isTriggerHeld = true;

            ShowLaser(true);
            Vector2 screenPos = GetLaserScreenPosition();

            // [핵심] 레이저가 가리키는 곳의 물체에 대해 Point Cloud 추출
            _ieExecutor.ExtractPointCloudAtScreenPos(screenPos);
            
            // (옵션) 처음 눌렀을 때 타겟을 고정(Select)하고 싶다면 아래 주석 해제
            if (triggerDown) 
            {
               _ieExecutor.SelectTargetFromScreenPos(screenPos);
            }
        }
        else if (_isTriggerHeld)
        {
            _isTriggerHeld = false;
            ShowLaser(false);
            // 트리거 뗐을 때 포인트 클라우드를 남기고 싶으면 아래 줄 삭제
            // _ieExecutor.ClearPointCloud(); 
        }
    }

    private void ShowLaser(bool show)
    {
        if (_laserLine != null)
        {
            _laserLine.enabled = show;
            if (show)
            {
                _laserLine.SetPosition(0, _rightController.position);
                _laserLine.SetPosition(1, _rightController.position + _rightController.forward * _laserLength);
            }
        }
    }

    private Vector2 GetLaserScreenPosition()
    {
        // 가상의 2m 앞 평면에 레이캐스트하여 화면 좌표 구하기
        Vector3 targetPoint = _rightController.position + _rightController.forward * 2.0f;
        Vector3 screenPoint = Camera.main.WorldToScreenPoint(targetPoint);
        return new Vector2(screenPoint.x, screenPoint.y);
    }
}