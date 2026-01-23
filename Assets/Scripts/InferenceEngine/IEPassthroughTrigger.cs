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
    [SerializeField] private OVRCameraRig _cameraRig;

    // 레이저 시각화
    [SerializeField] private LineRenderer _laserLineRenderer;
    [SerializeField] private float _laserLength = 50.0f;

    // [신규] Point Cloud 생성용 스냅샷 버튼 (예: Hand Trigger)
    [Tooltip("Point Cloud를 생성할 트리거 버튼 (기본: Hand Trigger / Grip)")]
    [SerializeField] private OVRInput.Button _snapshotButton = OVRInput.Button.SecondaryHandTrigger;

    private IEnumerator Start()
    {
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

        while (!_ieExecutor.IsModelLoaded) yield return null;
        Debug.Log("IEPassthroughTrigger: Sentis model is loaded");
    }

    private void Update()
    {
        var hasWebCamTextureData = _webCamTextureManager.WebCamTexture != null;
        if (!hasWebCamTextureData) return;

        if (!_ieExecutor.IsRunning())
        {
            _outputImage.texture = _webCamTextureManager.WebCamTexture;
            _outputImage.SetNativeSize();
            _ieExecutor.RunInference(_webCamTextureManager.WebCamTexture);
        }

        if (_cameraRig != null)
        {
            Transform rightHand = _cameraRig.rightHandAnchor;
            _laserLineRenderer.SetPosition(0, rightHand.position);
            _laserLineRenderer.SetPosition(1, rightHand.position + rightHand.forward * 3.0f);

            // 1. Index Trigger: 물체 선택 (2D Tracking 시작)
            if (OVRInput.GetDown(OVRInput.Button.SecondaryIndexTrigger) ||
                OVRInput.GetDown(OVRInput.Button.PrimaryIndexTrigger, OVRInput.Controller.RTouch))
            {
                Debug.Log("Index Trigger -> Selecting Target...");
                Vector3 targetWorldPoint = rightHand.position + rightHand.forward * _laserLength;
                Vector3 screenPoint = Camera.main.WorldToScreenPoint(targetWorldPoint);
                _ieExecutor.SelectTargetFromScreenPos(screenPoint);
            }

            // 2. Hand Trigger (Grip): 3D Mesh Snapshot 생성
            if (OVRInput.GetDown(_snapshotButton))
            {
                Debug.Log("Hand Trigger -> Capturing Snapshot & Generating Mesh...");
                _ieExecutor.CaptureSnapshot();
            }

            // 3. B Button: 리셋
            if (OVRInput.GetDown(OVRInput.Button.Two))
            {
                Debug.Log("B Button -> Reset Tracking");
                _ieExecutor.ResetTracking();
            }
        }
    }
}
