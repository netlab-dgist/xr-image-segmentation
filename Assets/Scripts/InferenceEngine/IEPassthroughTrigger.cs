using System.Collections;
using UnityEngine;
using UnityEngine.UI;
using PassthroughCameraSamples;

public class IEPassthroughTrigger : MonoBehaviour
{
    [SerializeField] private WebCamTextureManager _webCamTextureManager;

    [SerializeField] private RawImage _outputImage;

    [SerializeField] private IEExecutor _ieExecutor;

    private IEnumerator Start()
    {
        // Wait until Sentis model is loaded
        while (!_ieExecutor.IsModelLoaded) yield return null;
        Debug.Log("IEPassthroughTrigger: Sentis model is loaded");
    }

    private void Update()
    {
        // Get the WebCamTexture CPU image
        var hasWebCamTextureData = _webCamTextureManager.WebCamTexture != null;

        if (!hasWebCamTextureData) return;

        // 컨트롤러 트리거 입력 처리
        HandleTriggerInput();

        // Run a new inference when the current inference finishes
        if (!_ieExecutor.IsRunning())
        {
            _outputImage.texture = _webCamTextureManager.WebCamTexture;
            _outputImage.SetNativeSize();

            _ieExecutor.RunInference(_webCamTextureManager.WebCamTexture);
        }
    }

    private void HandleTriggerInput()
    {
        // 오른손 또는 왼손 트리거 입력 감지
        bool triggerPressed = OVRInput.GetDown(OVRInput.Button.PrimaryIndexTrigger) ||
                              OVRInput.GetDown(OVRInput.Button.SecondaryIndexTrigger);

        if (triggerPressed)
        {
            // 이미 추적 중이면 추적 해제
            if (_ieExecutor.IsTracking)
            {
                _ieExecutor.ResetTracking();
                Debug.Log("[IEPassthroughTrigger] Tracking reset by trigger");
                return;
            }

            // 화면 중앙을 선택 지점으로 사용 (컨트롤러가 가리키는 곳)
            Vector2 screenCenter = new Vector2(Screen.width / 2f, Screen.height / 2f);
            _ieExecutor.SelectTargetFromScreenPos(screenCenter);
            Debug.Log("[IEPassthroughTrigger] Trigger pressed, selecting target at screen center");
        }
    }
}
