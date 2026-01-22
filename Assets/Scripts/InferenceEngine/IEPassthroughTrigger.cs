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

        // Run a new inference when the current inference finishes
        if (!_ieExecutor.IsRunning())
        {
            _outputImage.texture = _webCamTextureManager.WebCamTexture;
            _outputImage.SetNativeSize();

            _ieExecutor.RunInference(_webCamTextureManager.WebCamTexture);
        }
    }
}
