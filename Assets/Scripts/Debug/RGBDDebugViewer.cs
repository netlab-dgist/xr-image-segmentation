using UnityEngine;
using UnityEngine.UI;
using PassthroughCameraSamples;

/// <summary>
/// 트리거를 눌렀을 때 선택된 물체의 RGB와 Depth를 사이드 RawImage에 표시
/// - 중앙 화면(메인 passthrough + 마스크)은 그대로 유지
/// - 트리거 눌렀을 때만 캡처하여 lag 최소화
/// </summary>
public class RGBDDebugViewer : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private IEExecutor _ieExecutor;
    [SerializeField] private WebCamTextureManager _webCamTextureManager;
    [SerializeField] private DepthTextureProvider _depthProvider;

    [Header("Debug UI - Side Panels")]
    [Tooltip("RGB 디버그 이미지를 표시할 RawImage (사이드 패널)")]
    [SerializeField] private RawImage _rgbDebugImage;

    [Tooltip("Depth 디버그 이미지를 표시할 RawImage (사이드 패널)")]
    [SerializeField] private RawImage _depthDebugImage;

    [Header("Debug UI - Full Depth (Optional)")]
    [Tooltip("전체 Depth 이미지를 표시할 RawImage (선택사항)")]
    [SerializeField] private RawImage _fullDepthImage;

    [Header("Settings")]
    [SerializeField] private int _debugImageSize = 256;
    [SerializeField] private bool _showMaskedRegionOnly = true;

    // 내부 텍스처
    private Texture2D _rgbCaptureTexture;
    private Texture2D _maskedRgbTexture;
    private Texture2D _maskedDepthTexture;

    // 상태
    private bool _hasCapture = false;

    void Start()
    {
        // 초기 상태에서 디버그 이미지 숨기기 (선택사항)
        if (_rgbDebugImage != null)
        {
            _rgbDebugImage.color = new Color(1, 1, 1, 0.8f);
        }
        if (_depthDebugImage != null)
        {
            _depthDebugImage.color = new Color(1, 1, 1, 0.8f);
        }
    }

    void Update()
    {
        // 전체 Depth 시각화 업데이트 (선택적, 항상 표시)
        if (_fullDepthImage != null && _depthProvider != null && _depthProvider.IsDepthAvailable)
        {
            _depthProvider.UpdateVisualization();
            _fullDepthImage.texture = _depthProvider.DepthVisualizationTexture;
        }
    }

    /// <summary>
    /// 트리거를 눌렀을 때 호출 - RGB와 Depth 캡처
    /// IEPassthroughTrigger에서 SelectTargetFromScreenPos 이후 호출
    /// </summary>
    public void CaptureRGBDForTrackedObject()
    {
        if (_ieExecutor == null || !_ieExecutor.IsTracking)
        {
            Debug.Log("[RGBDDebugViewer] Not tracking any object");
            ClearDebugImages();
            return;
        }

        // 디버그: 각 텍스처의 해상도 출력
        LogTextureResolutions();

        // RGB 캡처
        CaptureRGB();

        // Depth 캡처
        CaptureDepth();

        _hasCapture = true;
        Debug.Log("[RGBDDebugViewer] RGBD captured for tracked object");
    }

    /// <summary>
    /// 각 소스의 해상도 및 좌표 정보 로깅
    /// </summary>
    private void LogTextureResolutions()
    {
        var webCamTex = _webCamTextureManager?.WebCamTexture;
        var maskTex = _ieExecutor?.Masker?.CurrentMaskTexture;

        Debug.Log("=== [RGBDDebugViewer] Texture Resolution Debug ===");

        // RGB 해상도
        if (webCamTex != null)
        {
            Debug.Log($"[RGB] WebCamTexture: {webCamTex.width}x{webCamTex.height}");
        }

        // Depth 해상도
        if (_depthProvider != null && _depthProvider.IsDepthAvailable)
        {
            Debug.Log($"[Depth] DepthTexture: {_depthProvider.DepthWidth}x{_depthProvider.DepthHeight}");
        }
        else
        {
            Debug.LogWarning("[Depth] Not available!");
        }

        // Mask 해상도
        if (maskTex != null)
        {
            Debug.Log($"[Mask] MaskTexture: {maskTex.width}x{maskTex.height}");
        }

        // BoundingBox 정보
        var box = _ieExecutor?.LockedTargetBox;
        if (box.HasValue)
        {
            var b = box.Value;
            Debug.Log($"[BBox] center=({b.CenterX:F1},{b.CenterY:F1}), size=({b.Width:F1},{b.Height:F1}) (in 640x640 space)");
        }

        // Camera Intrinsics (Quest에서만 동작)
        try
        {
            var intrinsics = PassthroughCameraSamples.PassthroughCameraUtils.GetCameraIntrinsics(
                PassthroughCameraSamples.PassthroughCameraEye.Left);
            Debug.Log($"[Intrinsics] fx={intrinsics.FocalLength.x:F1}, fy={intrinsics.FocalLength.y:F1}");
            Debug.Log($"[Intrinsics] cx={intrinsics.PrincipalPoint.x:F1}, cy={intrinsics.PrincipalPoint.y:F1}");
            Debug.Log($"[Intrinsics] Resolution={intrinsics.Resolution.x}x{intrinsics.Resolution.y}");
        }
        catch (System.Exception e)
        {
            Debug.Log($"[Intrinsics] Not available (Editor mode?): {e.Message}");
        }

        Debug.Log("=== End Debug ===");
    }

    /// <summary>
    /// RGB 이미지 캡처 및 표시
    /// </summary>
    private void CaptureRGB()
    {
        if (_rgbDebugImage == null || _webCamTextureManager == null)
            return;

        var webCamTex = _webCamTextureManager.WebCamTexture;
        if (webCamTex == null)
            return;

        Texture2D maskTex = _ieExecutor?.Masker?.CurrentMaskTexture;
        if (_showMaskedRegionOnly && maskTex != null)
        {
            // 마스크 영역만 RGB 추출
            if (_maskedRgbTexture != null)
                Destroy(_maskedRgbTexture);

            _maskedRgbTexture = CreateMaskedRGBFromMask(webCamTex, maskTex, _debugImageSize);
            _rgbDebugImage.texture = _maskedRgbTexture;
        }
        else
        {
            // 전체 RGB 이미지 표시
            _rgbDebugImage.texture = webCamTex;
        }
    }

    /// <summary>
    /// Depth 이미지 캡처 및 표시
    /// </summary>
    private void CaptureDepth()
    {
        if (_depthDebugImage == null || _depthProvider == null || !_depthProvider.IsDepthAvailable)
        {
            Debug.LogWarning("[RGBDDebugViewer] Depth not available");
            return;
        }

        Texture2D maskTex = _ieExecutor?.Masker?.CurrentMaskTexture;
        if (_showMaskedRegionOnly && maskTex != null)
        {
            // 마스크 영역만 Depth 추출
            if (_maskedDepthTexture != null)
                Destroy(_maskedDepthTexture);

            _maskedDepthTexture = _depthProvider.CreateMaskedDepthFromMask(
                maskTex, _debugImageSize, _ieExecutor.InputSize);

            if (_maskedDepthTexture != null)
                _depthDebugImage.texture = _maskedDepthTexture;
        }
        else
        {
            // 전체 Depth 시각화
            _depthProvider.UpdateVisualization();
            _depthDebugImage.texture = _depthProvider.DepthVisualizationTexture;
        }
    }

    /// <summary>
    /// 마스크 텍스처를 기반으로 RGB 추출
    /// 마스크가 있는 영역(alpha > 0)의 bounding box를 찾아 해당 RGB 영역만 크롭
    /// </summary>
    private Texture2D CreateMaskedRGBFromMask(WebCamTexture srcTex, Texture2D maskTex, int outputSize)
    {
        if (srcTex == null || maskTex == null)
            return null;

        int maskW = maskTex.width;   // 160
        int maskH = maskTex.height;  // 160
        int srcW = srcTex.width;     // 1280
        int srcH = srcTex.height;    // 1280

        Color32[] maskColors = maskTex.GetPixels32();

        // 1. 마스크에서 유효 영역(alpha > 0)의 bounding box 찾기
        int minX = maskW, minY = maskH, maxX = 0, maxY = 0;
        for (int y = 0; y < maskH; y++)
        {
            for (int x = 0; x < maskW; x++)
            {
                if (maskColors[y * maskW + x].a > 0)
                {
                    minX = Mathf.Min(minX, x);
                    minY = Mathf.Min(minY, y);
                    maxX = Mathf.Max(maxX, x);
                    maxY = Mathf.Max(maxY, y);
                }
            }
        }

        if (maxX < minX || maxY < minY)
        {
            Debug.LogWarning("[RGB Mask] No valid mask pixels found!");
            return null;
        }

        int maskBBoxW = maxX - minX + 1;
        int maskBBoxH = maxY - minY + 1;

        // 마스크 좌표 → RGB 좌표 스케일 (mask 160 → YOLO 640 → RGB 1280)
        // mask 1 pixel = 4 YOLO pixels, YOLO 640 → RGB 1280 = x2
        // 따라서 mask 1 pixel = 8 RGB pixels
        float maskToRgbScale = (float)srcW / maskW;

        int rgbMinX = Mathf.RoundToInt(minX * maskToRgbScale);
        int rgbMinY = Mathf.RoundToInt(minY * maskToRgbScale);
        int rgbBBoxW = Mathf.RoundToInt(maskBBoxW * maskToRgbScale);
        int rgbBBoxH = Mathf.RoundToInt(maskBBoxH * maskToRgbScale);

        Debug.Log($"[RGB Mask] Mask BBox: ({minX},{minY}) size({maskBBoxW},{maskBBoxH}) in {maskW}x{maskH}");
        Debug.Log($"[RGB Mask] → RGB BBox: ({rgbMinX},{rgbMinY}) size({rgbBBoxW},{rgbBBoxH}) in {srcW}x{srcH}");

        Texture2D result = new Texture2D(outputSize, outputSize, TextureFormat.RGBA32, false);
        Color[] resultColors = new Color[outputSize * outputSize];
        Color[] srcColors = srcTex.GetPixels();

        for (int y = 0; y < outputSize; y++)
        {
            for (int x = 0; x < outputSize; x++)
            {
                // output 좌표 → mask bbox 내부 좌표
                int maskX = minX + Mathf.RoundToInt((float)x / outputSize * maskBBoxW);
                int maskY = minY + Mathf.RoundToInt((float)y / outputSize * maskBBoxH);

                maskX = Mathf.Clamp(maskX, 0, maskW - 1);
                maskY = Mathf.Clamp(maskY, 0, maskH - 1);

                // 마스크 체크
                if (maskColors[maskY * maskW + maskX].a > 0)
                {
                    // mask 좌표 → RGB 좌표
                    int srcX = Mathf.RoundToInt(maskX * maskToRgbScale);
                    int srcY = Mathf.RoundToInt(maskY * maskToRgbScale);

                    srcX = Mathf.Clamp(srcX, 0, srcW - 1);
                    srcY = Mathf.Clamp(srcY, 0, srcH - 1);

                    resultColors[y * outputSize + x] = srcColors[srcY * srcW + srcX];
                }
                else
                {
                    resultColors[y * outputSize + x] = Color.clear;
                }
            }
        }

        result.SetPixels(resultColors);
        result.Apply();
        return result;
    }

    /// <summary>
    /// 디버그 이미지 초기화
    /// </summary>
    public void ClearDebugImages()
    {
        if (_rgbDebugImage != null)
            _rgbDebugImage.texture = null;
        if (_depthDebugImage != null)
            _depthDebugImage.texture = null;

        _hasCapture = false;
    }

    /// <summary>
    /// 현재 캡처된 마스크 영역의 BoundingBox 정보 반환
    /// </summary>
    public BoundingBox? GetCurrentTrackedBox()
    {
        if (_ieExecutor != null && _ieExecutor.IsTracking)
            return _ieExecutor.LockedTargetBox;
        return null;
    }

    void OnDestroy()
    {
        if (_rgbCaptureTexture != null)
            Destroy(_rgbCaptureTexture);
        if (_maskedRgbTexture != null)
            Destroy(_maskedRgbTexture);
        if (_maskedDepthTexture != null)
            Destroy(_maskedDepthTexture);
    }
}
