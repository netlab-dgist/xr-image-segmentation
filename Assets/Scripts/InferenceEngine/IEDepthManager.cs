using System;
using System.Collections;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.UI;
using Meta.XR.EnvironmentDepth;
using PassthroughCameraSamples;

// RGBD 데이터 구조체
public struct RGBDData
{
    public Color[] Colors;
    public float[] Depths;
    public int Width;
    public int Height;
    public Vector3 WorldPosition;
}

public class IEDepthManager : MonoBehaviour
{
    [Header("Dependencies")]
    [SerializeField] private IEExecutor _ieExecutor;
    [SerializeField] private EnvironmentDepthManager _depthManager;
    [SerializeField] private WebCamTextureManager _webCamManager;

    [Header("Debug UI Settings")]
    [Tooltip("추적 중인 물체의 RGB 모습을 보여줄 RawImage")]
    [SerializeField] private RawImage _debugRgbImage;
    [Tooltip("추적 중인 물체의 Depth 모습을 보여줄 RawImage")]
    [SerializeField] private RawImage _debugDepthImage;
    
    [Header("Status UI Settings")]
    [SerializeField] private Transform _displayLocation; 
    [SerializeField] private Font _font;
    [SerializeField] private int _fontSize = 30;
    [SerializeField] private Color _textColor = Color.cyan;

    private static readonly int DepthTextureID = Shader.PropertyToID("_EnvironmentDepthTexture");
    private bool _isProcessing = false;
    private Text _statusText;
    
    // 디버그용 텍스처 캐싱 (메모리 할당 최소화)
    private Texture2D _cachedRgbTex;
    private Texture2D _cachedDepthTex;

    public event Action<RGBDData> OnRGBDDataReady;

    private void Start()
    {
        CreateStatusUI();
    }

    private void CreateStatusUI()
    {
        if (_displayLocation == null) return;

        GameObject textObj = new GameObject("RGBD_Status_Text");
        textObj.transform.SetParent(_displayLocation, false);

        RectTransform rt = textObj.AddComponent<RectTransform>();
        rt.anchorMin = new Vector2(0, 1); 
        rt.anchorMax = new Vector2(0, 1);
        rt.pivot = new Vector2(0, 1);
        rt.anchoredPosition = new Vector2(20, -20);
        rt.sizeDelta = new Vector2(800, 300);

        _statusText = textObj.AddComponent<Text>();
        _statusText.font = _font != null ? _font : Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        _statusText.fontSize = _fontSize;
        _statusText.color = _textColor;
        _statusText.text = "Waiting for target selection...";
    }

    private void Update()
    {
        // 1. 추적 중인지 확인
        if (!_ieExecutor.IsTracking || !_ieExecutor.LockedTargetBox.HasValue)
        {
            UpdateStatusText("Status: Idle (Select object with R-Trigger)");
            if (_debugRgbImage) _debugRgbImage.gameObject.SetActive(false);
            if (_debugDepthImage) _debugDepthImage.gameObject.SetActive(false);
            return;
        }

        // 2. Depth API 준비 확인
        if (!_depthManager.IsDepthAvailable)
        {
            UpdateStatusText("Status: Waiting for Depth API...");
            return;
        }

        // 3. 중복 처리 방지
        if (_isProcessing) return; 

        // 4. 데이터 획득 시작
        var box = _ieExecutor.LockedTargetBox.Value;
        var webCamTex = _webCamManager.WebCamTexture;
        if (webCamTex == null) return;

        StartCoroutine(FetchRGBDCoroutine(box, webCamTex));
    }
    
    private void UpdateStatusText(string message)
    {
        if (_statusText != null) _statusText.text = message;
    }

    private IEnumerator FetchRGBDCoroutine(BoundingBox box, WebCamTexture webCamTex)
    {
        _isProcessing = true;
        
        // ----------------------------------------------------------------
        // A. RGB 텍스처에서 BoundingBox 영역 잘라내기 (ROI)
        // ----------------------------------------------------------------
        int imgW = webCamTex.width;
        int imgH = webCamTex.height;
        float halfW = imgW / 2f;
        float halfH = imgH / 2f;
        
        // YOLO 좌표(Center, W, H) -> Texture 좌표(Left, Bottom, W, H) 변환
        float boxMinX = (box.CenterX - box.Width / 2f) + halfW;
        float boxMinY = (halfH - box.CenterY) - (box.Height / 2f); // Y축 반전 주의
        
        int x = Mathf.Clamp(Mathf.FloorToInt(boxMinX), 0, imgW - 1);
        int y = Mathf.Clamp(Mathf.FloorToInt(boxMinY), 0, imgH - 1);
        int w = Mathf.Clamp(Mathf.FloorToInt(box.Width), 1, imgW - x);
        int h = Mathf.Clamp(Mathf.FloorToInt(box.Height), 1, imgH - y);

        // 유효하지 않은 영역이면 중단
        if (w <= 0 || h <= 0)
        {
            _isProcessing = false;
            yield break;
        }

        // RGB 픽셀 읽기 (CPU 부하 발생 지점)
        Color[] rgbPixels = webCamTex.GetPixels(x, y, w, h);

        // [디버그] 잘라낸 RGB 이미지를 UI에 표시
        if (_debugRgbImage != null)
        {
            _debugRgbImage.gameObject.SetActive(true);
            if (_cachedRgbTex == null || _cachedRgbTex.width != w || _cachedRgbTex.height != h)
            {
                Destroy(_cachedRgbTex);
                _cachedRgbTex = new Texture2D(w, h);
            }
            _cachedRgbTex.SetPixels(rgbPixels);
            _cachedRgbTex.Apply();
            _debugRgbImage.texture = _cachedRgbTex;
        }

        // ----------------------------------------------------------------
        // B. Depth 텍스처에서 해당 영역 가져오기 (GPU Readback)
        // ----------------------------------------------------------------
        Texture depthTexture = Shader.GetGlobalTexture(DepthTextureID);
        
        if (depthTexture != null)
        {
            // [중요] 현재는 단순 UV 매핑을 사용합니다. 
            // 테스트 후 정합성이 맞지 않으면 EnvironmentDepthUtils의 행렬을 사용하여 좌표를 보정해야 합니다.
            float u = (float)x / imgW;
            float v = (float)y / imgH;
            float uWidth = (float)w / imgW;
            float vHeight = (float)h / imgH;

            int depthTexW = depthTexture.width;
            int depthTexH = depthTexture.height;

            int dx = Mathf.Clamp(Mathf.FloorToInt(u * depthTexW), 0, depthTexW - 1);
            int dy = Mathf.Clamp(Mathf.FloorToInt(v * depthTexH), 0, depthTexH - 1);
            int dw = Mathf.Clamp(Mathf.FloorToInt(uWidth * depthTexW), 1, depthTexW - dx);
            int dh = Mathf.Clamp(Mathf.FloorToInt(vHeight * depthTexH), 1, depthTexH - dy);

            // 비동기 GPU 읽기 요청
            var request = AsyncGPUReadback.Request(depthTexture, 0, dx, dy, dw, dh, 0, 1, null);

            yield return new WaitUntil(() => request.done);

            if (!request.hasError)
            {
                var depthRawData = request.GetData<ushort>();
                float[] depthFloats = new float[depthRawData.Length];
                
                // [디버그] Depth 시각화를 위한 컬러 배열
                Color[] depthVisColors = new Color[depthRawData.Length];
                
                float centerDepthVal = 0f;
                int centerIndex = (dh / 2) * dw + (dw / 2);
                if (centerIndex < depthRawData.Length)
                    centerDepthVal = (float)depthRawData[centerIndex] / 65535f;

                // 데이터 변환 (ushort -> float) 및 시각화 색상 생성
                for (int i = 0; i < depthRawData.Length; i++)
                {
                    // Meta Depth는 0(근접) ~ 1(무한대) 정규화된 값이 아닐 수 있음 (보통 미터 단위일 수도 있음)
                    // 하지만 셰이더 프로퍼티 텍스처라면 0~1 사이 값일 확률이 큼.
                    float val = (float)depthRawData[i] / 65535f; 
                    depthFloats[i] = val;

                    // 시각화: 값이 작을수록(가까울수록) 밝게, 멀수록 어둡게 반전
                    // Depth값이 너무 작아서 안 보일 수 있으니 10배 증폭해서 확인
                    float visVal = val * 20.0f; 
                    depthVisColors[i] = new Color(visVal, visVal, visVal, 1f);
                }

                // [디버그] 잘라낸 Depth 이미지를 UI에 표시
                if (_debugDepthImage != null)
                {
                    _debugDepthImage.gameObject.SetActive(true);
                    if (_cachedDepthTex == null || _cachedDepthTex.width != dw || _cachedDepthTex.height != dh)
                    {
                        Destroy(_cachedDepthTex);
                        _cachedDepthTex = new Texture2D(dw, dh);
                    }
                    _cachedDepthTex.SetPixels(depthVisColors);
                    _cachedDepthTex.Apply();
                    _debugDepthImage.texture = _cachedDepthTex;
                }

                // 이벤트 발생 (Point Cloud 생성기로 전달)
                RGBDData data = new RGBDData
                {
                    Colors = rgbPixels,
                    Depths = depthFloats,
                    Width = w,
                    Height = h,
                    WorldPosition = box.WorldPos ?? Vector3.zero
                };
                OnRGBDDataReady?.Invoke(data);

                UpdateStatusText($"[Target: {box.ClassName}]\n" +
                                 $"RGB: {w}x{h} / Depth: {dw}x{dh}\n" +
                                 $"Center Depth: {centerDepthVal:F6}");
            }
        }
        _isProcessing = false;
    }
}