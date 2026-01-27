using UnityEngine;
using UnityEngine.UI;
using Meta.XR.EnvironmentDepth;

/// <summary>
/// Environment Depth API를 통해 depth 텍스처를 제공하는 클래스
/// - GPU에서 depth 텍스처를 읽어 CPU 텍스처로 변환
/// - 디버그용 시각화 텍스처 생성 지원
/// </summary>
public class DepthTextureProvider : MonoBehaviour
{
    [Header("Depth Manager")]
    [SerializeField] private EnvironmentDepthManager _depthManager;

    [Header("Visualization Settings")]
    [SerializeField] private float _visNear = 0.2f;
    [SerializeField] private float _visFar = 3.0f;
    [SerializeField] private float _cutoffDistance = 5.0f;

    // CPU-side depth buffer (meters encoded as RHalf)
    private Texture2D _cpuDepthTex;

    // CPU-side visualization texture
    private Texture2D _depthVisTex;

    // 외부에서 접근 가능한 텍스처
    public Texture2D DepthTexture => _cpuDepthTex;
    public Texture2D DepthVisualizationTexture => _depthVisTex;

    // Depth 텍스처 크기
    public int DepthWidth => _cpuDepthTex != null ? _cpuDepthTex.width : 0;
    public int DepthHeight => _cpuDepthTex != null ? _cpuDepthTex.height : 0;

    // Depth 사용 가능 여부
    public bool IsDepthAvailable { get; private set; } = false;

    private RenderTexture _lastDepthRT;

    void Update()
    {
        UpdateDepthTexture();
    }

    /// <summary>
    /// 매 프레임 depth 텍스처 업데이트
    /// </summary>
    private void UpdateDepthTexture()
    {
        if (_depthManager == null)
        {
            IsDepthAvailable = false;
            return;
        }

        // SDK에서 depth 텍스처 가져오기
        var depthRT = Shader.GetGlobalTexture("_PreprocessedEnvironmentDepthTexture") as RenderTexture;
        if (depthRT == null)
        {
            IsDepthAvailable = false;
            return;
        }

        _lastDepthRT = depthRT;

        // CPU depth 텍스처 할당 (크기 변경 시에만)
        if (_cpuDepthTex == null ||
            _cpuDepthTex.width != depthRT.width ||
            _cpuDepthTex.height != depthRT.height)
        {
            if (_cpuDepthTex != null)
                Destroy(_cpuDepthTex);
            if (_depthVisTex != null)
                Destroy(_depthVisTex);

            _cpuDepthTex = new Texture2D(
                depthRT.width,
                depthRT.height,
                TextureFormat.RHalf,
                false,
                true
            );

            _depthVisTex = new Texture2D(
                depthRT.width,
                depthRT.height,
                TextureFormat.RGBA32,
                false
            );

            Debug.Log($"[DepthTextureProvider] Depth texture initialized: {depthRT.width}x{depthRT.height}");
        }

        // GPU → CPU 복사
        RenderTexture prev = RenderTexture.active;
        RenderTexture.active = depthRT;
        _cpuDepthTex.ReadPixels(new Rect(0, 0, depthRT.width, depthRT.height), 0, 0);
        _cpuDepthTex.Apply();
        RenderTexture.active = prev;

        IsDepthAvailable = true;
    }

    /// <summary>
    /// 전체 depth 이미지를 grayscale로 시각화
    /// </summary>
    public void UpdateVisualization()
    {
        if (!IsDepthAvailable || _cpuDepthTex == null || _depthVisTex == null)
            return;

        int w = _cpuDepthTex.width;
        int h = _cpuDepthTex.height;

        var src = _cpuDepthTex.GetRawTextureData<ushort>();
        var colors = _depthVisTex.GetPixels();

        float invRange = 1.0f / Mathf.Max(0.0001f, (_visFar - _visNear));

        for (int i = 0; i < colors.Length; i++)
        {
            float d = Mathf.HalfToFloat(src[i]);

            if (d <= 0.0001f || d > _cutoffDistance)
            {
                colors[i] = Color.black;
                continue;
            }

            // normalize depth → grayscale (가까우면 밝게, 멀면 어둡게)
            float v = 1.0f - Mathf.Clamp01((d - _visNear) * invRange);
            colors[i] = new Color(v, v, v);
        }

        _depthVisTex.SetPixels(colors);
        _depthVisTex.Apply();
    }

    /// <summary>
    /// 특정 픽셀 좌표의 depth 값(미터) 반환
    /// </summary>
    public float GetDepthAtPixel(int x, int y)
    {
        if (!IsDepthAvailable || _cpuDepthTex == null)
            return -1f;

        if (x < 0 || x >= _cpuDepthTex.width || y < 0 || y >= _cpuDepthTex.height)
            return -1f;

        var data = _cpuDepthTex.GetRawTextureData<ushort>();
        int index = y * _cpuDepthTex.width + x;
        return Mathf.HalfToFloat(data[index]);
    }

    /// <summary>
    /// BoundingBox 영역의 depth를 추출하여 시각화 텍스처 생성
    /// 주의: Depth 텍스처는 RGB 카메라와 다른 좌표계를 사용할 수 있음
    /// </summary>
    public Texture2D CreateBBoxDepthVisualization(BoundingBox box, int outputSize, Vector2Int yoloInputSize)
    {
        if (!IsDepthAvailable || _cpuDepthTex == null)
            return null;

        int depthW = _cpuDepthTex.width;
        int depthH = _cpuDepthTex.height;

        // YOLO 좌표(640x640) → Depth 텍스처 좌표로 변환
        // 주의: 이 변환은 단순 비율이며, 실제로는 카메라 보정이 필요할 수 있음
        float scaleX = (float)depthW / yoloInputSize.x;
        float scaleY = (float)depthH / yoloInputSize.y;

        // [수정] YOLO centered space -> Unity Texture space 변환
        // CenterY가 음수(상단)일 때 Unity Y좌표는 커야 함 (상단)
        int bboxX = Mathf.RoundToInt(box.CenterX * scaleX + depthW / 2f - (box.Width * scaleX / 2f));
        // Y축 반전: (Height/2 - CenterY) - (BoxHeight/2)
        // CenterY는 음수(상단) -> Height/2 - (-상단) = 큰값(상단)
        // Box의 Top-Left(Unity 기준 상단) 구하기
        int bboxY = Mathf.RoundToInt(depthH / 2f - box.CenterY * scaleY - (box.Height * scaleY / 2f));
        int bboxW = Mathf.RoundToInt(box.Width * scaleX);
        int bboxH = Mathf.RoundToInt(box.Height * scaleY);

        // 경계 체크
        bboxX = Mathf.Clamp(bboxX, 0, depthW - 1);
        bboxY = Mathf.Clamp(bboxY, 0, depthH - 1);
        bboxW = Mathf.Clamp(bboxW, 1, depthW - bboxX);
        bboxH = Mathf.Clamp(bboxH, 1, depthH - bboxY);

        Debug.Log($"[Depth BBox] YOLO center({box.CenterX:F0},{box.CenterY:F0}) size({box.Width:F0},{box.Height:F0})");
        Debug.Log($"[Depth BBox] → Depth top-left({bboxX},{bboxY}) size({bboxW},{bboxH}) in {depthW}x{depthH}");

        Texture2D result = new Texture2D(outputSize, outputSize, TextureFormat.RGBA32, false);
        Color[] resultColors = new Color[outputSize * outputSize];
        var depthData = _cpuDepthTex.GetRawTextureData<ushort>();

        float invRange = 1.0f / Mathf.Max(0.0001f, (_visFar - _visNear));

        // BBox 영역의 평균 depth도 계산
        float totalDepth = 0f;
        int validCount = 0;

        for (int y = 0; y < outputSize; y++)
        {
            for (int x = 0; x < outputSize; x++)
            {
                // output 좌표를 bbox 내부 좌표로 매핑
                int depthX = bboxX + Mathf.RoundToInt((float)x / outputSize * bboxW);
                int depthY = bboxY + Mathf.RoundToInt((float)y / outputSize * bboxH);

                depthX = Mathf.Clamp(depthX, 0, depthW - 1);
                depthY = Mathf.Clamp(depthY, 0, depthH - 1);

                int depthIdx = depthY * depthW + depthX;
                float d = Mathf.HalfToFloat(depthData[depthIdx]);

                if (d > 0.0001f && d <= _cutoffDistance)
                {
                    float v = 1.0f - Mathf.Clamp01((d - _visNear) * invRange);
                    resultColors[y * outputSize + x] = new Color(v, v, v);
                    totalDepth += d;
                    validCount++;
                }
                else
                {
                    resultColors[y * outputSize + x] = Color.black;
                }
            }
        }

        if (validCount > 0)
        {
            float avgDepth = totalDepth / validCount;
            Debug.Log($"[Depth BBox] Average depth: {avgDepth:F3}m ({validCount} valid pixels)");
        }

        result.SetPixels(resultColors);
        result.Apply();
        return result;
    }

    /// <summary>
    /// 마스크 텍스처를 기반으로 Depth 추출
    /// 마스크가 있는 영역(alpha > 0)의 bounding box를 찾아 해당 Depth 영역만 크롭
    /// </summary>
    public Texture2D CreateMaskedDepthFromMask(Texture2D maskTex, int outputSize, Vector2Int yoloInputSize)
    {
        if (!IsDepthAvailable || _cpuDepthTex == null || maskTex == null)
            return null;

        int maskW = maskTex.width;   // 160
        int maskH = maskTex.height;  // 160
        int depthW = _cpuDepthTex.width;   // 320
        int depthH = _cpuDepthTex.height;  // 320

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
            Debug.LogWarning("[Depth Mask] No valid mask pixels found!");
            return null;
        }

        int maskBBoxW = maxX - minX + 1;
        int maskBBoxH = maxY - minY + 1;

        // 마스크 좌표 → Depth 좌표 스케일
        // mask 160 → YOLO 640 → Depth 320
        // mask 1 pixel = 4 YOLO pixels, YOLO 640 → Depth 320 = x0.5
        // 따라서 mask 1 pixel = 2 Depth pixels
        float maskToDepthScale = (float)depthW / maskW;

        int depthMinX = Mathf.RoundToInt(minX * maskToDepthScale);
        int depthMinY = Mathf.RoundToInt(minY * maskToDepthScale);
        int depthBBoxW = Mathf.RoundToInt(maskBBoxW * maskToDepthScale);
        int depthBBoxH = Mathf.RoundToInt(maskBBoxH * maskToDepthScale);

        Debug.Log($"[Depth Mask] Mask BBox: ({minX},{minY}) size({maskBBoxW},{maskBBoxH}) in {maskW}x{maskH}");
        Debug.Log($"[Depth Mask] → Depth BBox: ({depthMinX},{depthMinY}) size({depthBBoxW},{depthBBoxH}) in {depthW}x{depthH}");

        Texture2D result = new Texture2D(outputSize, outputSize, TextureFormat.RGBA32, false);
        Color[] resultColors = new Color[outputSize * outputSize];
        var depthData = _cpuDepthTex.GetRawTextureData<ushort>();

        float invRange = 1.0f / Mathf.Max(0.0001f, (_visFar - _visNear));

        // 통계용
        float totalDepth = 0f;
        int validCount = 0;

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
                    // mask 좌표 → Depth 좌표
                    int depthX = Mathf.RoundToInt(maskX * maskToDepthScale);
                    int depthY = Mathf.RoundToInt(maskY * maskToDepthScale);

                    depthX = Mathf.Clamp(depthX, 0, depthW - 1);
                    depthY = Mathf.Clamp(depthY, 0, depthH - 1);

                    int depthIdx = depthY * depthW + depthX;
                    float d = Mathf.HalfToFloat(depthData[depthIdx]);

                    if (d > 0.0001f && d <= _cutoffDistance)
                    {
                        float v = 1.0f - Mathf.Clamp01((d - _visNear) * invRange);
                        resultColors[y * outputSize + x] = new Color(v, v, v);
                        totalDepth += d;
                        validCount++;
                    }
                    else
                    {
                        resultColors[y * outputSize + x] = Color.black;
                    }
                }
                else
                {
                    resultColors[y * outputSize + x] = Color.clear;
                }
            }
        }

        if (validCount > 0)
        {
            float avgDepth = totalDepth / validCount;
            Debug.Log($"[Depth Mask] Average depth: {avgDepth:F3}m ({validCount} valid pixels)");
        }

        result.SetPixels(resultColors);
        result.Apply();
        return result;
    }

    /// <summary>
    /// 마스크 영역의 depth를 추출하여 시각화 텍스처 생성 (레거시)
    /// </summary>
    public Texture2D CreateMaskedDepthVisualization(Texture2D maskTex, int outputWidth, int outputHeight)
    {
        if (!IsDepthAvailable || _cpuDepthTex == null || maskTex == null)
            return null;

        Texture2D result = new Texture2D(outputWidth, outputHeight, TextureFormat.RGBA32, false);
        Color[] resultColors = new Color[outputWidth * outputHeight];

        // 마스크 크기
        int maskW = maskTex.width;
        int maskH = maskTex.height;

        // Depth 크기
        int depthW = _cpuDepthTex.width;
        int depthH = _cpuDepthTex.height;

        var depthData = _cpuDepthTex.GetRawTextureData<ushort>();
        Color32[] maskColors = maskTex.GetPixels32();

        float invRange = 1.0f / Mathf.Max(0.0001f, (_visFar - _visNear));

        for (int y = 0; y < outputHeight; y++)
        {
            for (int x = 0; x < outputWidth; x++)
            {
                // output 좌표를 mask 좌표로 매핑
                int maskX = (int)((float)x / outputWidth * maskW);
                int maskY = (int)((float)y / outputHeight * maskH);
                maskX = Mathf.Clamp(maskX, 0, maskW - 1);
                maskY = Mathf.Clamp(maskY, 0, maskH - 1);

                int maskIdx = maskY * maskW + maskX;
                Color32 maskPixel = maskColors[maskIdx];

                // 마스크가 있는 영역만 처리 (alpha > 0)
                if (maskPixel.a > 0)
                {
                    // output 좌표를 depth 좌표로 매핑
                    int depthX = (int)((float)x / outputWidth * depthW);
                    int depthY = (int)((float)y / outputHeight * depthH);
                    depthX = Mathf.Clamp(depthX, 0, depthW - 1);
                    depthY = Mathf.Clamp(depthY, 0, depthH - 1);

                    int depthIdx = depthY * depthW + depthX;
                    float d = Mathf.HalfToFloat(depthData[depthIdx]);

                    if (d > 0.0001f && d <= _cutoffDistance)
                    {
                        float v = 1.0f - Mathf.Clamp01((d - _visNear) * invRange);
                        resultColors[y * outputWidth + x] = new Color(v, v, v);
                    }
                    else
                    {
                        resultColors[y * outputWidth + x] = Color.black;
                    }
                }
                else
                {
                    resultColors[y * outputWidth + x] = Color.clear;
                }
            }
        }

        result.SetPixels(resultColors);
        result.Apply();
        return result;
    }

    void OnDestroy()
    {
        if (_cpuDepthTex != null)
            Destroy(_cpuDepthTex);
        if (_depthVisTex != null)
            Destroy(_depthVisTex);
    }
}
