using UnityEngine;
using Unity.InferenceEngine;
using UnityEngine.UI;
using System.Collections.Generic;

public class IEMasker
{
    [SerializeField] private Transform _displayLocation;
    private const int YOLO11_MASK_HEIGHT = 160;
    private const int YOLO11_MASK_WIDTH = 160;

    private readonly List<RawImage> _maskImages = new();
    private readonly List<Color> _maskColors = new();

    private float _confidenceThreshold = 0.5f;

    // [깜빡임 방지] 픽셀 배열 캐싱 (GC 감소)
    private Color32[] _cachedPixelArray;

    // [깜빡임 방지] 마지막으로 그린 마스크 상태 캐싱
    private int _lastTargetIndex = -1;
    private bool _hasCachedMask = false;

    public IEMasker(Transform displayLocation, float confidenceThreshold)
    {
        _displayLocation = displayLocation;
        _confidenceThreshold = confidenceThreshold;
        _cachedPixelArray = new Color32[YOLO11_MASK_HEIGHT * YOLO11_MASK_WIDTH];
    }

    public void DrawMask(List<BoundingBox> boundBoxes, Tensor<float> mask, int imageWidth, int imageHeight)
    {
        int numObjects = mask.shape[0];
        if (numObjects <= 0 || mask.shape[1] != YOLO11_MASK_HEIGHT || mask.shape[2] != YOLO11_MASK_WIDTH)
        {
            Debug.LogWarning("No objects found or mask shape is invalid.");
            return;
        }

        Color32[] pixelArray = new Color32[YOLO11_MASK_HEIGHT * YOLO11_MASK_WIDTH];

        for (int i = 0; i < numObjects; i++)
        {
            Texture2D maskTexture = GetTexture(i, imageWidth, imageHeight);
            for (int y = 0; y < YOLO11_MASK_HEIGHT; y++)
            {
                for (int x = 0; x < YOLO11_MASK_WIDTH; x++)
                {
                    // Get the probability value at the current pixel
                    float value = mask[i, y, x];

                    int posX = x;
                    int posY = YOLO11_MASK_HEIGHT - y - 1;

                    // If the value is greater than confidenceThreshold and is inside the box, draw the pixel
                    if (value > _confidenceThreshold && PixelInBoundingBox(boundBoxes[i], posX, posY, imageWidth, imageHeight))
                    {
                        pixelArray[posY * YOLO11_MASK_WIDTH + posX] = GetColor(i);
                    }
                    else
                    {
                        pixelArray[posY * YOLO11_MASK_WIDTH + posX] = Color.clear;
                    }
                }
            }
            // DebugDrawBoundingBox(maskTexture, boundBoxes[i], Color.black, (int)imageWidth, (int)imageHeight);
            maskTexture.SetPixels32(pixelArray);
            maskTexture.Apply();
        }
        ClearMasks(numObjects);
    }

    /// <summary>
    /// 단일 객체의 마스크만 그립니다. 트래킹 모드에서 사용됩니다.
    /// </summary>
    /// <param name="targetIndex">그릴 객체의 인덱스. -1이면 모든 마스크를 숨깁니다.</param>
    public void DrawSingleMask(int targetIndex, BoundingBox box, Tensor<float> mask, int imageWidth, int imageHeight)
    {
        // 인덱스가 -1이면 모든 마스크 숨기기
        if (targetIndex < 0 || mask == null)
        {
            ClearMasks(0);
            _hasCachedMask = false;
            _lastTargetIndex = -1;
            return;
        }

        if (mask.shape[1] != YOLO11_MASK_HEIGHT || mask.shape[2] != YOLO11_MASK_WIDTH)
        {
            Debug.LogWarning("Mask shape is invalid.");
            return;
        }

        // [깜빡임 방지] 텍스처를 먼저 가져와서 활성화 상태 유지
        Texture2D maskTexture = GetTexture(0, imageWidth, imageHeight);

        // [깜빡임 방지] 캐싱된 배열 사용 (GC 감소)
        Color32 maskColor = GetColor(0);

        for (int y = 0; y < YOLO11_MASK_HEIGHT; y++)
        {
            for (int x = 0; x < YOLO11_MASK_WIDTH; x++)
            {
                float value = mask[targetIndex, y, x];

                int posX = x;
                int posY = YOLO11_MASK_HEIGHT - y - 1;

                if (value > _confidenceThreshold && PixelInBoundingBox(box, posX, posY, imageWidth, imageHeight))
                {
                    _cachedPixelArray[posY * YOLO11_MASK_WIDTH + posX] = maskColor;
                }
                else
                {
                    _cachedPixelArray[posY * YOLO11_MASK_WIDTH + posX] = Color.clear;
                }
            }
        }

        // [깜빡임 방지] 텍스처 업데이트 후 불필요한 마스크 숨기기
        maskTexture.SetPixels32(_cachedPixelArray);
        maskTexture.Apply();

        // 캐시 상태 업데이트
        _lastTargetIndex = targetIndex;
        _hasCachedMask = true;

        // 첫 번째 마스크만 사용하고 나머지는 숨기기
        ClearMasks(1);
    }

    /// <summary>
    /// [깜빡임 방지] Lost frames 동안 마스크 가시성만 유지 (텍스처 업데이트 없음)
    /// </summary>
    public void KeepCurrentMask()
    {
        if (_hasCachedMask && _maskImages.Count > 0)
        {
            _maskImages[0].gameObject.SetActive(true);
        }
    }

    /// <summary>
    /// [깜빡임 방지] 현재 캐시된 마스크가 있는지 확인
    /// </summary>
    public bool HasCachedMask => _hasCachedMask;

    private void ClearMasks(int lastBoxCount)
    {
        // Disable all mask images that are not needed
        for (int i = lastBoxCount; i < _maskImages.Count; i++)
        {
            _maskImages[i].gameObject.SetActive(false);
        }
    }

    /// <summary>
    /// [깜빡임 방지] 모든 마스크를 숨기는 public 메서드
    /// </summary>
    public void ClearAllMasks()
    {
        ClearMasks(0);
    }

    private void DebugDrawBoundingBox(Texture2D maskTexture, BoundingBox box, Color color, int imageWidth, int imageHeight)
    {
        // Scale factor to reduce the bounding box size
        float xScaleFactor = YOLO11_MASK_WIDTH / (float)imageWidth;
        float yScaleFactor = YOLO11_MASK_HEIGHT / (float)imageHeight;

        // Convert bounding box center from middle-origin to top-left origin and apply scaling factor
        int centerX = Mathf.RoundToInt(box.CenterX * xScaleFactor) + YOLO11_MASK_WIDTH / 2;
        int centerY = Mathf.RoundToInt(YOLO11_MASK_HEIGHT / 2 - (box.CenterY * yScaleFactor));  // Invert Y coordinate

        // Calculate dimensions of the scaled bounding box
        int width = Mathf.RoundToInt(box.Width * xScaleFactor);
        int height = Mathf.RoundToInt(box.Height * yScaleFactor);

        // Draw the bounding box on the mask texture
        for (int x = centerX - width / 2; x <= centerX + width / 2; x++)
        {
            DrawPixel(maskTexture, x, centerY - height / 2, color); // Top edge
            DrawPixel(maskTexture, x, centerY + height / 2, color); // Bottom edge
        }
        for (int y = centerY - height / 2; y <= centerY + height / 2; y++)
        {
            DrawPixel(maskTexture, centerX - width / 2, y, color); // Left edge
            DrawPixel(maskTexture, centerX + width / 2, y, color); // Right edge
        }
    }

    private bool PixelInBoundingBox(BoundingBox box, int x, int y, int imageWidth, int imageHeight)
    {
        // Scale factor to reduce the bounding box size
        float xScaleFactor = YOLO11_MASK_WIDTH / (float)imageWidth;
        float yScaleFactor = YOLO11_MASK_HEIGHT / (float)imageHeight;

        // Convert bounding box center from middle-origin to top-left origin and apply scaling factor
        float centerX = (box.CenterX * xScaleFactor) + (YOLO11_MASK_WIDTH / 2);
        float centerY = (YOLO11_MASK_HEIGHT / 2) - (box.CenterY * yScaleFactor); // Invert Y coordinate

        // Calculate half-dimensions of the scaled bounding box
        float halfWidth = box.Width * xScaleFactor / 2;
        float halfHeight = box.Height * yScaleFactor / 2;

        // Check if pixel is within the bounding box boundaries
        return x >= (centerX - halfWidth) &&
               x <= (centerX + halfWidth) &&
               y >= (centerY - halfHeight) &&
               y <= (centerY + halfHeight);
    }

    private Texture2D GetTexture(int segmentationId, int imageWidth, int imageHeight)
    {
        RawImage maskImage;
        if (segmentationId < _maskImages.Count)
        {
            maskImage = _maskImages[segmentationId];
        }
        else
        {
            maskImage = CreateRawImage(segmentationId, imageWidth, imageHeight);
            _maskImages.Add(maskImage);
        }
        maskImage.gameObject.SetActive(true);

        RectTransform rectTransform = maskImage.GetComponent<RectTransform>();
        rectTransform.sizeDelta = new Vector2(imageWidth, imageHeight);
        rectTransform.localPosition = Vector3.zero;

        return maskImage.texture as Texture2D;
    }

    private Color GetColor(int segmentationId)
    {
        if (segmentationId < _maskColors.Count)
        {
            return _maskColors[segmentationId];
        }
        else
        {
            Color newColor = new(UnityEngine.Random.value, UnityEngine.Random.value, UnityEngine.Random.value, 0.75f);
            _maskColors.Add(newColor);
            return newColor;
        }
    }

    private RawImage CreateRawImage(int segmentationId, int imageWidth, int imageHeight)
    {
        GameObject maskObject = new GameObject("MaskImage " + segmentationId);
        maskObject.transform.SetParent(_displayLocation, false);

        RawImage rawImage = maskObject.AddComponent<RawImage>();
        rawImage.color = Color.white;
        rawImage.texture = CreateTexture();

        return rawImage;
    }

    private Texture2D CreateTexture()
    {
        return new(YOLO11_MASK_WIDTH, YOLO11_MASK_HEIGHT, TextureFormat.RGBA32, false)
        {
            filterMode = FilterMode.Bilinear,
            wrapMode = TextureWrapMode.Clamp
        };
    }

    private void DrawPixel(Texture2D maskTexture, int x, int y, Color color)
    {
        // Ensure coordinates are within bounds
        if (x < 0 || x >= maskTexture.width || y < 0 || y >= maskTexture.height) return;

        maskTexture.SetPixel(x, y, color);
    }
}
