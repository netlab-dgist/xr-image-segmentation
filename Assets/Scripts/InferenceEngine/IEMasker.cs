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

    // 외부에서 현재 마스크 텍스처에 접근할 수 있도록 추가
    private Texture2D _currentMaskTexture;
    public Texture2D CurrentMaskTexture => _currentMaskTexture;

    public void UpdateDisplayLocation(Transform newLoc)
    {
        _displayLocation = newLoc;
        foreach (var img in _maskImages)
        {
            if (img != null) img.transform.SetParent(_displayLocation, false);
        }
    }

    public IEMasker(Transform displayLocation, float confidenceThreshold)
    {
        _displayLocation = displayLocation;
        _confidenceThreshold = confidenceThreshold;
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
                    float value = mask[i, y, x];

                    int posX = x;
                    int posY = YOLO11_MASK_HEIGHT - y - 1;

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
            maskTexture.SetPixels32(pixelArray);
            maskTexture.Apply();
        }
        ClearMasks(numObjects);
    }

    public void DrawSingleMask(int targetIndex, BoundingBox box, Tensor<float> mask, int imageWidth, int imageHeight)
    {
        ClearMasks(0); 

        if (targetIndex < 0 || targetIndex >= mask.shape[0]) return;

        Texture2D maskTexture = GetTexture(0, imageWidth, imageHeight);
        
        int maskW = YOLO11_MASK_WIDTH; 
        int maskH = YOLO11_MASK_HEIGHT;
        
        Color32[] pixelArray = new Color32[maskW * maskH];

        for (int y = 0; y < maskH; y++)
        {
            for (int x = 0; x < maskW; x++)
            {
                float value = mask[targetIndex, y, x];
                int posX = x;
                int posY = maskH - y - 1;

                if (value > _confidenceThreshold && PixelInBoundingBox(box, posX, posY, imageWidth, imageHeight))
                {
                    pixelArray[posY * maskW + posX] = new Color(0, 1, 0, 0.6f); 
                }
                else
                {
                    pixelArray[posY * maskW + posX] = Color.clear;
                }
            }
        }

        maskTexture.SetPixels32(pixelArray);
        maskTexture.Apply();

        // 현재 마스크 텍스처 저장
        _currentMaskTexture = maskTexture;
    }

    public void KeepCurrentMask()
    {
        // 아무것도 하지 않으면 현재 그려진 마스크가 그대로 유지됨
    }

    public void ClearAllMasks()
    {
        ClearMasks(0);
    }

    private void ClearMasks(int lastBoxCount)
    {
        for (int i = lastBoxCount; i < _maskImages.Count; i++)
        {
            _maskImages[i].gameObject.SetActive(false);
        }
    }

    private bool PixelInBoundingBox(BoundingBox box, int x, int y, int imageWidth, int imageHeight)
    {
        float xScaleFactor = YOLO11_MASK_WIDTH / (float)imageWidth;
        float yScaleFactor = YOLO11_MASK_HEIGHT / (float)imageHeight;

        float centerX = (box.CenterX * xScaleFactor) + (YOLO11_MASK_WIDTH / 2);
        float centerY = (YOLO11_MASK_HEIGHT / 2) - (box.CenterY * yScaleFactor);

        float halfWidth = box.Width * xScaleFactor / 2;
        float halfHeight = box.Height * yScaleFactor / 2;

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
        if (x < 0 || x >= maskTexture.width || y < 0 || y >= maskTexture.height) return;
        maskTexture.SetPixel(x, y, color);
    }
}