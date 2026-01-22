using UnityEngine;
using Unity.InferenceEngine;
using UnityEngine.UI;
using System.Collections.Generic;

/// <summary>
/// [최적화] MonoBehaviour로 변경하여 매 프레임 마스크 위치를 스무딩 업데이트
/// </summary>
public class IEMasker : MonoBehaviour
{
    private const int YOLO11_MASK_HEIGHT = 160;
    private const int YOLO11_MASK_WIDTH = 160;

    [SerializeField] private Transform _displayLocation;
    [SerializeField] private float _positionSmoothTime = 0.05f;
    [SerializeField] private float _sizeSmoothTime = 0.1f;

    private readonly List<RawImage> _maskImages = new();
    private readonly List<Color> _maskColors = new();

    private float _confidenceThreshold = 0.5f;

    // [최적화] 픽셀 배열 캐싱 (GC 감소)
    private Color32[] _cachedPixelArray;

    // [최적화] 마지막으로 그린 마스크 상태 캐싱
    private int _lastTargetIndex = -1;
    private bool _hasCachedMask = false;

    // [최적화] 스무딩을 위한 현재/목표 위치 및 크기
    private Vector2 _currentPosition;
    private Vector2 _targetPosition;
    private Vector2 _currentSize;
    private Vector2 _targetSize;
    private Vector2 _positionVelocity;
    private Vector2 _sizeVelocity;
    private bool _hasTarget = false;

    // 이미지 크기 캐싱
    private int _imageWidth;
    private int _imageHeight;

    /// <summary>
    /// 외부에서 초기화 (IEExecutor에서 호출)
    /// </summary>
    public void Initialize(Transform displayLocation, float confidenceThreshold)
    {
        _displayLocation = displayLocation;
        _confidenceThreshold = confidenceThreshold;
        _cachedPixelArray = new Color32[YOLO11_MASK_HEIGHT * YOLO11_MASK_WIDTH];
    }

    private void Update()
    {
        // 매 프레임 마스크 위치 스무딩 업데이트
        if (_hasTarget && _maskImages.Count > 0 && _maskImages[0].gameObject.activeSelf)
        {
            UpdateMaskTransform();
        }
    }

    /// <summary>
    /// [최적화] 매 프레임 호출 - 마스크 RawImage의 위치와 크기를 스무딩
    /// </summary>
    private void UpdateMaskTransform()
    {
        // 위치 스무딩
        _currentPosition = Vector2.SmoothDamp(_currentPosition, _targetPosition, ref _positionVelocity, _positionSmoothTime);

        // 크기 스무딩
        _currentSize = Vector2.SmoothDamp(_currentSize, _targetSize, ref _sizeVelocity, _sizeSmoothTime);

        // RawImage 업데이트
        if (_maskImages.Count > 0)
        {
            RectTransform rectTransform = _maskImages[0].GetComponent<RectTransform>();
            rectTransform.localPosition = new Vector3(_currentPosition.x, _currentPosition.y, 0);
            rectTransform.sizeDelta = _currentSize;
        }
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

    /// <summary>
    /// [최적화] 단일 객체의 마스크를 그리고, 목표 위치/크기 설정
    /// </summary>
    public void DrawSingleMask(int targetIndex, BoundingBox box, Tensor<float> mask, int imageWidth, int imageHeight)
    {
        if (targetIndex < 0 || mask == null)
        {
            ClearMasks(0);
            _hasCachedMask = false;
            _lastTargetIndex = -1;
            _hasTarget = false;
            return;
        }

        if (mask.shape[1] != YOLO11_MASK_HEIGHT || mask.shape[2] != YOLO11_MASK_WIDTH)
        {
            Debug.LogWarning("Mask shape is invalid.");
            return;
        }

        _imageWidth = imageWidth;
        _imageHeight = imageHeight;

        // [최적화] 목표 위치와 크기 설정 (스무딩은 Update에서 처리)
        Vector2 newTargetPos = new Vector2(box.CenterX, -box.CenterY);
        Vector2 newTargetSize = new Vector2(box.Width, box.Height);

        // 첫 번째 프레임이거나 타겟이 없었으면 즉시 이동
        if (!_hasTarget)
        {
            _currentPosition = newTargetPos;
            _currentSize = newTargetSize;
            _positionVelocity = Vector2.zero;
            _sizeVelocity = Vector2.zero;
        }

        _targetPosition = newTargetPos;
        _targetSize = newTargetSize;
        _hasTarget = true;

        // 텍스처를 먼저 가져와서 활성화 상태 유지
        Texture2D maskTexture = GetMaskTexture(0);

        // 캐싱된 배열 사용 (GC 감소)
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

        maskTexture.SetPixels32(_cachedPixelArray);
        maskTexture.Apply();

        // 캐시 상태 업데이트
        _lastTargetIndex = targetIndex;
        _hasCachedMask = true;

        // 첫 번째 마스크만 사용하고 나머지는 숨기기
        ClearMasks(1);
    }

    /// <summary>
    /// [최적화] Lost frames 동안 마스크 가시성 유지하고 예측 이동
    /// </summary>
    public void KeepCurrentMask()
    {
        if (_hasCachedMask && _maskImages.Count > 0)
        {
            _maskImages[0].gameObject.SetActive(true);
            // 스무딩은 Update에서 계속 처리됨
        }
    }

    /// <summary>
    /// 현재 캐시된 마스크가 있는지 확인
    /// </summary>
    public bool HasCachedMask => _hasCachedMask;

    private void ClearMasks(int lastBoxCount)
    {
        for (int i = lastBoxCount; i < _maskImages.Count; i++)
        {
            _maskImages[i].gameObject.SetActive(false);
        }
    }

    /// <summary>
    /// 모든 마스크를 숨기는 public 메서드
    /// </summary>
    public void ClearAllMasks()
    {
        ClearMasks(0);
        _hasTarget = false;
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

    /// <summary>
    /// [최적화] 마스크용 RawImage와 텍스처 가져오기 (물체 크기에 맞춤)
    /// </summary>
    private Texture2D GetMaskTexture(int segmentationId)
    {
        RawImage maskImage;
        if (segmentationId < _maskImages.Count)
        {
            maskImage = _maskImages[segmentationId];
        }
        else
        {
            maskImage = CreateRawImage(segmentationId);
            _maskImages.Add(maskImage);
        }
        maskImage.gameObject.SetActive(true);

        return maskImage.texture as Texture2D;
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
            maskImage = CreateRawImage(segmentationId);
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

    private RawImage CreateRawImage(int segmentationId)
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
}
