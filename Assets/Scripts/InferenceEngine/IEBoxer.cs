using System.Collections.Generic;
using UnityEngine;
using Unity.InferenceEngine;
using UnityEngine.UI;

public struct BoundingBox
{
    public float CenterX;
    public float CenterY;
    public float Width;
    public float Height;
    public string Label;
    public Vector3? WorldPos;
    public string ClassName;
}

public class IEBoxer : MonoBehaviour
{
    [SerializeField] private Transform _displayLocation;
    [SerializeField] private TextAsset _labelsAsset;
    [SerializeField] private Color _boxColor;
    [SerializeField] private Sprite _boxTexture;
    [SerializeField] private Font _font;
    [SerializeField] private Color _fontColor;
    [SerializeField] private int _fontSize = 80;

    private string[] _labels;
    private List<GameObject> _boxPool = new();


    private void Start()
    {
        _labels = _labelsAsset.text.Split(new[] { '\n', '\r' }, System.StringSplitOptions.RemoveEmptyEntries);
        Debug.Log($"Loaded {_labels.Length} labels from {_labelsAsset.name}");
    }

    public List<BoundingBox> DrawBoxes(Tensor<float> output, Tensor<int> labelIds, float imageWidth, float imageHeight)
    {
        List<BoundingBox> boundingBoxes = new();

        var scaleX = imageWidth / 640;
        var scaleY = imageHeight / 640;

        var halfWidth = imageWidth / 2;
        var halfHeight = imageHeight / 2;

        int boxesFound = output.shape[0];
        if (boxesFound <= 0) return boundingBoxes;

        var maxBoxes = Mathf.Min(boxesFound, 200);

        for (var n = 0; n < maxBoxes; n++)
        {
            // Get bounding box center coordinates
            var centerX = output[n, 0] * scaleX - halfWidth;
            var centerY = output[n, 1] * scaleY - halfHeight;

            // Get object class name
            var classname = _labels[labelIds[n]].Replace(" ", "_");

            // Create a new bounding box
            var box = new BoundingBox
            {
                CenterX = centerX,
                CenterY = centerY,
                ClassName = classname,
                Width = output[n, 2] * scaleX,
                Height = output[n, 3] * scaleY,
                Label = $"{classname}",
            };

            //Debug.Log($"Box {n}: {box.Label} - Center: ({box.CenterX}, {box.CenterY}), Size: ({box.Width}, {box.Height})");

            boundingBoxes.Add(box);

            DrawBox(box, n);
        }
        ClearBoxes(maxBoxes);

        return boundingBoxes;
    }

    public void ClearBoxes(int lastBoxCount)
    {
        if (lastBoxCount < _boxPool.Count)
        {
            for (int i = lastBoxCount; i < _boxPool.Count; i++)
            {
                if (_boxPool[i] != null)
                {
                    _boxPool[i].SetActive(false);
                }
            }
        }
    }

    private void DrawBox(BoundingBox box, int id)
    {   
        
        GameObject panel;
        if (id < _boxPool.Count)
        {
            panel = _boxPool[id];
            if (panel == null)
            {
                panel = CreateNewBox(_boxColor);
            }
            else
            {
                panel.SetActive(true);
            }
        }
        else
        {
            panel = CreateNewBox(_boxColor);
        }

        // Set box position
        panel.transform.localPosition = new Vector3(box.CenterX, -box.CenterY, box.WorldPos.HasValue ? box.WorldPos.Value.z : 0.0f);

        // Set box size
        RectTransform rectTransform = panel.GetComponent<RectTransform>();
        rectTransform.sizeDelta = new Vector2(box.Width, box.Height);

        // Set label text
        Text label = panel.GetComponentInChildren<Text>();
        label.text = box.Label;
    }

    private GameObject CreateNewBox(Color color)
    {
        // Create the box and set image
        GameObject panel = new("ObjectBox");
        panel.AddComponent<CanvasRenderer>();

        Image image = panel.AddComponent<Image>();
        image.color = color;
        image.sprite = _boxTexture;
        image.type = Image.Type.Sliced;
        image.fillCenter = false;
        panel.transform.SetParent(_displayLocation, false);

        // Create the label
        GameObject textGameObject = new("ObjectLabel");
        textGameObject.AddComponent<CanvasRenderer>();
        textGameObject.transform.SetParent(panel.transform, false);

        Text text = textGameObject.AddComponent<Text>();
        text.font = _font;
        text.color = _fontColor;
        text.fontSize = _fontSize;
        text.horizontalOverflow = HorizontalWrapMode.Overflow;

        RectTransform rectTransform = textGameObject.GetComponent<RectTransform>();
        rectTransform.offsetMin = new Vector2(20, rectTransform.offsetMin.y);
        rectTransform.offsetMax = new Vector2(0, rectTransform.offsetMax.y);
        rectTransform.offsetMin = new Vector2(rectTransform.offsetMin.x, 0);
        rectTransform.offsetMax = new Vector2(rectTransform.offsetMax.x, 30);
        rectTransform.anchorMin = new Vector2(0, 0);
        rectTransform.anchorMax = new Vector2(1, 1);

        _boxPool.Add(panel);

        return panel;
    }
    public void HideAllBoxes()
    {
        foreach (var box in _boxPool)
        {
            if (box != null) box.SetActive(false);
        }
    }

    // 특정 박스 하나만 시각화하는 공개 함수
    public void DrawSpecificBox(BoundingBox box, int id)
    {
        DrawBox(box, id);
    }

    /// <summary>
    /// [깜빡임 방지] 레이블 ID로 클래스 이름 가져오기
    /// </summary>
    public string GetClassName(int labelId)
    {
        if (_labels == null || labelId < 0 || labelId >= _labels.Length)
            return "unknown";
        return _labels[labelId].Replace(" ", "_");
    }

}
