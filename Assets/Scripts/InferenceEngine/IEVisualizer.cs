using UnityEngine;
using System.Collections.Generic;

public class IEVisualizer : MonoBehaviour
{
    [Header("Visual Settings")]
    [SerializeField] private float _markerSize = 0.05f; // 5cm sphere
    [SerializeField] private Color _markerColor = Color.cyan;
    [SerializeField] private Vector3 _textOffset = new Vector3(0, 0.1f, 0); // Text 10cm above sphere

    private GameObject _currentMarker;
    private TextMesh _currentLabel;

    private void Start()
    {
        CreateMarker();
        HideMarker();
    }

    private void CreateMarker()
    {
        if (_currentMarker != null) return;

        // Create a simple sphere marker
        _currentMarker = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        _currentMarker.name = "AR_Tracked_Marker";
        _currentMarker.transform.localScale = Vector3.one * _markerSize;
        
        // Remove collider to avoid physics interference
        Destroy(_currentMarker.GetComponent<Collider>());

        // Set color
        var renderer = _currentMarker.GetComponent<Renderer>();
        renderer.material = new Material(Shader.Find("Standard"));
        renderer.material.color = _markerColor;

        // Add 3D Text Label
        GameObject textObj = new GameObject("Label");
        textObj.transform.SetParent(_currentMarker.transform);
        textObj.transform.localPosition = _textOffset / _markerSize; // Adjust for parent scale
        
        _currentLabel = textObj.AddComponent<TextMesh>();
        _currentLabel.characterSize = 0.05f;
        _currentLabel.fontSize = 60;
        _currentLabel.anchor = TextAnchor.MiddleCenter;
        _currentLabel.alignment = TextAlignment.Center;
        _currentLabel.color = Color.white;
    }

    public void UpdateMarker(Vector3 worldPosition, string text)
    {
        if (_currentMarker == null) CreateMarker();

        _currentMarker.SetActive(true);
        _currentMarker.transform.position = worldPosition;
        
        // Make text always face the camera (Billboard)
        if (Camera.main != null)
        {
            _currentLabel.transform.rotation = Quaternion.LookRotation(_currentLabel.transform.position - Camera.main.transform.position);
        }

        if (_currentLabel != null)
        {
            _currentLabel.text = text;
        }
    }

    public void HideMarker()
    {
        if (_currentMarker != null)
        {
            _currentMarker.SetActive(false);
        }
    }
}
