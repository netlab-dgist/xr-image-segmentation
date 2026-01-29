using UnityEngine;
using PassthroughCameraSamples;
using Meta.XR.EnvironmentDepth;
using UnityEngine.Rendering;
#if XR_OCULUS_4_2_0_OR_NEWER
using Unity.XR.Oculus;
#endif

/// <summary>
/// Converts Inference (RGB) coordinates to World Space coordinates using the Preprocessed Depth API.
/// Prioritizes accuracy and calibration.
/// </summary>
public class IEInferenceToWorld : MonoBehaviour
{
    [SerializeField] private EnvironmentDepthManager _depthManager;
    private OVRCameraRig _cameraRig; 
    
    // Cached Calibration Data
    private PassthroughCameraIntrinsics _rgbIntrinsics;
    private bool _isCalibrationReady = false;

    // CPU Depth Buffer
    private Texture2D _cpuDepthTexture;
    private int _depthWidth;
    private int _depthHeight;

    private void Start()
    {
        if (_depthManager == null)
            _depthManager = FindFirstObjectByType<EnvironmentDepthManager>();
        
        _cameraRig = FindFirstObjectByType<OVRCameraRig>();
    }

    private void OnDestroy()
    {
        if (_cpuDepthTexture != null) Destroy(_cpuDepthTexture);
    }

    private void Update()
    {
        UpdateCpuDepthMap();
    }

    public void UpdateCpuDepthMap()
    {
        // 1. Get the Preprocessed Depth Texture (Safety first)
        var depthRT = Shader.GetGlobalTexture("_PreprocessedEnvironmentDepthTexture") as RenderTexture;
        
        // If preprocessed is not ready, try raw environment depth (fallback)
        if (depthRT == null)
            depthRT = Shader.GetGlobalTexture("_EnvironmentDepthTexture") as RenderTexture;

        if (depthRT == null) 
        {
            // Log only occasionally to avoid spam
            if (Time.frameCount % 120 == 0) Debug.LogWarning("[IEInferenceToWorld] Depth Texture is NULL. Check EnvironmentDepthManager.");
            return;
        }

        // 2. Initialize CPU Texture if needed
        // Note: Preprocessed texture is usually Texture2DArray (Stereo). We need Slice 0 (Left Eye).
        if (_cpuDepthTexture == null || _cpuDepthTexture.width != depthRT.width || _cpuDepthTexture.height != depthRT.height)
        {
            if (_cpuDepthTexture != null) Destroy(_cpuDepthTexture);
            
            // Format 48 is RGBAHalf (64 bit). Match it to avoid CopyTexture errors.
            _cpuDepthTexture = new Texture2D(depthRT.width, depthRT.height, TextureFormat.RGBAHalf, false);
            _depthWidth = depthRT.width;
            _depthHeight = depthRT.height;
            Debug.Log($"[IEInferenceToWorld] Initialized CPU Depth Texture: {_depthWidth}x{_depthHeight} (RGBAHalf)");
        }

        // 3. Copy from GPU to CPU
        // Handle Texture2DArray (Slice 0) or standard Texture2D
        if (depthRT.dimension == TextureDimension.Tex2DArray)
        {
            Graphics.CopyTexture(depthRT, 0, 0, _cpuDepthTexture, 0, 0);
        }
        else
        {
            // Standard Blit/Copy for 2D
            RenderTexture currentActive = RenderTexture.active;
            RenderTexture.active = depthRT;
            _cpuDepthTexture.ReadPixels(new Rect(0, 0, depthRT.width, depthRT.height), 0, 0);
            _cpuDepthTexture.Apply();
            RenderTexture.active = currentActive;
        }

        // 4. Update Calibration once
        if (!_isCalibrationReady)
        {
            UpdateCalibration();
        }
    }

    private void UpdateCalibration()
    {
#if XR_OCULUS_4_2_0_OR_NEWER
        try
        {
            _rgbIntrinsics = PassthroughCameraUtils.GetCameraIntrinsics(PassthroughCameraEye.Left);
            _isCalibrationReady = true;
            Debug.Log("[IEInferenceToWorld] Calibration Initialized Successfully.");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"[IEInferenceToWorld] Calibration Failed: {e.Message}");
        }
#endif
    }

    public Vector3 GetWorldPositionFromRGB(Vector2 rgbPixel, int rgbWidth, int rgbHeight)
    {
        if (!_isCalibrationReady)
        {
            if (Time.frameCount % 60 == 0) Debug.LogWarning("[IEInferenceToWorld] Calibration NOT Ready");
            return Vector3.zero;
        }
        if (_cpuDepthTexture == null)
        {
            if (Time.frameCount % 60 == 0) Debug.LogWarning("[IEInferenceToWorld] CPU Depth Texture is NULL");
            return Vector3.zero;
        }

        // 1. Normalize RGB Coordinate (0..1)
        float u = rgbPixel.x / rgbWidth;
        float v = rgbPixel.y / rgbHeight;

        // 2. Sample Depth
        float depth = SampleDepth(u, 1.0f - v); 

        if (depth <= 0.05f) 
        {
            // Log only occasionally to avoid spam, but enough to know it's happening
            if (Time.frameCount % 60 == 0) Debug.Log($"[IEInferenceToWorld] Invalid Depth: {depth:F4} at UV({u:F2}, {v:F2})");
            return Vector3.zero; 
        }

        // ... (rest of the logic)

        // 3. Unproject to 3D (Camera Space)
        // Formula: P = (u - cx) * Z / fx, (v - cy) * Z / fy
        // Note: RGB Intrinsics are in RGB Pixel Space.
        
        float z_cam = depth;
        float x_cam = (rgbPixel.x - _rgbIntrinsics.PrincipalPoint.x) * z_cam / _rgbIntrinsics.FocalLength.x;
        float y_cam = (rgbPixel.y - _rgbIntrinsics.PrincipalPoint.y) * z_cam / _rgbIntrinsics.FocalLength.y;
        
        // Coordinate System Correction:
        // Image Y increases downwards (Top-Left origin).
        // Camera Space Y increases upwards.
        // So we must invert Y.
        y_cam = -y_cam;
        
        // Also check X mirroring.
        // If previous tests showed X was inverted, apply -x.
        x_cam = -x_cam; 

        Vector3 pointInRgbCamera = new Vector3(x_cam, y_cam, z_cam);

        // 4. Transform to World Space
        // RGB Camera Pose in World
        var rgbPose = PassthroughCameraUtils.GetCameraPoseInWorld(PassthroughCameraEye.Left);
        Matrix4x4 rgbToWorld = Matrix4x4.TRS(rgbPose.position, rgbPose.rotation, Vector3.one);

        return rgbToWorld.MultiplyPoint3x4(pointInRgbCamera);
    }

    private float SampleDepth(float u, float v)
    {
        int x = Mathf.Clamp((int)(u * _depthWidth), 0, _depthWidth - 1);
        int y = Mathf.Clamp((int)(v * _depthHeight), 0, _depthHeight - 1);
        
        // Raw RHalf data is linear meters
        return _cpuDepthTexture.GetPixel(x, y).r;
    }
}
