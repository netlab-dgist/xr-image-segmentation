// Depth 센서 Calibration 파라미터 디버거
// RGB 카메라와 Depth 센서의 intrinsic/extrinsic 파라미터를 비교 출력합니다.

using UnityEngine;
#if XR_OCULUS_4_2_0_OR_NEWER
using Unity.XR.Oculus;
#endif
using PassthroughCameraSamples;
using Meta.XR.EnvironmentDepth;

public class DepthCalibrationDebugger : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private EnvironmentDepthManager _depthManager;

    [Header("Debug Settings")]
    [SerializeField] private bool _logEveryFrame = false;
    [SerializeField] private float _logInterval = 1.0f;
    [SerializeField] private KeyCode _logKey = KeyCode.D;

    private float _lastLogTime;

    // Depth 센서 Intrinsic (계산된 값)
    public struct DepthIntrinsics
    {
        public float fx;
        public float fy;
        public float cx;
        public float cy;
        public int width;
        public int height;
    }

    private void Update()
    {
        bool inputDetected = false;

#if XR_OCULUS_4_2_0_OR_NEWER
        // Oculus Controller Input (A Button or Right Index Trigger)
        if (OVRInput.GetDown(OVRInput.Button.One) || 
            OVRInput.GetDown(OVRInput.Button.SecondaryIndexTrigger))
        {
            inputDetected = true;
        }
#endif

        if (inputDetected)
        {
            LogAllCalibrationData();
        }

        // 주기적 로그 출력
        if (_logEveryFrame || Time.time - _lastLogTime > _logInterval)
        {
            LogAllCalibrationData();
            _lastLogTime = Time.time;
        }
    }

    [ContextMenu("Log Calibration Data")]
    public void LogAllCalibrationData()
    {
        Debug.Log("========== RGB-Depth Calibration Data ==========");

        LogRGBCameraData();
        LogDepthSensorData();
        LogRGBToDepthTransform();

        Debug.Log("================================================");
    }

    private void LogRGBCameraData()
    {
        Debug.Log("----- RGB Camera (Left) -----");

        try
        {
            // RGB Intrinsics
            var intrinsics = PassthroughCameraUtils.GetCameraIntrinsics(PassthroughCameraEye.Left);
            Debug.Log($"[RGB Intrinsic]");
            Debug.Log($"  Focal Length: fx={intrinsics.FocalLength.x:F2}, fy={intrinsics.FocalLength.y:F2}");
            Debug.Log($"  Principal Point: cx={intrinsics.PrincipalPoint.x:F2}, cy={intrinsics.PrincipalPoint.y:F2}");
            Debug.Log($"  Resolution: {intrinsics.Resolution.x} x {intrinsics.Resolution.y}");
            Debug.Log($"  Skew: {intrinsics.Skew:F4}");

            // RGB Extrinsics (World Pose)
            var pose = PassthroughCameraUtils.GetCameraPoseInWorld(PassthroughCameraEye.Left);
            Debug.Log($"[RGB Extrinsic - World Pose]");
            Debug.Log($"  Position: {pose.position}");
            Debug.Log($"  Rotation: {pose.rotation.eulerAngles}");
        }
        catch (System.Exception e)
        {
            Debug.LogWarning($"RGB Camera data not available: {e.Message}");
        }
    }

    private void LogDepthSensorData()
    {
        Debug.Log("----- Depth Sensor (Left Eye) -----");

#if !XR_OCULUS_4_2_0_OR_NEWER
        Debug.LogWarning("Depth API requires XR_OCULUS_4_2_0_OR_NEWER. Please check your Oculus XR Plugin version.");
        return;
#else
        try
        {
            // Depth Frame Descriptor 획득
            var frameDesc = Utils.GetEnvironmentDepthFrameDesc(0); // 0 = Left eye

            if (!frameDesc.isValid)
            {
                Debug.LogWarning("Depth frame is not valid. Is EnvironmentDepthManager enabled?");
                return;
            }

            // Depth Extrinsics (Pose)
            Debug.Log($"[Depth Extrinsic - Tracking Space Pose]");
            Debug.Log($"  Position: {frameDesc.createPoseLocation}");
            var rotation = new Quaternion(
                frameDesc.createPoseRotation.x,
                frameDesc.createPoseRotation.y,
                frameDesc.createPoseRotation.z,
                frameDesc.createPoseRotation.w
            );
            Debug.Log($"  Rotation: {rotation.eulerAngles}");

            // FOV 정보
            Debug.Log($"[Depth FOV (tangent)]");
            Debug.Log($"  Left: {frameDesc.fovLeftAngle:F4}, Right: {frameDesc.fovRightAngle:F4}");
            Debug.Log($"  Top: {frameDesc.fovTopAngle:F4}, Down: {frameDesc.fovDownAngle:F4}");

            // FOV를 각도로 변환
            float fovHorizontal = Mathf.Atan(frameDesc.fovLeftAngle) + Mathf.Atan(frameDesc.fovRightAngle);
            float fovVertical = Mathf.Atan(frameDesc.fovTopAngle) + Mathf.Atan(frameDesc.fovDownAngle);
            Debug.Log($"[Depth FOV (degrees)]");
            Debug.Log($"  Horizontal: {fovHorizontal * Mathf.Rad2Deg:F2}°");
            Debug.Log($"  Vertical: {fovVertical * Mathf.Rad2Deg:F2}°");

            // Depth 범위
            Debug.Log($"[Depth Range]");
            Debug.Log($"  Near Z: {frameDesc.nearZ:F3}m, Far Z: {frameDesc.farZ:F3}m");
            Debug.Log($"  Min Depth: {frameDesc.minDepth:F3}m, Max Depth: {frameDesc.maxDepth:F3}m");

            // Depth 텍스처 해상도 확인 (EnvironmentDepthManager가 있는 경우)
            if (_depthManager != null && _depthManager.IsDepthAvailable)
            {
                // Shader global에서 depth 텍스처 정보 가져오기
                var depthTex = Shader.GetGlobalTexture("_EnvironmentDepthTexture");
                if (depthTex != null)
                {
                    Debug.Log($"[Depth Texture]");
                    Debug.Log($"  Resolution: {depthTex.width} x {depthTex.height}");

                    // Intrinsic 계산
                    var depthIntrinsics = CalculateDepthIntrinsics(frameDesc, depthTex.width, depthTex.height);
                    Debug.Log($"[Depth Intrinsic (calculated from FOV)]");
                    Debug.Log($"  Focal Length: fx={depthIntrinsics.fx:F2}, fy={depthIntrinsics.fy:F2}");
                    Debug.Log($"  Principal Point: cx={depthIntrinsics.cx:F2}, cy={depthIntrinsics.cy:F2}");
                }
            }

            // 타이밍 정보
            Debug.Log($"[Timing]");
            Debug.Log($"  Create Time: {frameDesc.createTime:F6}");
            Debug.Log($"  Predicted Display Time: {frameDesc.predictedDisplayTime:F6}");
        }
        catch (System.Exception e)
        {
            Debug.LogWarning($"Depth sensor data not available: {e.Message}");
        }
#endif
    }

#if XR_OCULUS_4_2_0_OR_NEWER
    /// <summary>
    /// FOV 탄젠트 값에서 Depth 센서의 Intrinsic 파라미터를 계산합니다.
    /// </summary>
    public static DepthIntrinsics CalculateDepthIntrinsics(Utils.EnvironmentDepthFrameDesc frameDesc, int width, int height)
    {
        float fovLeft = frameDesc.fovLeftAngle;
        float fovRight = frameDesc.fovRightAngle;
        float fovTop = frameDesc.fovTopAngle;
        float fovDown = frameDesc.fovDownAngle;

        // 초점 거리 계산: fx = width / (tan(left) + tan(right))
        // FOV 값이 이미 탄젠트이므로 그대로 사용
        float fx = width / (fovLeft + fovRight);
        float fy = height / (fovTop + fovDown);

        // 주점 계산: 비대칭 FOV 고려
        // cx = width * tan(left) / (tan(left) + tan(right))
        float cx = width * fovLeft / (fovLeft + fovRight);
        float cy = height * fovTop / (fovTop + fovDown);

        return new DepthIntrinsics
        {
            fx = fx,
            fy = fy,
            cx = cx,
            cy = cy,
            width = width,
            height = height
        };
    }
#endif

    private void LogRGBToDepthTransform()
    {
        Debug.Log("----- RGB → Depth Transform -----");

#if !XR_OCULUS_4_2_0_OR_NEWER
        Debug.LogWarning("Depth API requires XR_OCULUS_4_2_0_OR_NEWER");
        return;
#else
        try
        {
            // RGB 카메라 월드 포즈
            var rgbPose = PassthroughCameraUtils.GetCameraPoseInWorld(PassthroughCameraEye.Left);

            // Depth 센서 프레임 정보
            var frameDesc = Utils.GetEnvironmentDepthFrameDesc(0);

            if (!frameDesc.isValid)
            {
                Debug.LogWarning("Cannot calculate transform: Depth frame not valid");
                return;
            }

            // Depth 센서 포즈 (Tracking Space 기준)
            Vector3 depthPos = frameDesc.createPoseLocation;
            Quaternion depthRot = new Quaternion(
                frameDesc.createPoseRotation.x,
                frameDesc.createPoseRotation.y,
                frameDesc.createPoseRotation.z,
                frameDesc.createPoseRotation.w
            );

            // 두 센서 간 위치 차이 (대략적인 오프셋)
            // 주의: RGB는 World space, Depth는 Tracking space이므로 직접 비교는 정확하지 않음
            // 정확한 계산을 위해서는 둘 다 같은 좌표계로 변환해야 함

            // Head 포즈 획득 (Tracking space → World 변환에 사용)
            var headPose = OVRPlugin.GetNodePoseStateImmediate(OVRPlugin.Node.Head).Pose.ToOVRPose();

            // Depth 포즈를 World space로 변환
            Matrix4x4 trackingToWorld = Matrix4x4.TRS(headPose.position, headPose.orientation, Vector3.one);
            // Depth는 Tracking space 기준이므로, Head 기준으로 해석
            // (정확한 변환은 EnvironmentDepthManager.GetTrackingSpaceWorldToLocalMatrix() 필요)

            Vector3 depthWorldPos = trackingToWorld.MultiplyPoint3x4(depthPos);
            Quaternion depthWorldRot = headPose.orientation * depthRot;

            // RGB → Depth 상대 변환
            Vector3 offset = depthWorldPos - rgbPose.position;
            Quaternion relativeRot = Quaternion.Inverse(rgbPose.rotation) * depthWorldRot;

            Debug.Log($"[Estimated Sensor Offset]");
            Debug.Log($"  Position Offset: {offset} (magnitude: {offset.magnitude:F4}m)");
            Debug.Log($"  Rotation Offset: {relativeRot.eulerAngles}");

            // 변환 행렬 계산
            Matrix4x4 rgbToWorld = Matrix4x4.TRS(rgbPose.position, rgbPose.rotation, Vector3.one);
            Matrix4x4 depthToWorld = Matrix4x4.TRS(depthWorldPos, depthWorldRot, Vector3.one);
            Matrix4x4 rgbToDepth = depthToWorld.inverse * rgbToWorld;

            Debug.Log($"[RGB → Depth Transform Matrix]");
            Debug.Log($"  {rgbToDepth.GetRow(0)}");
            Debug.Log($"  {rgbToDepth.GetRow(1)}");
            Debug.Log($"  {rgbToDepth.GetRow(2)}");
            Debug.Log($"  {rgbToDepth.GetRow(3)}");
        }
        catch (System.Exception e)
        {
            Debug.LogWarning($"Transform calculation failed: {e.Message}");
        }
#endif
    }

    /// <summary>
    /// RGB 픽셀 좌표를 Depth 픽셀 좌표로 변환합니다.
    /// </summary>
    public static Vector2Int ProjectRGBToDepth(
        Vector2Int rgbPixel,
        float depth,
        PassthroughCameraIntrinsics rgbIntrinsics,
        DepthIntrinsics depthIntrinsics,
        Matrix4x4 rgbToDepth)
    {
        // 1. RGB 픽셀 → RGB 카메라 좌표계 3D 점
        float x_rgb = (rgbPixel.x - rgbIntrinsics.PrincipalPoint.x) / rgbIntrinsics.FocalLength.x * depth;
        float y_rgb = (rgbPixel.y - rgbIntrinsics.PrincipalPoint.y) / rgbIntrinsics.FocalLength.y * depth;
        float z_rgb = depth;
        Vector3 pointInRGB = new Vector3(x_rgb, y_rgb, z_rgb);

        // 2. RGB 좌표계 → Depth 좌표계
        Vector3 pointInDepth = rgbToDepth.MultiplyPoint3x4(pointInRGB);

        // 3. Depth 좌표계 → Depth 픽셀
        int u_depth = Mathf.RoundToInt(depthIntrinsics.fx * pointInDepth.x / pointInDepth.z + depthIntrinsics.cx);
        int v_depth = Mathf.RoundToInt(depthIntrinsics.fy * pointInDepth.y / pointInDepth.z + depthIntrinsics.cy);

        return new Vector2Int(u_depth, v_depth);
    }
}
