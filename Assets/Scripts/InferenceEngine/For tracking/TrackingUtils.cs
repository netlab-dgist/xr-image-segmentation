using UnityEngine;

public static class TrackingUtils
{
    // 두 BoundingBox의 겹치는 영역 비율(IoU)을 계산
    public static float CalculateIoU(BoundingBox boxA, BoundingBox boxB)
    {
        float xA = Mathf.Max(boxA.CenterX - boxA.Width / 2, boxB.CenterX - boxB.Width / 2);
        float yA = Mathf.Max(boxA.CenterY - boxA.Height / 2, boxB.CenterY - boxB.Height / 2);
        float xB = Mathf.Min(boxA.CenterX + boxA.Width / 2, boxB.CenterX + boxB.Width / 2);
        float yB = Mathf.Min(boxA.CenterY + boxA.Height / 2, boxB.CenterY + boxB.Height / 2);

        float interArea = Mathf.Max(0, xB - xA) * Mathf.Max(0, yB - yA);
        if (interArea == 0) return 0;

        float boxAArea = boxA.Width * boxA.Height;
        float boxBArea = boxB.Width * boxB.Height;

        return interArea / (boxAArea + boxBArea - interArea);
    }
}