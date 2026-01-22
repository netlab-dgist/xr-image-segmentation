using UnityEngine;

public static class TrackingUtils
{
    /// <summary>
    /// 두 BoundingBox 사이의 IoU(Intersection over Union)를 계산합니다.
    /// </summary>
    public static float CalculateIoU(BoundingBox boxA, BoundingBox boxB)
    {
        // 각 박스의 좌상단, 우하단 좌표 계산
        float aLeft = boxA.CenterX - boxA.Width / 2f;
        float aRight = boxA.CenterX + boxA.Width / 2f;
        float aTop = boxA.CenterY + boxA.Height / 2f;
        float aBottom = boxA.CenterY - boxA.Height / 2f;

        float bLeft = boxB.CenterX - boxB.Width / 2f;
        float bRight = boxB.CenterX + boxB.Width / 2f;
        float bTop = boxB.CenterY + boxB.Height / 2f;
        float bBottom = boxB.CenterY - boxB.Height / 2f;

        // 교차 영역 계산
        float intersectLeft = Mathf.Max(aLeft, bLeft);
        float intersectRight = Mathf.Min(aRight, bRight);
        float intersectTop = Mathf.Min(aTop, bTop);
        float intersectBottom = Mathf.Max(aBottom, bBottom);

        float intersectWidth = Mathf.Max(0, intersectRight - intersectLeft);
        float intersectHeight = Mathf.Max(0, intersectTop - intersectBottom);
        float intersectionArea = intersectWidth * intersectHeight;

        // 합집합 영역 계산
        float areaA = boxA.Width * boxA.Height;
        float areaB = boxB.Width * boxB.Height;
        float unionArea = areaA + areaB - intersectionArea;

        if (unionArea <= 0) return 0f;

        return intersectionArea / unionArea;
    }
}
