Shader "Custom/PointColorShader"
{
    Properties
    {
        _PointSize("Point Size", Float) = 5.0
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            struct appdata {
                float4 vertex : POSITION;
                float4 color : COLOR;
            };

            struct v2f {
                float4 vertex : SV_POSITION;
                float4 color : COLOR;
                float size : PSIZE;
            };

            float _PointSize;

            v2f vert (appdata v) {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.color = v.color;
                o.size = _PointSize;
                return o;
            }

            fixed4 frag (v2f i) : SV_Target {
                return i.color;
            }
            ENDCG
        }
    }
}