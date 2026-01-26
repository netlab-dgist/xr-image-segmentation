Shader "Custom/PointCloud"
{
    Properties
    {
        _PointSize ("Point Size", Float) = 0.01
        _SizeAttenuation ("Size Attenuation", Float) = 1.0
        _AttenuationMinDist ("Attenuation Min Distance", Float) = 0.5
        _AttenuationMaxDist ("Attenuation Max Distance", Float) = 3.0
    }

    SubShader
    {
        Tags
        {
            "RenderType" = "Opaque"
            "Queue" = "Geometry+100"
        }

        LOD 100
        ZWrite On
        ZTest LEqual
        Cull Off

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 4.5
            #pragma multi_compile_instancing

            #include "UnityCG.cginc"

            StructuredBuffer<float3> _PositionBuffer;
            StructuredBuffer<uint> _ColorBuffer;

            float _PointSize;
            float _SizeAttenuation;
            float _AttenuationMinDist;
            float _AttenuationMaxDist;

            struct v2f
            {
                float4 pos : SV_POSITION;
                float4 color : COLOR;
                float pointSize : PSIZE;

                UNITY_VERTEX_OUTPUT_STEREO
            };

            float4 UnpackColor32(uint packed)
            {
                float4 c;
                c.r = (packed & 0xFF) / 255.0;
                c.g = ((packed >> 8) & 0xFF) / 255.0;
                c.b = ((packed >> 16) & 0xFF) / 255.0;
                c.a = ((packed >> 24) & 0xFF) / 255.0;
                return c;
            }

            v2f vert(uint vertexID : SV_VertexID)
            {
                v2f o;

                UNITY_SETUP_INSTANCE_ID(o);
                UNITY_INITIALIZE_OUTPUT(v2f, o);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);

                float3 worldPos = _PositionBuffer[vertexID];

                o.pos = UnityWorldToClipPos(float4(worldPos, 1.0));
                o.color = UnpackColor32(_ColorBuffer[vertexID]);

                float dist = length(worldPos - _WorldSpaceCameraPos);
                float sizeFactor = 1.0;

                if (_SizeAttenuation > 0.5)
                {
                    float t = saturate((dist - _AttenuationMinDist) /
                                       (_AttenuationMaxDist - _AttenuationMinDist));
                    sizeFactor = lerp(1.0, 0.3, t);
                }

                float screenScale = _ScreenParams.y / 1000.0;
                o.pointSize = _PointSize * sizeFactor * screenScale * 100.0;

                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);
                return i.color;
            }
            ENDCG
        }
    }

    FallBack Off
}
