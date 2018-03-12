#include <pycuda-helpers.hpp>

const int NUM_COEFFICIENTS = 10;
const int MAX_PATCH_SIZE = 2 * 5 + 1;
__constant__ float weights[NUM_COEFFICIENTS * MAX_PATCH_SIZE * MAX_PATCH_SIZE * MAX_PATCH_SIZE];
__constant__ float invG[NUM_COEFFICIENTS * NUM_COEFFICIENTS];

__device__ inline int getWeightIdx(int x, int y, int z, int weightIdx, int patchSize)
{
    return weightIdx * patchSize * patchSize * patchSize + z * patchSize * patchSize + y * patchSize + x;
}

__device__ inline int getImageIdx(int x, int y, int z, int imgWidth, int imgHeight)
{
    return z * imgHeight * imgWidth + y * imgWidth + x;
}

__global__ void calcPolyCoeficients(const float *__restrict__ img3d,
                                    float *__restrict__ polyCoefficients,
                                    int imgWidth,
                                    int imgHeight,
                                    int imgDepth,
                                    int patchSize)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= imgWidth || y >= imgHeight)
        return;

    int halfX = patchSize / 2;
    int halfY = patchSize / 2;
    int halfZ = patchSize / 2;

    for (int z = 0; z < imgDepth; z++)
    {

        float sum[NUM_COEFFICIENTS] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        for (int filterIdxZ = 0; filterIdxZ < patchSize; filterIdxZ++)
        {
            int idxZ = z + filterIdxZ - halfZ;
            idxZ = max(0, idxZ);
            idxZ = min(idxZ, imgDepth - 1);

            for (int filterIdxY = 0; filterIdxY < patchSize; filterIdxY++)
            {
                int idxY = y + filterIdxY - halfY;
                idxY = max(0, idxY);
                idxY = min(idxY, imgHeight - 1);

                for (int filterIdxX = 0; filterIdxX < patchSize; filterIdxX++)
                {
                    int idxX = x + filterIdxX - halfX;
                    idxX = max(0, idxX);
                    idxX = min(idxX, imgWidth - 1);

                    float fxyz = img3d[getImageIdx(idxX, idxY, idxZ, imgWidth, imgHeight)];

                    for (int i = 0; i < NUM_COEFFICIENTS; i++)
                    {
                        sum[i] += weights[getWeightIdx(filterIdxX, filterIdxY, filterIdxZ, i, patchSize)] * fxyz;
                    }
                }
            }
        }

        for (int j = 1; j < NUM_COEFFICIENTS; j++)
        {
            float sumProduct = 0.f;
            for (int i = 0; i < NUM_COEFFICIENTS; i++)
            {
                sumProduct += invG[j * NUM_COEFFICIENTS + i] * sum[i];
            }
            polyCoefficients[getImageIdx(x, y, z, imgWidth, imgHeight) + (j - 1) * imgHeight * imgWidth * imgDepth] = sumProduct;
        }
    }
}

texture<fp_tex_float, cudaTextureType3D, cudaReadModeElementType> sourceTex;

__global__ void warpByFlowField(const float *__restrict__ flow3d,
                                float *__restrict__ interpolated,
                                int imgWidth,
                                int imgHeight,
                                int imgDepth,
                                float spacingX,
                                float spacingY,
                                float spacingZ)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int imgSize = imgWidth * imgHeight * imgDepth;

    if (x >= imgWidth || y >= imgHeight)
        return;

    for (int z = 0; z < imgDepth; z++)
    {
        int imgIdx = getImageIdx(x, y, z, imgWidth, imgHeight);
        float flowX = flow3d[imgIdx + 0 * imgSize];
        float flowY = flow3d[imgIdx + 1 * imgSize];
        float flowZ = flow3d[imgIdx + 2 * imgSize];

        float3 warpedPos = {static_cast<float>(x) * spacingX + flowX + 0.5f,
                            static_cast<float>(y) * spacingY + flowY + 0.5f,
                            static_cast<float>(z) * spacingZ + flowZ + 0.5f};
        interpolated[imgIdx] = tex3D(sourceTex, warpedPos.x, warpedPos.y, warpedPos.z);
    }
}

__global__ void FarnebackUpdateMatrices(float *__restrict__ R0,
                                        float *__restrict__ R1,
                                        float *__restrict__ flow,
                                        float *__restrict__ m,
                                        int imgWidth,
                                        int imgHeight,
                                        int imgDepth)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= imgWidth || y >= imgHeight)
        return;

    int imgSize = imgWidth * imgHeight * imgDepth;

    for (int z = 0; z < imgDepth; ++z)
    {
        int imgIdx = getImageIdx(x, y, z, imgWidth, imgHeight);

        float a[3 + 3];
        float b[3];

#pragma unroll
        for (int i = 0; i < 3 + 3 + 3; i++)
        {
            if (i < 3)
            {
                b[i] = 0.5f * (R0[i * imgSize + imgIdx] - R1[i * imgSize + imgIdx]);
            }
            else if (i < 6)
            {
                a[i - 3] = 0.5f * (R0[i * imgSize + imgIdx] + R1[i * imgSize + imgIdx]);
            }
            else
            {
                a[i - 3] = 0.25f * (R0[i * imgSize + imgIdx] + R1[i * imgSize + imgIdx]);
            }
        }
        float flowX = flow[imgIdx + 0 * imgSize];
        float flowY = flow[imgIdx + 1 * imgSize];
        float flowZ = flow[imgIdx + 2 * imgSize];

        // ~b = 0.5(b1-b2) + A.flow
        b[0] += flowX * a[0] + flowY * a[3] + flowZ * a[4];
        b[1] += flowX * a[3] + flowY * a[1] + flowZ * a[5];
        b[2] += flowX * a[4] + flowY * a[5] + flowZ * a[2];

        // generated by CAS (wxMaxima), see file A.A+A.b.wxmx
        // version for compressed format
        m[0 * imgSize + imgIdx] = a[0] * a[0] + a[3] * a[3] + a[4] * a[4];
        m[1 * imgSize + imgIdx] = a[1] * a[1] + a[3] * a[3] + a[5] * a[5];
        m[2 * imgSize + imgIdx] = a[2] * a[2] + a[4] * a[4] + a[5] * a[5];
        m[3 * imgSize + imgIdx] = a[4] * a[5] + a[1] * a[3] + a[0] * a[3];
        m[4 * imgSize + imgIdx] = a[3] * a[5] + a[2] * a[4] + a[0] * a[4];
        m[5 * imgSize + imgIdx] = a[2] * a[5] + a[1] * a[5] + a[3] * a[4];
        m[6 * imgSize + imgIdx] = a[4] * b[2] + a[3] * b[1] + a[0] * b[0];
        m[7 * imgSize + imgIdx] = a[5] * b[2] + a[1] * b[1] + a[3] * b[0];
        m[8 * imgSize + imgIdx] = a[2] * b[2] + a[5] * b[1] + a[4] * b[0];

        // matrix format (off-diagonal entries are saved twice, symmetry! )
        // float m0_ = a[0] * a[0] + a[3] * a[3] + a[4] * a[4];
        // float m1_ = a[1] * a[1] + a[3] * a[3] + a[5] * a[5];
        // float m2_ = a[2] * a[2] + a[4] * a[4] + a[5] * a[5];
        // float m3_ = a[4] * a[5] + a[1] * a[3] + a[0] * a[3];
        // float m4_ = a[3] * a[5] + a[2] * a[4] + a[0] * a[4];
        // float m5_ = a[2] * a[5] + a[1] * a[5] + a[3] * a[4];
        // float m6_ = a[4] * b[2] + a[3] * b[1] + a[0] * b[0];
        // float m7_ = a[5] * b[2] + a[1] * b[1] + a[3] * b[0];
        // float m8_ = a[2] * b[2] + a[5] * b[1] + a[4] * b[0];

        // m[0 * imgSize + imgIdx] = m0_;
        // m[1 * imgSize + imgIdx] = m3_;
        // m[2 * imgSize + imgIdx] = m4_;

        // m[3 * imgSize + imgIdx] = m3_;
        // m[4 * imgSize + imgIdx] = m1_;
        // m[5 * imgSize + imgIdx] = m5_;

        // m[6 * imgSize + imgIdx] = m4_;
        // m[7 * imgSize + imgIdx] = m5_;
        // m[8 * imgSize + imgIdx] = m2_;

        // m[9 * imgSize + imgIdx] = m6_;
        // m[10 * imgSize + imgIdx] = m7_;
        // m[11 * imgSize + imgIdx] = m8_;
    }
}

__global__ void solveEquationsCramer(float *__restrict__ M, float *__restrict__ flow3d, int imgWidth, int imgHeight, int imgDepth)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= imgWidth || y >= imgHeight)
        return;

    int imgSize = imgWidth * imgHeight * imgDepth;

    for (int z = 0; z < imgDepth; ++z)
    {
        int imgIdx = getImageIdx(x, y, z, imgWidth, imgHeight);
        float m[6]; // symetric 3x3 matrix, i.e. independent 6 entries
        float h[3]; // right side

        for (size_t i = 0; i < 9; i++)
        {
            if (i < 6)
            {
                m[i] = M[imgIdx + imgSize * i];
            }
            else
            {
                h[i - 6] = M[imgIdx + imgSize * i];
            }
        }
        float a0 = m[0];
        float a1 = m[1];
        float a2 = m[2];
        float a3 = m[3];
        float a4 = m[4];
        float a5 = m[5];
        float b0 = h[0];
        float b1 = h[1];
        float b2 = h[2];

        float det = a0 * a5 * a5 - 2 * a3 * a4 * a5 + a1 * a4 * a4 + a2 * a3 * a3 - a0 * a1 * a2;
        float invDet = 1.f / (det);
        float flowX = -invDet * ((a1 * a2 - a5 * a5) * b0 + (a4 * a5 - a2 * a3) * b1 + (a3 * a5 - a1 * a4) * b2);
        float flowY = invDet * ((a2 * a3 - a4 * a5) * b0 + (a4 * a4 - a0 * a2) * b1 + (a0 * a5 - a3 * a4) * b2);
        float flowZ = invDet * ((a1 * a4 - a3 * a5) * b0 + (a0 * a5 - a3 * a4) * b1 + (a3 * a3 - a0 * a1) * b2);

        if (fabsf(det) < 1e-2)
        {
            flowX = flowY = flowZ = 0.f;
        }
        flow3d[imgIdx + 0 * imgSize] = flowX;
        flow3d[imgIdx + 1 * imgSize] = flowY;
        flow3d[imgIdx + 2 * imgSize] = flowZ;
    }
}
// texture<fp_tex_float, cudaTextureType3D, cudaReadModeElementType> R1;

// __global__ void FarnebackUpdateMatrices(float *__restrict__ R0,
//                                         float *__restrict__ flow,
//                                         float *__restrict__ M,
//                                         int imgWidth,
//                                         int imgHeight,
//                                         int imgDepth)
// {

//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;

//     if (x >= imgWidth || y >= imgHeight)
//         return;

//     int imgSize = imgWidth * imgHeight * imgDepth;

//     for (int z = 0; z < imgDepth; ++z)
//     {
//         int imgIdx = getImageIdx(x,y,z,imgWidth,imgHeight);

//         float a[3*3+3];
//         float3 flowVec = { flow[ 0 * imgSize + imgIdx ], flow[ 1 * imgSize + imgIdx ], flow[ 2 * imgSize + imgIdx ] };

//         for(int i = 0; i < 3*3+3; i++)
//         {
//             a[i] = 0.5f * ( R0[i*imgSize + getImageIdx(x,y,z,imgWidth,imgHeight]) + tex3d<float>(R1,x+flowVec.x, y+flowVec.y, z+flowVec.z) );
//         }
//     }
// }
