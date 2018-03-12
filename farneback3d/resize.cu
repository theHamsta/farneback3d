texture<float, cudaTextureType3D, cudaReadModeElementType> sourceTex;

__device__ inline int getImageIdx(int x, int y, int z, int imgWidth, int imgHeight)
{
    return z * imgHeight * imgWidth + y * imgWidth + x;
}

__global__ void resize(float *__restrict__ dstImage,
                       int srcWidth,
                       int srcHeight,
                       int srcDepth,
                       int dstWidth,
                       int dstHeight,
                       int dstDepth)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dstWidth || y >= dstHeight)
        return;

    float inverseScalingX = (float)srcWidth / dstWidth;
    float inverseScalingY = (float)srcHeight / dstHeight;
    float inverseScalingZ = (float)srcDepth / dstDepth;

    for (int z = 0; z < dstDepth; z++)
    {
        int imgIdx = getImageIdx(x, y, z, dstWidth, dstHeight);

        float3 sampledPointInSrc = {static_cast<float>(x) * inverseScalingX + 0.5f,
                                    static_cast<float>(y) * inverseScalingY + 0.5f,
                                    static_cast<float>(z) * inverseScalingZ + 0.5f};
        dstImage[imgIdx] = tex3D(sourceTex, sampledPointInSrc.x, sampledPointInSrc.y, sampledPointInSrc.z);
    }
}