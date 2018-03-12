
__device__ inline int getImageIdx(int x, int y, int z, int imgWidth, int imgHeight)
{
    return z * imgHeight * imgWidth + y * imgWidth + x;
}

__device__ float normalPdf(float x, float sigmaSquared)
{
    return __expf(-0.5 * x * x / sigmaSquared);
}

__device__ float getCoefficientGauss3d(float3 x, float3 sigma)
{

    float coefficient = normalPdf(x.x, sigma.x);
    coefficient *= normalPdf(x.y, sigma.y);
    coefficient *= normalPdf(x.z, sigma.z);

    return coefficient;
}

__device__ float getCoefficientGauss2d(float2 x, float2 sigma)
{

    float coefficient = normalPdf(x.x, sigma.x);
    coefficient *= normalPdf(x.y, sigma.y);

    return coefficient;
}

__global__ void convolve3d_gauss(float *__restrict__ in,
                                 float *__restrict__ out,
                                 int imgWidth,
                                 int imgHeight,
                                 int imgDepth,
                                 int filterWidth,
                                 int filterHeight,
                                 int filterDepth,
                                 float sigmaX,
                                 float sigmaY,
                                 float sigmaZ,
                                 bool filterNonZerosOnly)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int halfX = filterWidth / 2;
    int halfY = filterHeight / 2;
    int halfZ = filterDepth / 2;

    if (x >= imgWidth || y >= imgHeight)
        return;

    for (int z = 0; z < imgDepth; z++)
    {

        float sum = 0.f;
        float sumCoefficients = 0.f;

        int imgIdx = z * imgHeight * imgWidth + y * imgWidth + x;

        if (filterNonZerosOnly &&
            in[imgIdx] <= 1e-6)
        {
            out[imgIdx] = in[imgIdx];
            continue;
        }

        for (int filterIdxZ = 0; filterIdxZ < filterDepth; filterIdxZ++)
        {
            int idxZ = z + filterIdxZ - halfZ;
            if (idxZ < 0 || idxZ >= imgDepth)
                continue;

            // int idxZ = z;
            for (int filterIdxY = 0; filterIdxY < filterHeight; filterIdxY++)
            {
                int idxY = y + filterIdxY - halfY;
                if (idxY < 0 || idxY >= imgHeight)
                    continue;
                for (int filterIdxX = 0; filterIdxX < filterWidth; filterIdxX++)
                {
                    int idxX = x + filterIdxX - halfX;
                    if (idxX < 0 || idxX >= imgWidth)
                        continue;

                    float3 delta{static_cast<float>(idxX - x), static_cast<float>(idxY - y), static_cast<float>(idxZ - z)};
                    float3 sigma{sigmaX, sigmaY, sigmaZ};
                    float coefficient = getCoefficientGauss3d(delta, sigma);
                    float inVal = in[idxZ * imgHeight * imgWidth + idxY * imgWidth + idxX];

                    if (!filterNonZerosOnly || inVal > 0.f)
                    {
                        sum += inVal * coefficient;
                        sumCoefficients += coefficient;
                    }
                }
            }
        }
        float outVal = sumCoefficients > 1e-6 ? sum / sumCoefficients : 0.f;
        out[imgIdx] = outVal;
    }
}

__global__ void convolve3d_gauss_with_mask(float *__restrict__ in,
                                           float *__restrict__ out,
                                           float *__restrict__ filterMask,
                                           int imgWidth,
                                           int imgHeight,
                                           int imgDepth,
                                           int filterWidth,
                                           int filterHeight,
                                           int filterDepth,
                                           float sigmaX,
                                           float sigmaY,
                                           float sigmaZ,
                                           bool filterNonZerosOnly)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int halfX = filterWidth / 2;
    int halfY = filterHeight / 2;
    int halfZ = filterDepth / 2;

    if (x >= imgWidth || y >= imgHeight)
        return;

    for (int z = 0; z < imgDepth; z++)
    {

        float sum = 0.f;
        float sumCoefficients = 0.f;

        int imgIdx = z * imgHeight * imgWidth + y * imgWidth + x;

        if ((filterMask && filterMask[imgIdx] == 0.f) || (filterNonZerosOnly &&
                                                          in[imgIdx] <= 1e-6))
        {
            out[imgIdx] = in[imgIdx];
            continue;
        }

        for (int filterIdxZ = 0; filterIdxZ < filterDepth; filterIdxZ++)
        {
            int idxZ = z + filterIdxZ - halfZ;
            if (idxZ < 0 || idxZ >= imgDepth)
                continue;

            // int idxZ = z;
            for (int filterIdxY = 0; filterIdxY < filterHeight; filterIdxY++)
            {
                int idxY = y + filterIdxY - halfY;
                if (idxY < 0 || idxY >= imgHeight)
                    continue;
                for (int filterIdxX = 0; filterIdxX < filterWidth; filterIdxX++)
                {
                    int idxX = x + filterIdxX - halfX;
                    if (idxX < 0 || idxX >= imgWidth)
                        continue;

                    float3 delta{static_cast<float>(idxX - x), static_cast<float>(idxY - y), static_cast<float>(idxZ - z)};
                    float3 sigma{sigmaX, sigmaY, sigmaZ};
                    float coefficient = getCoefficientGauss3d(delta, sigma);
                    float inVal = in[idxZ * imgHeight * imgWidth + idxY * imgWidth + idxX];

                    if (!filterNonZerosOnly || inVal > 0.f)
                    {
                        sum += inVal * coefficient;
                        sumCoefficients += coefficient;
                    }
                }
            }
        }
        float outVal = sumCoefficients > 1e-6 ? sum / sumCoefficients : 0.f;
        out[imgIdx] = outVal;
    }
}

__global__ void convolve2d_gauss(float *__restrict__ in,
                                 float *__restrict__ out,
                                 int imgWidth,
                                 int imgHeight,
                                 int filterWidth,
                                 int filterHeight,
                                 float sigmaX,
                                 float sigmaY)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int halfX = filterWidth / 2;
    int halfY = filterHeight / 2;

    if (x >= imgWidth || y >= imgHeight)
        return;

    float sum = 0.f;
    float sumCoefficients = 0.f;

    for (int filterIdxY = 0; filterIdxY < filterHeight; filterIdxY++)
    {
        int idxY = y + filterIdxY - halfY;
        if (idxY < 0 || idxY >= imgHeight)
            continue;
        for (int filterIdxX = 0; filterIdxX < filterWidth; filterIdxX++)
        {
            int idxX = x + filterIdxX - halfX;
            if (idxX < 0 || idxX >= imgWidth)
                continue;

            float2 delta{static_cast<float>(idxX - x), static_cast<float>(idxY - y)};
            float2 sigma{sigmaX, sigmaY};
            float coefficient = getCoefficientGauss2d(delta, sigma);
            float inVal = in[idxY * imgWidth + idxX];
            sum += inVal * coefficient;
            sumCoefficients += coefficient;
        }
    }
    float outVal = sum / sumCoefficients;
    out[y * imgWidth + x] = outVal;
}
