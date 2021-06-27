
#include <iostream>
#include <chrono>
#include <stdlib.h>
#include <stdio.h>

#define DIM 512

#include "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Versions/Current/Headers/cblas.h"

#include "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Versions/Current/Headers/BNNS/bnns.h"

using namespace std::chrono;
using namespace std;




void cblasTest() {
    size_t size = DIM*DIM*sizeof(float);
    float *matA = (float*) malloc(size);
    float *matB = (float*) malloc(size);
    float *matC = (float*) malloc(size);

    for (int i=0;i<DIM*DIM;i++)
    {
       matA[i] = 1.0f;
       matB[i] = 0.5f;
       
    }

    double calculations = ((double)DIM)*DIM*DIM*2;

    for(int l=0;l<100;l++)
    {
      auto start = high_resolution_clock::now();
      cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                DIM, DIM, DIM, 1.0f, matA, DIM,
                matB, DIM, 0.0f, matC,
                DIM);


      auto end = high_resolution_clock::now();
    
      auto duration = duration_cast<microseconds>(end - start);
      double seconds = duration.count() / 1000000.0;
      double flops = calculations / seconds;
      cout << "microseconds: " << duration.count() << ", GFlops: " << flops / 1000000000.0 << endl;
    }
    
    printf("hello apple, %f\n", matC[0]);

}

void BNNSTestConv()
{
    const int N = 256;
    const int C = 384;
    const int H = 8;
    const int W = 8;
    const int F = 3;
    
    size_t size = N*C*H*W*sizeof(float);
    size_t filterSize = C*C*F*F*sizeof(float);
    
    float *input = (float *) malloc(size);
    float *output = (float *) malloc(size);
    float *filter = (float *) malloc(filterSize);
    
    BNNSNDArrayDescriptor aDesc = {};
    aDesc.data = input;
    aDesc.data_scale = 1.0f;
    aDesc.data_type = BNNSDataTypeFloat32;
    
    aDesc.layout = BNNSDataLayoutImageCHW;
    aDesc.size[0] = W;
    aDesc.size[1] = H;
    aDesc.size[2] = C;
    
    aDesc.stride[0] = 1;
    aDesc.stride[1] = W;
    aDesc.stride[2] = W*H;

    
    BNNSNDArrayDescriptor bDesc = aDesc;
    bDesc.data = filter;
    bDesc.layout = BNNSDataLayoutConvolutionWeightsOIHW;
    bDesc.size[0] = F;
    bDesc.size[1] = F;
    bDesc.size[2] = C;
    bDesc.size[3] = C;

    bDesc.stride[0] = 1;
    bDesc.stride[1] = F;
    bDesc.stride[2] = F*F;
    bDesc.stride[3] = F*F*C;
    
    BNNSNDArrayDescriptor outDesc = aDesc;
    outDesc.data = output;
    
    BNNSNDArrayDescriptor biasDesc = bDesc;
    biasDesc.layout = BNNSDataLayoutVector;
    biasDesc.size[0] = C;
    biasDesc.stride[0] = 1;

    BNNSLayerParametersConvolution params = {};
    params.i_desc = aDesc;
    params.o_desc = outDesc;
    params.w_desc = bDesc;
    params.groups = 1;
    params.x_padding = 2;
    params.y_padding = 2;
    params.x_stride = 1;
    params.y_stride = 1;
    params.x_dilation_stride = 1;
    params.y_dilation_stride = 1;
    params.activation.function = BNNSActivationFunctionIdentity;
    params.bias = biasDesc;
    
    
    BNNSFilterParameters filterParams = {};
    //filterParams.flags = BNNSFlagsUseClientPtr;
    
    BNNSFilter f = BNNSFilterCreateLayerConvolution(&params, &filterParams);
    
    double calculations = ((double)N)*C*H*W*C*9*2;

    for(int l=0;l<100;l++)
    {
        auto start = high_resolution_clock::now();
        int ret = BNNSFilterApplyBatch(f, N, input, C*H*W, output, C*H*W);
        
        if (ret != 0)
        {
            printf("\nError code %d\n", ret);
            exit(0);
        }

        auto end = high_resolution_clock::now();
      
        auto duration = duration_cast<microseconds>(end - start);
        double seconds = duration.count() / 1000000.0;
        double flops = calculations / seconds;
        cout << "microseconds: " << duration.count() << ", GFlops: " << flops / 1000000000.0 << endl;
    }

    
    printf("\nConv Done!\n");
}

void BNNSTestGemmBatched()
{
    const int M = 256*4;
    const int N = 384;
    const int K = 384;
    const int batch = 36;
    
    size_t matSize = batch * M * N * sizeof(float);
    size_t weightSize = batch * N * K * sizeof(float);
    float *input = (float*) malloc(matSize);
    float *output = (float*) malloc(matSize);
    float *weight = (float*) malloc(weightSize);
    
    for (int i=0;i<batch * M * N;i++)
    {
       input[i] = 1.0f;
    }

    for (int i=0;i<batch * K * N;i++)
    {
       weight[i] = 0.5f;
    }

    
    BNNSNDArrayDescriptor aDesc = {};
    aDesc.data = input;
    aDesc.data_scale = 1.0f;
    aDesc.data_type = BNNSDataTypeFloat32;
    aDesc.layout = BNNSDataLayout3DFirstMajor;
    aDesc.size[0] = batch;
    aDesc.size[1] = M;
    aDesc.size[2] = K;
    aDesc.stride[0] = 1;
    aDesc.stride[1] = K;
    aDesc.stride[2] = K*M;

    
    BNNSNDArrayDescriptor bDesc = aDesc;
    bDesc.data = weight;
    bDesc.size[0] = batch;
    bDesc.size[1] = K;
    bDesc.size[2] = N;
    bDesc.stride[0] = 1;
    bDesc.stride[1] = N;
    bDesc.stride[2] = N * K;
    BNNSNDArrayDescriptor cDesc = aDesc;
    cDesc.data = output;
    
    double calculations = ((double)batch) * M * N * K * 2;
    
    for(int l=0;l<100;l++)
    {
        auto start = high_resolution_clock::now();
        
        BNNSDirectApplyBroadcastMatMul(false,
                                       false,
                                       1.0f,
                                       &aDesc,
                                       &bDesc,
                                       &cDesc,
                                       nullptr);

        auto end = high_resolution_clock::now();
      
        auto duration = duration_cast<microseconds>(end - start);
        double seconds = duration.count() / 1000000.0;
        double flops = calculations / seconds;
        cout << "microseconds: " << duration.count() << ", GFlops: " << flops / 1000000000.0 << endl;
    }
    
    printf("hello batched bnns, %f\n", output[0]);
}

void BNNSTest()
{
    size_t size = DIM*DIM*sizeof(float);
    float *matA = (float*) malloc(size);
    float *matB = (float*) malloc(size);
    float *matC = (float*) malloc(size);

    for (int i=0;i<DIM*DIM;i++)
    {
       matA[i] = 1.0f;
       matB[i] = 0.5f;
       
    }

    double calculations = ((double)DIM)*DIM*DIM*2;

    BNNSNDArrayDescriptor aDesc = {};
    aDesc.data = matA;
    aDesc.data_scale = 1.0f;
    aDesc.data_type = BNNSDataTypeFloat32;
    aDesc.layout = BNNSDataLayoutRowMajorMatrix;
    aDesc.size[0] = aDesc.size[1] = DIM;
    aDesc.stride[0] = 1; aDesc.stride[1]=DIM;
    
    BNNSNDArrayDescriptor bDesc = aDesc;
    BNNSNDArrayDescriptor cDesc = aDesc;
    bDesc.data = matB;
    cDesc.data = matC;

    for(int l=0;l<100;l++)
    {
        auto start = high_resolution_clock::now();
        
        BNNSDirectApplyBroadcastMatMul(false,
                                       false,
                                       1.0f,
                                       &aDesc,
                                       &bDesc,
                                       &cDesc,
                                       nullptr);

        auto end = high_resolution_clock::now();
      
        auto duration = duration_cast<microseconds>(end - start);
        double seconds = duration.count() / 1000000.0;
        double flops = calculations / seconds;
        cout << "microseconds: " << duration.count() << ", GFlops: " << flops / 1000000000.0 << endl;
    }
    
    printf("hello bnns, %f\n", matC[0]);
}

int main()
{
    cblasTest();
    BNNSTest();
    BNNSTestConv();
    BNNSTestGemmBatched();
    return 0;
}

