#include <iostream>
using std::cout;
using std::endl;

// kernel declaration

__global__ void multiply(float *d_out,float *d_a,float *d_b) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float f = d_a[idx];
    float g = d_b[idx];
    d_out[idx] = f*g;
}

// driver code
int main()
{
    const int ARRAY_SIZE = 64;              // Always use size in multples of 32 as threads are also in each block in multiples of 32.
    const int ARRAY_BYTE =  ARRAY_SIZE * sizeof(float);

    // Host variables (CPU)
    
    float h_a[ARRAY_SIZE];
    float h_b[ARRAY_SIZE];
    float h_out[ARRAY_SIZE];

    for(int i=0;i<ARRAY_SIZE;i++)
    {
        h_a[i] = float(2*i);
        h_b[i] = float(4*i);
    }    

    // Device variables (GPU)

    float *d_a;
    float *d_b;
    float *d_out;

    // Device variables initialisation

    cudaMalloc((void **) &d_a,ARRAY_BYTE);
    cudaMalloc((void **) &d_b,ARRAY_BYTE);
    cudaMalloc((void **) &d_out,ARRAY_BYTE);

    // Copying Host variable values to Device variables

    cudaMemcpy(d_a,h_a,ARRAY_BYTE,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,ARRAY_BYTE,cudaMemcpyHostToDevice);

    // Launching kernel

    multiply<<<2,32>>> (d_out,d_a,d_b); // 2 blocks in use where 32 threads of each block are used from the Grid, as 32*2 = 64 (ARRAY_SIZE)  

    // Copying output device variable value to host variable.

    cudaMemcpy(h_out,d_out,ARRAY_BYTE,cudaMemcpyDeviceToHost);

    // Releasing GPU memory after excecution.

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    // Printing

    for(int i = 0;i<ARRAY_SIZE;i++)
    {
        cout<<h_out[i];
        cout<<endl;
    }
}

/* 
                    CONSOLE_OUTPUT
0
8
32
72
128
200
288
392
512
648
800
968
1152
1352
1568
1800
2048
2312
2592
2888
3200
3528
3872
4232
4608
5000
5408
5832
6272
6728
7200
7688
8192
8712
9248
9800
10368
10952
11552
12168
12800
13448
14112
14792
15488
16200
16928
17672
18432
19208
20000
20808
21632
22472
23328
24200
25088
25992
26912
27848
28800
29768
30752
31752
*/