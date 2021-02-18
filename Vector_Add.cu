#include<stdio.h>

__global__ void vecAdd(int *c_d,int *a_d,int *b_d)
{
    int idx=threadIdx.x;
    c_d[idx]=a_d[idx]+b_d[idx];
}



int main()
{
    const int N=12;
    int a_h[N],b_h[N],c_h[N];

    for(int i=0;i<12;i++)
    {
        a_h[i]=i;
        b_h[i]=i*2;
    }
    //initialize gpu pointer
    int *a_d,*b_d,*c_d;

    const int size=N*sizeof(int);
    //allocate memory on gpu
    cudaMalloc((void **) &a_d,size);
    cudaMalloc((void **) &b_d,size);
    cudaMalloc((void **) &c_d,size);

    cudaMemcpy(a_d,a_h,size,cudaMemcpyHostToDevice);
    cudaMemcpy(b_d,b_h,size,cudaMemcpyHostToDevice);

    //call the kernal 

    vecAdd<<<1,N>>>(c_d,a_d,b_d);

    cudaMemcpy(c_h,c_d,size,cudaMemcpyDeviceToHost);
    for(int i=0;i<N;i++)
    {
        printf("%d",c_h[i]);
        printf("\n");
    }

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

}
//output console
/*
0
3
6
9
12
15
18
21
24
27
30
33
*/