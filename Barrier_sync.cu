#include<stdio.h>



__global__ void kernel(int *d_o,int*d_i)
{
    int index=threadIdx.x+blockIdx.x*blockDim.x;
    if(index<10)
    {
        int temp=d_i[index+1];
        __syncthreads();
        d_o[index]=temp;
        __syncthreads();
    }
}


int main()
{
    const int N=10;
    int h_i[N];
    int h_o[N];
    for(int i=0;i<N;i++)
    {
        h_i[i]=i;
    }

    int *d_i;
    int *d_o;
    const int size=N*sizeof(int);

    cudaMalloc((void **) &d_i,size);
    cudaMalloc((void **) &d_o,size);

    cudaMemcpy(d_i,h_i,size,cudaMemcpyHostToDevice);
    kernel<<<1,N>>>(d_o,d_i);
    cudaMemcpy(h_o,d_o,size,cudaMemcpyDeviceToHost);

    for (int i=0;i<N;i++)
    {
        printf("%d",h_o[i]);
        printf("\t");
        printf("%d",h_i[i]);
        printf("\n");
    }

}