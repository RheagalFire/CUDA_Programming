#include<stdio.h>
//gpu code
__global__ void square(float * d_out,float *d_in) // arguments are pointers to input and the return is a value to the pointer in the argument list
{
    int idx=threadIdx.x;
    float f= d_in[idx];
    d_out[idx]=f*f;
}
//cpu code
int main()
{
    //generate input array
    const int ARRAY_SIZE=64;
    const int ARRAY_BYTES=ARRAY_SIZE * sizeof(float);
    float h_in[ARRAY_SIZE];
    for (int i=0;i<64;i++)
    {
        h_in[i]=float(i);
    }
    float h_out[ARRAY_SIZE];
    //declare gpu pointers
    float * d_in;
    float * d_out;
    //allocate gpu memory
    //The cudamalloc function returns an integer(instead of pointer) as error code to the memory block
    //All CUDA API function follows the convention of returning an integer error code 
    //when cudamalloc is called, a local variable named d_array is created and assigned with the value of the first function argument.
    //There is no way we can retrieve the value in that local variable outside the function's scope. That why we need to a pointer to a pointer here.
    cudaMalloc((void **) &d_in,ARRAY_BYTES);
    cudaMalloc((void **) &d_out,ARRAY_BYTES);
    //Copy the input array from cpu to gpu
    cudaMemcpy(d_in,h_in,ARRAY_BYTES,cudaMemcpyHostToDevice);
    //Launch The Kernal
    square<<<1,ARRAY_SIZE>>>(d_out,d_in); // launching on one Block with 64 elements 
    //copy back the result array to the CPU
    cudaMemcpy(h_out,d_out,ARRAY_BYTES,cudaMemcpyDeviceToHost);
    //print
    for (int i=0;i<64;i++)
    {
        printf("%f",h_out[i]);
        printf(((i%4)!=3)?"\t":"\n");
    }
    //free the gpu memory 
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;


}
// output console
/*
256.000000      289.000000      324.000000      361.000000
400.000000      441.000000      484.000000      529.000000
576.000000      625.000000      676.000000      729.000000
784.000000      841.000000      900.000000      961.000000
1024.000000     1089.000000     1156.000000     1225.000000
1296.000000     1369.000000     1444.000000     1521.000000
1600.000000     1681.000000     1764.000000     1849.000000
1936.000000     2025.000000     2116.000000     2209.000000
2304.000000     2401.000000     2500.000000     2601.000000
2704.000000     2809.000000     2916.000000     3025.000000
3136.000000     3249.000000     3364.000000     3481.000000
3600.000000     3721.000000     3844.000000     3969.000000
*/