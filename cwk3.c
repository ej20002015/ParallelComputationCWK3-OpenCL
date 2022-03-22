//
// Starting point for the OpenCL coursework for COMP/XJCO3221 Parallel Computation.
//
// Once compiled, execute with the size of the square grid as a command line argument, i.e.
//
// ./cwk3 16
//
// will generate a 16 by 16 grid. The C-code below will then display the initial grid,
// followed by the same grid again. You will need to implement OpenCL that applies the heat
// equation as per the instructions, so that the final grid is different.
//


//
// Includes.
//
#include <stdio.h>
#include <stdlib.h>

// For this coursework, the helper file has 2 routines in addition to simpleOpenContext_GPU() and compileKernelFromFile():
// getCmdLineArg()  :  Parses grid size N from command line argument, or fails with error message.
// fillGrid()       :  Fills the grid with random values, except boundary values which are always zero.
// Do not alter these routines, as they will be replaced with different versions for assessment.
#include "helper_cwk.h"


//
// Main.
//
int main( int argc, char **argv )
{
 
    //
    // Parse command line argument and check it is valid. Handled by a routine in the helper file.
    //
    int N;
    getCmdLineArg( argc, argv, &N );

    //
    // Initialisation.
    //

    // Set up OpenCL using the routines provided in helper_cwk.h.
    cl_device_id device;
    cl_context context = simpleOpenContext_GPU(&device);

    // Open up a single command queue, with the profiling option off (third argument = 0).
    cl_int status;
    cl_command_queue queue = clCreateCommandQueue( context, device, 0, &status );

    // Allocate memory for the grid. For simplicity, this uses a one-dimensional array.
	float *hostGrid = (float*) malloc( N * N * sizeof(float) );

	// Fill the grid with some initial values, and display to stdout. fillGrid() is defined in the helper file.
    fillGrid( hostGrid, N );
    printf( "Original grid (only top-left shown if too large):\n" );
    displayGrid( hostGrid, N );

	//
	// Allocate memory for the grid on the GPU and apply the heat equation as per the instructions.
	//

    // Create space for data on the GPU and copy over the source grid

    cl_mem device_sourceGrid = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * N * sizeof(float), hostGrid, &status);
    cl_mem device_outputGrid = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * N * sizeof(float), NULL, &status);

    // Compile kernel

    cl_kernel kernel = compileKernelFromFile("cwk3.cl", "performHeatEquation", context, device);

    // Specify kernel arguments

    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &device_sourceGrid);
    status = clSetKernelArg(kernel, 1, sizeof(int), &N);
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &device_outputGrid);

    // Set global and local work size, and then queue the kernel to be run

    const size_t globalWorkSize[2] = { N, N };
    // Let OpenCL decide on a suitable work group size. If a value
    // is hard coded then it restricts the values of N that can be provided
    const size_t* localWorkSize = NULL;
    
    status = clEnqueueNDRangeKernel(queue, kernel, 2, 0, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if (status != CL_SUCCESS)
    {
        printf("ERROR: Couldn't queue the kernel - error code = %d\n", status);
        return EXIT_FAILURE;
    }

    // Get the resulting grid back from the GPU

    status = clEnqueueReadBuffer(queue, device_outputGrid, CL_TRUE, 0, N * N * sizeof(float), hostGrid, 0, NULL, NULL);
    
    //
    // Display the final result. This assumes that the iterated grid was copied back to the hostGrid array.
    //
    printf( "Final grid (only top-left shown if too large):\n" );
    displayGrid( hostGrid, N );

    //
    // Release all resources.
    //

    clReleaseMemObject(device_sourceGrid);
	clReleaseMemObject(device_outputGrid);

    clReleaseCommandQueue( queue   );
    clReleaseContext     ( context );

    free( hostGrid );

    return EXIT_SUCCESS;
}
