/*
Both kernels perform the same operation. The only difference is that the upper one reads the source grid
from constant memory, whereas the latter reads it from global memory. If the source grid is greater than
the maximum constant buffer size of the device, then the bottom kernel is called, otherwise the top one.
Constant memory is quicker to access so is the preffered option.
*/

__kernel
void performHeatEquationConstant(__constant float* sourceGrid, int n, __global float* outputGrid)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    bool edgeCell = (x == 0 || y == 0 || x == n - 1 || y == n - 1);

    if (edgeCell)
        outputGrid[y * n + x] = 0.0f;
    else
    {
        float bottom = sourceGrid[(y + 1) * n + x];
        float top =    sourceGrid[(y - 1) * n + x];
        float right =  sourceGrid[y * n + (x + 1)];
        float left =   sourceGrid[y * n + (x - 1)];
        outputGrid[y * n + x] = (bottom + top + right + left) / 4.0f;
    }
}

__kernel
void performHeatEquationGlobal(__global float* sourceGrid, int n, __global float* outputGrid)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    bool edgeCell = (x == 0 || y == 0 || x == n - 1 || y == n - 1);

    if (edgeCell)
        outputGrid[y * n + x] = 0.0f;
    else
    {
        float bottom = sourceGrid[(y + 1) * n + x];
        float top =    sourceGrid[(y - 1) * n + x];
        float right =  sourceGrid[y * n + (x + 1)];
        float left =   sourceGrid[y * n + (x - 1)];
        outputGrid[y * n + x] = (bottom + top + right + left) / 4.0f;
    }
}