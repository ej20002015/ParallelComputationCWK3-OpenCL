__kernel
void performHeatEquation(__global float* sourceGrid, int n, __global float* outputGrid)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    bool edgeCell = x == 0 || y == 0 || x == n - 1 || y == n - 1;

    if (!edgeCell)
    {
        float bottom = sourceGrid[(y + 1) * n + x];
        float top =    sourceGrid[(y - 1) * n + x];
        float right =  sourceGrid[y * n + (x + 1)];
        float left =   sourceGrid[y * n + (x - 1)];
        outputGrid[y * n + x] = (bottom + top + right + left) / 4.0f;
    }
}