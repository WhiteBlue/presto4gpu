// Computes the Mandelbrot Set to N Iterations


typedef struct ffdotpows{
    long long rlo;      /* Lowest Fourier freq present */
    int zlo;            /* Lowest Fourier f-dot present */
    int wlo;            /* Lowest Fourier f-dot-dot present */
    int numrs;          /* Number of Fourier freqs present */
    int numzs;          /* Number of Fourier f-dots present */
    int numws;          /* Number of Fourier f-dot-dots present */
    float ***powers;     /* 3D Matrix of the powers */
    unsigned short *rinds; /* Table of lookup indices for Fourier Freqs */
    unsigned short *zinds; /* Table of lookup indices for Fourier f-dots */
} ffdotpows;



__kernel void solve_mandelbrot(__global float const * real,
                               __global float const * imag,
                               int iterations,
                               __global int * result)
{
    // Get Parallel Index
    unsigned int i = get_global_id(0);

    float x = real[i]; // Real Component
    float y = imag[i]; // Imaginary Component
    int   n = 0;       // Tracks Color Information

    // Compute the Mandelbrot Set
    while ((x * x + y * y <= 2 * 2) && n < iterations)
    {
        float xtemp = x * x - y * y + real[i];
        y = 2
        * x * y + imag[i];
        x = xtemp;
        n++;
    }

    // Write Results to Output Arrays
    result[i] = x * x + y * y <= 2 * 2 ? -1 : n;
}