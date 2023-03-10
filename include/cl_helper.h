#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

cl_device_id create_device();
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename);
