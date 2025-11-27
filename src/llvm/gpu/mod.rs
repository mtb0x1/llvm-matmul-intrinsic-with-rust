pub(crate) mod compiled;
pub(crate) mod cuda_driver;
pub(crate) mod jit;
pub use compiled::ll_matmul as ll_matmul_gpu_compiled;
pub use jit::ll_matmul as ll_matmul_gpu_jit;

use cuda_driver::{CUcontext, CUdevice, check_cuda_error, cuCtxCreate_v2, cuDeviceGet, cuInit};
use std::ptr;
use std::sync::OnceLock;

pub struct GpuContext {
    pub jitcontext: CUcontext,
    pub compiledcontext: CUcontext,
}

unsafe impl Send for GpuContext {}
unsafe impl Sync for GpuContext {}

pub static GPU_CONTEXT: OnceLock<GpuContext> = OnceLock::new();

impl GpuContext {
    pub fn new() -> Self {
        unsafe {
            check_cuda_error(cuInit(0), "cuInit");
            let mut device: CUdevice = 0;
            check_cuda_error(cuDeviceGet(&mut device, 0), "cuDeviceGet");
            let mut jitcontext: CUcontext = ptr::null_mut();
            check_cuda_error(cuCtxCreate_v2(&mut jitcontext, 0, device), "cuCtxCreate");
            let mut compiledcontext: CUcontext = ptr::null_mut();
            check_cuda_error(cuCtxCreate_v2(&mut compiledcontext, 0, device), "cuCtxCreate");
            Self { jitcontext, compiledcontext }
        }
    }
}

impl Default for GpuContext {
    fn default() -> Self {
        Self::new()
    }
}
