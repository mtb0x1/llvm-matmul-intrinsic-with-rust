/// CUDA Driver API bindings
/// this is a rip off from docs
/// https://docs.nvidia.com/cuda/cuda-driver-api/index.html
/// cuda version: 13.0 (2025-11-27)
/// use at your own risk
use std::ffi::c_void;
use std::os::raw::{c_char, c_int, c_uint};

pub type CUdevice = c_int;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
// Using *mut c_void for simplicity, though it's technically a u64 or pointer depending on arch
pub type CUdeviceptr = *mut c_void;

#[repr(C)]
#[allow(non_camel_case_types)]
#[derive(PartialEq)]
pub enum CUresult {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_NOT_INITIALIZED = 1,
}

#[link(name = "cuda")]
unsafe extern "C" {
    pub fn cuInit(flags: c_uint) -> CUresult;
    pub fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;
    pub fn cuDeviceGetCount(count: *mut c_int) -> CUresult;
    pub fn cuDeviceGetName(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult;
    pub fn cuCtxCreate_v2(pctx: *mut CUcontext, flags: c_uint, dev: CUdevice) -> CUresult;
    pub fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult;
    pub fn cuModuleGetFunction(
        hfunc: *mut CUfunction,
        hmod: CUmodule,
        name: *const c_char,
    ) -> CUresult;
    pub fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;
    pub fn cuMemcpyHtoD_v2(
        dstDevice: CUdeviceptr,
        srcHost: *const c_void,
        ByteCount: usize,
    ) -> CUresult;
    pub fn cuMemcpyDtoH_v2(
        dstHost: *mut c_void,
        srcDevice: CUdeviceptr,
        ByteCount: usize,
    ) -> CUresult;
    pub fn cuLaunchKernel(
        f: CUfunction,
        gridDimX: c_uint,
        gridDimY: c_uint,
        gridDimZ: c_uint,
        blockDimX: c_uint,
        blockDimY: c_uint,
        blockDimZ: c_uint,
        sharedMemBytes: c_uint,
        hStream: *mut c_void,
        kernelParams: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult;
    pub fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;
    pub fn cuCtxDestroy_v2(ctx: CUcontext) -> CUresult;
    pub fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;
}

pub fn check_cuda_error(result: CUresult, msg: &str) {
    if result == CUresult::CUDA_SUCCESS {
        return;
    }
    panic!("CUDA Error: {} (code: {:?})", msg, result as i32);
}
