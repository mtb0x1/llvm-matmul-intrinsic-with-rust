use crate::common::GPU_FATBIN_PAYLOAD;
use crate::llvm::gpu::cuda_driver::*;
use crate::llvm::gpu::{GPU_CONTEXT, GpuContext};
use std::collections::{HashMap, hash_map::Entry};
use std::ffi::{CString, c_void};
use std::ptr;
use std::sync::{Arc, Mutex, OnceLock};

type ShapeKey = (usize, usize, usize);
type KernelKey = (ShapeKey, &'static str);

pub struct GpuKernelEntry {
    pub _module: CUmodule,
    pub function: CUfunction,
}

unsafe impl Send for GpuKernelEntry {}
unsafe impl Sync for GpuKernelEntry {}

pub struct GpuKernelCache {
    map: Mutex<HashMap<KernelKey, Arc<GpuKernelEntry>>>,
}

pub static GPU_KERNEL_CACHE: OnceLock<GpuKernelCache> = OnceLock::new();

impl GpuKernelCache {
    fn new() -> Self {
        Self {
            map: Mutex::new(HashMap::new()),
        }
    }

    fn get_or_load(
        &self,
        shape: ShapeKey,
        kernel_name: &'static str,
        fatbin: &[u8],
    ) -> Arc<GpuKernelEntry> {
        let key = (shape, kernel_name);
        {
            let map = self.map.lock().unwrap();
            if let Some(e) = map.get(&key).cloned() {
                return e;
            }
        }
        let mut module: CUmodule = ptr::null_mut();
        unsafe {
            check_cuda_error(
                cuModuleLoadData(&mut module, fatbin.as_ptr() as *const _),
                "cuModuleLoadData",
            );
        }
        let mut function: CUfunction = ptr::null_mut();
        let func_name = CString::new(kernel_name).unwrap();
        unsafe {
            check_cuda_error(
                cuModuleGetFunction(&mut function, module, func_name.as_ptr()),
                "Failed to get kernel function",
            );
        }
        let entry = Arc::new(GpuKernelEntry {
            _module: module,
            function,
        });
        let mut map = self.map.lock().unwrap();
        match map.entry(key) {
            Entry::Occupied(e) => e.get().clone(),
            Entry::Vacant(e) => {
                e.insert(entry.clone());
                entry
            }
        }
    }
}

pub unsafe fn ll_matmul(
    a: &[f32],
    a_shape: (usize, usize),
    b: &[f32],
    b_shape: (usize, usize),
) -> Vec<f32> {
    let (m, k) = a_shape;
    let (k2, n) = b_shape;
    assert_eq!(k, k2, "Matrix dimensions mismatch");

    let vec_a_size = m * k;
    let vec_b_size = k * n;
    let vec_c_size = m * n;

    let mut result = vec![0.0f32; vec_c_size];
    unsafe {
        let ctx = GPU_CONTEXT.get_or_init(GpuContext::new);
        check_cuda_error(cuCtxSetCurrent(ctx.compiledcontext), "cuCtxSetCurrent for compiled");

        let shape_key: ShapeKey = (m, n, k);
        let kernel_name: &'static str = "ll_matmul_gpu";
        let cache = GPU_KERNEL_CACHE.get_or_init(GpuKernelCache::new);
        let entry = cache.get_or_load(shape_key, kernel_name, GPU_FATBIN_PAYLOAD);

        let mut device_ptr_a: CUdeviceptr = 0 as *mut c_void;
        let mut device_ptr_b: CUdeviceptr = 0 as *mut c_void;
        let mut device_ptr_c: CUdeviceptr = 0 as *mut c_void;

        check_cuda_error(
            cuMemAlloc_v2(&mut device_ptr_a, vec_a_size * 4),
            "Failed to allocate memory for A",
        );
        check_cuda_error(
            cuMemAlloc_v2(&mut device_ptr_b, vec_b_size * 4),
            "Failed to allocate memory for B",
        );
        check_cuda_error(
            cuMemAlloc_v2(&mut device_ptr_c, vec_c_size * 4),
            "Failed to allocate memory for C",
        );

        check_cuda_error(
            cuMemcpyHtoD_v2(device_ptr_a, a.as_ptr() as *const c_void, vec_a_size * 4),
            "Failed to copy A to device",
        );
        check_cuda_error(
            cuMemcpyHtoD_v2(device_ptr_b, b.as_ptr() as *const c_void, vec_b_size * 4),
            "Failed to copy B to device",
        );

        let mut m_i32 = m as i32;
        let mut n_i32 = n as i32;
        let mut k_i32 = k as i32;

        let mut args: [*mut c_void; 6] = [
            &mut device_ptr_a as *mut _ as *mut c_void,
            &mut device_ptr_b as *mut _ as *mut c_void,
            &mut device_ptr_c as *mut _ as *mut c_void,
            &mut m_i32 as *mut _ as *mut c_void,
            &mut n_i32 as *mut _ as *mut c_void,
            &mut k_i32 as *mut _ as *mut c_void,
        ];

        let block_dim_x = 16u32;
        let block_dim_y = 16u32;
        let grid_dim_x = ((n as u32 + block_dim_x - 1) / block_dim_x).max(1);
        let grid_dim_y = ((m as u32 + block_dim_y - 1) / block_dim_y).max(1);

        check_cuda_error(
            cuLaunchKernel(
                entry.function,
                grid_dim_x,
                grid_dim_y,
                1,
                block_dim_x,
                block_dim_y,
                1,
                0,
                ptr::null_mut(),
                args.as_mut_ptr(),
                ptr::null_mut(),
            ),
            "Failed to launch kernel",
        );

        check_cuda_error(
            cuMemcpyDtoH_v2(
                result.as_mut_ptr() as *mut c_void,
                device_ptr_c,
                vec_c_size * 4,
            ),
            "Failed to copy result from device to host",
        );

        check_cuda_error(cuMemFree_v2(device_ptr_a), "Free A");
        check_cuda_error(cuMemFree_v2(device_ptr_b), "Free B");
        check_cuda_error(cuMemFree_v2(device_ptr_c), "Free C");
    }

    result
}
