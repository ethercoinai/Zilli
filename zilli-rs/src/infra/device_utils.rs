use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeviceType {
    CPU,
    CUDA,
    MPS,
    Auto,
}

static CURRENT_DEVICE: OnceLock<DeviceType> = OnceLock::new();

pub fn detect_device(_prefer: &str) -> DeviceType {
    DeviceType::CPU
}

pub fn get_device() -> &'static str {
    match CURRENT_DEVICE.get().unwrap_or(&DeviceType::CPU) {
        DeviceType::CPU => "cpu",
        DeviceType::CUDA => "cuda",
        DeviceType::MPS => "mps",
        DeviceType::Auto => "auto",
    }
}

pub fn set_device(device: DeviceType) {
    let _ = CURRENT_DEVICE.set(device);
}

pub fn is_cuda_available() -> bool {
    false
}

pub fn is_mps_available() -> bool {
    false
}

pub fn is_gpu_available() -> bool {
    is_cuda_available() || is_mps_available()
}

pub fn get_device_count() -> i32 {
    1
}
