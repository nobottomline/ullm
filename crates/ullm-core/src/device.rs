//! Best-effort detection of the host's compute hardware.
//!
//! Phase 0 implementation: on macOS we read a few `sysctl` values. This is
//! intentionally dependency-free and will be replaced by a richer probe (GPU
//! core count, live memory pressure, Metal feature set) later.

/// A snapshot of the host's inference-relevant hardware.
#[derive(Debug, Clone)]
pub struct Hardware {
    /// Human-readable chip / CPU name, e.g. "Apple M4 Max".
    pub chip: String,
    /// Total physical (unified, on Apple Silicon) memory in bytes.
    pub total_memory: u64,
    /// Logical CPU core count.
    pub cpu_cores: u32,
    /// True on Apple Silicon (arm64 macOS).
    pub apple_silicon: bool,
    /// Whether a Metal GPU backend is expected to be available.
    pub metal: bool,
}

impl Hardware {
    /// Detect the host hardware. Never fails; unknown fields fall back to
    /// conservative defaults.
    pub fn detect() -> Self {
        let apple_silicon = cfg!(all(target_os = "macos", target_arch = "aarch64"));
        Hardware {
            chip: sysctl("machdep.cpu.brand_string").unwrap_or_else(|| "unknown".into()),
            total_memory: sysctl("hw.memsize")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0),
            cpu_cores: sysctl("hw.ncpu").and_then(|s| s.parse().ok()).unwrap_or(0),
            apple_silicon,
            metal: apple_silicon,
        }
    }

    /// Total memory in gibibytes, rounded to one decimal place.
    pub fn total_memory_gib(&self) -> f64 {
        (self.total_memory as f64 / (1024.0 * 1024.0 * 1024.0) * 10.0).round() / 10.0
    }
}

#[cfg(target_os = "macos")]
fn sysctl(key: &str) -> Option<String> {
    use std::process::Command;
    let out = Command::new("sysctl").arg("-n").arg(key).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() { None } else { Some(s) }
}

#[cfg(not(target_os = "macos"))]
fn sysctl(_key: &str) -> Option<String> {
    None
}
