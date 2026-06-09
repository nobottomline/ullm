//! uLLM core: fundamental types and the container-agnostic intermediate
//! representation (IR) shared by every loader, backend, and runtime.
//!
//! The IR decouples *where weights come from* (GGUF, SafeTensors, PyTorch) from
//! *how a model is executed*, so the rest of the engine never branches on file
//! format.

pub mod device;
pub mod dtype;
pub mod error;
pub mod ir;

pub use dtype::DType;
pub use error::{Error, Result};
