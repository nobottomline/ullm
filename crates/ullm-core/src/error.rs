//! The crate-wide error and result types.

use std::fmt;

/// Result type used across uLLM.
pub type Result<T> = std::result::Result<T, Error>;

/// Top-level error type for uLLM.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// An I/O error while reading a model or other file.
    Io(std::io::Error),
    /// A model file was malformed or could not be parsed.
    Format(String),
    /// A requested feature, format, or architecture is not yet supported.
    Unsupported(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "i/o error: {e}"),
            Error::Format(m) => write!(f, "format error: {m}"),
            Error::Unsupported(m) => write!(f, "unsupported: {m}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}
