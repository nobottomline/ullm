//! GGUF metadata values and their on-disk encoding.

use crate::reader::Reader;
use ullm_core::{Error, Result};

/// A single GGUF metadata value.
#[derive(Debug, Clone)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
}

impl Value {
    /// The string, if this value is a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    /// The elements, if this value is an array.
    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }

    /// Coerce any integer or bool variant to `u64`.
    pub fn to_u64(&self) -> Option<u64> {
        Some(match self {
            Value::U8(v) => *v as u64,
            Value::I8(v) => *v as u64,
            Value::U16(v) => *v as u64,
            Value::I16(v) => *v as u64,
            Value::U32(v) => *v as u64,
            Value::I32(v) => *v as u64,
            Value::U64(v) => *v,
            Value::I64(v) => *v as u64,
            Value::Bool(v) => *v as u64,
            _ => return None,
        })
    }

    /// Coerce a float variant to `f32`.
    pub fn to_f32(&self) -> Option<f32> {
        match self {
            Value::F32(v) => Some(*v),
            Value::F64(v) => Some(*v as f32),
            _ => None,
        }
    }
}

// GGUF metadata value-type ids.
const T_U8: u32 = 0;
const T_I8: u32 = 1;
const T_U16: u32 = 2;
const T_I16: u32 = 3;
const T_U32: u32 = 4;
const T_I32: u32 = 5;
const T_F32: u32 = 6;
const T_BOOL: u32 = 7;
const T_STRING: u32 = 8;
const T_ARRAY: u32 = 9;
const T_U64: u32 = 10;
const T_I64: u32 = 11;
const T_F64: u32 = 12;

/// Read one value of the given type id from the cursor.
pub fn read_value(r: &mut Reader, type_id: u32) -> Result<Value> {
    Ok(match type_id {
        T_U8 => Value::U8(r.u8()?),
        T_I8 => Value::I8(r.i8()?),
        T_U16 => Value::U16(r.u16()?),
        T_I16 => Value::I16(r.i16()?),
        T_U32 => Value::U32(r.u32()?),
        T_I32 => Value::I32(r.i32()?),
        T_F32 => Value::F32(r.f32()?),
        T_BOOL => Value::Bool(r.bool()?),
        T_STRING => Value::String(r.gguf_string()?),
        T_U64 => Value::U64(r.u64()?),
        T_I64 => Value::I64(r.i64()?),
        T_F64 => Value::F64(r.f64()?),
        T_ARRAY => {
            let elem_type = r.u32()?;
            if elem_type == T_ARRAY {
                return Err(Error::Format("nested GGUF arrays are not supported".into()));
            }
            let count = r.u64()? as usize;
            // Cap the pre-allocation so a corrupt count can't request a huge Vec.
            let mut items = Vec::with_capacity(count.min(1 << 16));
            for _ in 0..count {
                items.push(read_value(r, elem_type)?);
            }
            Value::Array(items)
        }
        other => return Err(Error::Format(format!("unknown GGUF value type id {other}"))),
    })
}
