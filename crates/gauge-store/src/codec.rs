//! Bit-packed Gorilla timestamp and float encoding.

use crate::Sample;

const MAGIC: &[u8; 8] = b"GAUGOR01";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CodecError(pub(crate) String);

impl std::fmt::Display for CodecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CodecError {}

#[derive(Clone, Copy)]
struct ValueWindow {
    leading: u8,
    trailing: u8,
}

pub(crate) fn encode(samples: &[Sample]) -> Result<Vec<u8>, CodecError> {
    let mut output = Vec::with_capacity(samples.len().saturating_mul(3).saturating_add(16));
    output.extend_from_slice(MAGIC);
    output.extend_from_slice(&(samples.len() as u32).to_le_bytes());
    if samples.is_empty() {
        return Ok(output);
    }

    let mut bits = BitWriter::new();
    bits.write_bits(samples[0].timestamp as u64, 64);
    bits.write_bits(samples[0].value.to_bits(), 64);
    if samples.len() > 1 {
        let mut previous_timestamp = samples[0].timestamp;
        let mut delta = samples[1]
            .timestamp
            .checked_sub(previous_timestamp)
            .ok_or_else(|| CodecError("timestamp delta overflow".to_owned()))?;
        if delta < 0 {
            return Err(CodecError("samples must be sorted by timestamp".to_owned()));
        }
        bits.write_bits(delta as u64, 64);
        let mut previous_value = samples[0].value.to_bits();
        let mut window = None;
        encode_value(
            &mut bits,
            previous_value,
            samples[1].value.to_bits(),
            &mut window,
        );
        previous_timestamp = samples[1].timestamp;
        previous_value = samples[1].value.to_bits();

        for sample in &samples[2..] {
            let next_delta = sample
                .timestamp
                .checked_sub(previous_timestamp)
                .ok_or_else(|| CodecError("timestamp delta overflow".to_owned()))?;
            if next_delta < 0 {
                return Err(CodecError("samples must be sorted by timestamp".to_owned()));
            }
            let dod = next_delta
                .checked_sub(delta)
                .ok_or_else(|| CodecError("delta-of-delta overflow".to_owned()))?;
            encode_delta_of_delta(&mut bits, dod)?;
            encode_value(
                &mut bits,
                previous_value,
                sample.value.to_bits(),
                &mut window,
            );
            delta = next_delta;
            previous_timestamp = sample.timestamp;
            previous_value = sample.value.to_bits();
        }
    }
    output.extend_from_slice(&bits.finish());
    Ok(output)
}

pub(crate) fn decode(bytes: &[u8]) -> Result<Vec<Sample>, CodecError> {
    if bytes.len() < MAGIC.len() + 4 || &bytes[..MAGIC.len()] != MAGIC {
        return Err(CodecError("invalid Gorilla chunk magic".to_owned()));
    }
    let count = u32::from_le_bytes(bytes[8..12].try_into().expect("four-byte count")) as usize;
    let payload = &bytes[12..];
    let max_count = payload.len().saturating_mul(8);
    if count > max_count {
        return Err(CodecError(
            "sample count exceeds encoded payload capacity".to_owned(),
        ));
    }
    if count == 0 {
        return Ok(Vec::new());
    }
    let mut bits = BitReader::new(payload);
    let first_timestamp = bits.read_bits(64)? as i64;
    let first_value = bits.read_bits(64)?;
    let mut samples = Vec::new();
    samples
        .try_reserve_exact(count)
        .map_err(|_| CodecError("decoded sample allocation failed".to_owned()))?;
    samples.push(Sample::new(first_timestamp, f64::from_bits(first_value)));
    if count == 1 {
        return Ok(samples);
    }

    let mut delta = bits.read_bits(64)? as i64;
    if delta < 0 {
        return Err(CodecError("negative first timestamp delta".to_owned()));
    }
    let mut timestamp = first_timestamp
        .checked_add(delta)
        .ok_or_else(|| CodecError("timestamp overflow".to_owned()))?;
    let mut previous_value = first_value;
    let mut window = None;
    let second_value = decode_value(&mut bits, previous_value, &mut window)?;
    samples.push(Sample::new(timestamp, f64::from_bits(second_value)));
    previous_value = second_value;

    for _ in 2..count {
        let dod = decode_delta_of_delta(&mut bits)?;
        delta = delta
            .checked_add(dod)
            .ok_or_else(|| CodecError("timestamp delta overflow".to_owned()))?;
        if delta < 0 {
            return Err(CodecError("negative timestamp delta".to_owned()));
        }
        timestamp = timestamp
            .checked_add(delta)
            .ok_or_else(|| CodecError("timestamp overflow".to_owned()))?;
        let value = decode_value(&mut bits, previous_value, &mut window)?;
        samples.push(Sample::new(timestamp, f64::from_bits(value)));
        previous_value = value;
    }
    Ok(samples)
}

fn encode_delta_of_delta(bits: &mut BitWriter, dod: i64) -> Result<(), CodecError> {
    if dod == 0 {
        bits.write_bit(false);
    } else if (-63..=64).contains(&dod) {
        bits.write_bits(0b10, 2);
        bits.write_bits((dod + 63) as u64, 7);
    } else if (-255..=256).contains(&dod) {
        bits.write_bits(0b110, 3);
        bits.write_bits((dod + 255) as u64, 9);
    } else if (-2047..=2048).contains(&dod) {
        bits.write_bits(0b1110, 4);
        bits.write_bits((dod + 2047) as u64, 12);
    } else {
        let value = i32::try_from(dod)
            .map_err(|_| CodecError("delta-of-delta exceeds 32-bit encoding".to_owned()))?;
        bits.write_bits(0b1111, 4);
        bits.write_bits(value as u32 as u64, 32);
    }
    Ok(())
}

fn decode_delta_of_delta(bits: &mut BitReader<'_>) -> Result<i64, CodecError> {
    if bits.read_bit()? == 0 {
        return Ok(0);
    }
    if bits.read_bit()? == 0 {
        return Ok(bits.read_bits(7)? as i64 - 63);
    }
    if bits.read_bit()? == 0 {
        return Ok(bits.read_bits(9)? as i64 - 255);
    }
    if bits.read_bit()? == 0 {
        return Ok(bits.read_bits(12)? as i64 - 2047);
    }
    Ok((bits.read_bits(32)? as u32 as i32) as i64)
}

fn encode_value(bits: &mut BitWriter, previous: u64, next: u64, window: &mut Option<ValueWindow>) {
    let xor = previous ^ next;
    if xor == 0 {
        bits.write_bit(false);
        return;
    }
    bits.write_bit(true);
    let leading = xor.leading_zeros() as u8;
    let trailing = xor.trailing_zeros() as u8;
    if let Some(previous_window) = *window
        && leading >= previous_window.leading
        && trailing >= previous_window.trailing
    {
        bits.write_bit(false);
        let meaningful = 64 - previous_window.leading - previous_window.trailing;
        bits.write_bits(xor >> previous_window.trailing, meaningful);
        return;
    }

    // Gorilla stores leading zeros in five bits. Values with more than 31
    // leading zeros use a 31-bit leading field and carry the extra zeros in
    // the meaningful payload, preserving every bit exactly.
    let stored_leading = leading.min(31);
    let meaningful = 64 - stored_leading - trailing;
    bits.write_bit(true);
    bits.write_bits(stored_leading as u64, 5);
    // Six bits represent lengths 1..63; zero is Gorilla's representation of
    // the full 64-bit payload.
    bits.write_bits(if meaningful == 64 { 0 } else { meaningful } as u64, 6);
    bits.write_bits(xor >> trailing, meaningful);
    *window = Some(ValueWindow {
        leading: stored_leading,
        trailing,
    });
}

fn decode_value(
    bits: &mut BitReader<'_>,
    previous: u64,
    window: &mut Option<ValueWindow>,
) -> Result<u64, CodecError> {
    if bits.read_bit()? == 0 {
        return Ok(previous);
    }
    if bits.read_bit()? == 0 {
        let window = window.ok_or_else(|| CodecError("missing Gorilla XOR window".to_owned()))?;
        let meaningful = 64 - window.leading - window.trailing;
        return Ok(previous ^ (bits.read_bits(meaningful)? << window.trailing));
    }
    let leading = bits.read_bits(5)? as u8;
    let meaningful_field = bits.read_bits(6)? as u8;
    let meaningful = if meaningful_field == 0 {
        64
    } else {
        meaningful_field
    };
    if leading > 31 || u16::from(leading) + u16::from(meaningful) > 64 {
        return Err(CodecError("invalid Gorilla XOR window".to_owned()));
    }
    let trailing = 64 - leading - meaningful;
    let xor = bits.read_bits(meaningful)? << trailing;
    *window = Some(ValueWindow { leading, trailing });
    Ok(previous ^ xor)
}

struct BitWriter {
    bytes: Vec<u8>,
    current: u8,
    used: u8,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            current: 0,
            used: 0,
        }
    }

    fn write_bit(&mut self, value: bool) {
        self.current = (self.current << 1) | u8::from(value);
        self.used += 1;
        if self.used == 8 {
            self.bytes.push(self.current);
            self.current = 0;
            self.used = 0;
        }
    }

    fn write_bits(&mut self, value: u64, count: u8) {
        for shift in (0..count).rev() {
            self.write_bit((value >> shift) & 1 != 0);
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.used != 0 {
            self.current <<= 8 - self.used;
            self.bytes.push(self.current);
        }
        self.bytes
    }
}

struct BitReader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn read_bit(&mut self) -> Result<u8, CodecError> {
        let byte = self.position / 8;
        if byte >= self.bytes.len() {
            return Err(CodecError("truncated Gorilla chunk".to_owned()));
        }
        let bit = 7 - self.position % 8;
        self.position += 1;
        Ok((self.bytes[byte] >> bit) & 1)
    }

    fn read_bits(&mut self, count: u8) -> Result<u64, CodecError> {
        let mut value = 0;
        for _ in 0..count {
            value = (value << 1) | u64::from(self.read_bit()?);
        }
        Ok(value)
    }
}
