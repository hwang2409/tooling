use core::sync::atomic::{AtomicUsize, Ordering};

static DEPTH_FALLBACKS: AtomicUsize = AtomicUsize::new(0);

#[cfg(test)]
pub(crate) fn reset_depth_fallbacks() {
    DEPTH_FALLBACKS.store(0, Ordering::Relaxed);
}

#[cfg(test)]
pub(crate) fn depth_fallbacks() -> usize {
    DEPTH_FALLBACKS.load(Ordering::Relaxed)
}

#[inline]
fn checked_slice_failed<T>() -> Option<T> {
    DEPTH_FALLBACKS.fetch_add(1, Ordering::Relaxed);
    None
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn depth_mask(
    covered: &[u8],
    depths: &[f32],
    buffer_depth: &[f32],
    offset: usize,
    buffer_offset: usize,
    depth_test: bool,
) -> Option<u8> {
    let Some(end) = offset.checked_add(4) else {
        return checked_slice_failed();
    };
    let Some(covered) = covered.get(offset..end) else {
        return checked_slice_failed();
    };
    let Some(depths) = depths.get(offset..end) else {
        return checked_slice_failed();
    };
    let Some(buffer_end) = buffer_offset.checked_add(4) else {
        return checked_slice_failed();
    };
    let Some(buffer_depth) = buffer_depth.get(buffer_offset..buffer_end) else {
        return checked_slice_failed();
    };

    let mut accepted = [0_u32; 4];
    // safety: all three four-lane slices are checked immediately above.
    unsafe {
        use core::arch::aarch64::{vcgeq_f32, vdupq_n_u32, vld1q_f32, vmvnq_u32, vst1q_u32};

        let depth = vld1q_f32(depths.as_ptr());
        let previous = vld1q_f32(buffer_depth.as_ptr());
        let rejected = if depth_test {
            vcgeq_f32(depth, previous)
        } else {
            vdupq_n_u32(0)
        };
        vst1q_u32(accepted.as_mut_ptr(), vmvnq_u32(rejected));
    }

    let mut mask = 0;
    for lane in 0..4 {
        if covered[lane] != 0 && accepted[lane] != 0 {
            mask |= 1 << lane;
        }
    }
    Some(mask)
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
pub(crate) fn depth_mask(
    _covered: &[u8],
    _depths: &[f32],
    _buffer_depth: &[f32],
    _offset: usize,
    _buffer_offset: usize,
    _depth_test: bool,
) -> Option<u8> {
    let _ = checked_slice_failed::<u8>();
    None
}
