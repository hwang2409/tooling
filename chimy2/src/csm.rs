//! Cascaded directional shadow-map API.
//!
//! This focused seam exposes the CSM configuration, fitting, rendering, and
//! sampling types. The legacy `shadow` paths remain available.

pub use crate::shadow::{
    CascadeShadowConfig, CascadeShadowState, cascade_index, fit_cascade_light_projection,
    practical_split_depths, render_cascade_shadow_maps, render_cascade_shadow_maps_with_config,
    snap_ortho_origin,
};
