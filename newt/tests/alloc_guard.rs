#![cfg(feature = "alloc-guard")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use newt::mjcf::load_mjcf_path;
use newt::solver::SolverMode;
use newt::world::Integrator;

struct CountingAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: each operation forwards its valid arguments to the system allocator.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: the caller provides a valid allocation layout.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: the pointer and layout came from this allocator.
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: the pointer and layout came from this allocator.
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn reset_allocations() {
    ALLOCATIONS.store(0, Ordering::Relaxed);
}

fn allocation_count() -> usize {
    ALLOCATIONS.load(Ordering::Relaxed)
}

#[test]
fn optimized_euler_muscle_step_allocations_are_stable() {
    const STEPS: usize = 16;
    const EXPECTED_ALLOCATIONS: usize = 12;
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/references/muscle_pendulum.xml");
    let mut scene = load_mjcf_path(&path).expect("muscle scene should load");
    scene.world.integrator = Integrator::Euler;
    scene.world.solver.mode = SolverMode::Pgs;
    let mut world = scene.world;

    for _ in 0..3 {
        world.step();
    }

    let mut allocations = [0; STEPS];
    for count in &mut allocations {
        reset_allocations();
        world.step();
        *count = allocation_count();
    }

    assert!(
        allocations
            .iter()
            .all(|&count| count == EXPECTED_ALLOCATIONS)
    );
    assert_eq!(allocations[0], EXPECTED_ALLOCATIONS);
}
