#![cfg(feature = "alloc-guard")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use newt::body::Body;
use newt::broadphase::Ray;
use newt::geom::{ConvexMesh, Geom};
use newt::joint::JointKind;
use newt::math::{Mat3, Quat, Vec3};
use newt::mjcf::load_mjcf_path;
use newt::solver::SolverMode;
use newt::tree::{Link, Tree};
use newt::world::{BroadPhaseMode, Integrator, ShapeDesc, World};

struct CountingAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static ALLOCATION_TEST_LOCK: Mutex<()> = Mutex::new(());

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
    let _lock = ALLOCATION_TEST_LOCK.lock().unwrap();
    const STEPS: usize = 16;
    // Each step allocates: tendon (2), external-force (2), and acceleration
    // result buffers. Previous state (tier-3 first optimization pass) held at
    // 12; NEWT-32 folds the ABA `tendon_qfrc` scratch into `AbaWorkspace` and
    // drops the dead `ext_body` Vec, dropping the steady-state count to 10.
    const EXPECTED_ALLOCATIONS: usize = 10;
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

#[test]
fn warmed_broadphase_updates_do_not_allocate() {
    let _lock = ALLOCATION_TEST_LOCK.lock().unwrap();
    let mut world = World::new();
    world.gravity = Vec3::ZERO;
    world.broadphase_mode = BroadPhaseMode::DynamicAabbTree;
    const COUNT: usize = 8;
    for index in 0..COUNT {
        let body = world.add_body(Body::solid_sphere(
            1.0,
            0.1,
            Vec3::new(index as f32 * 2.0, 0.0, 0.0),
            Quat::IDENTITY,
        ));
        world.add_geom(Geom::sphere(body, 0.1, Vec3::ZERO, 0.0));
    }

    let mut tree = Tree::new();
    tree.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::new(1000.0, 0.0, 0.0), Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    for index in 1..COUNT {
        tree.push_link(Link::new(
            Some(index - 1),
            JointKind::hinge(Vec3::Z),
            (Vec3::new(2.0, 0.0, 0.0), Quat::IDENTITY),
            (Vec3::ZERO, Quat::IDENTITY),
            1.0,
            Mat3::diag(1.0, 1.0, 1.0),
        ));
    }
    let tree_index = world.add_tree(tree);
    for link in 0..COUNT {
        world.add_geom(Geom::sphere_on_link(tree_index, link, 0.1, Vec3::ZERO, 0.0));
    }

    for _ in 0..4 {
        world.broadphase_pair_count();
    }
    reset_allocations();
    for _ in 0..100 {
        assert_eq!(world.broadphase_pair_count(), 0);
    }
    assert_eq!(allocation_count(), 0);

    reset_allocations();
    for _ in 0..100 {
        assert_eq!(world.broadphase_pair_count(), 0);
    }
    assert_eq!(allocation_count(), 0);
}

#[test]
fn warmed_scene_queries_do_not_allocate_for_links_or_meshes() {
    let _lock = ALLOCATION_TEST_LOCK.lock().unwrap();
    let mut world = World::new();
    world.gravity = Vec3::ZERO;

    let body = world.add_body(Body::solid_sphere(
        1.0,
        0.25,
        Vec3::new(2.0, 0.0, 0.0),
        Quat::IDENTITY,
    ));
    world.add_geom(Geom::sphere(body, 0.25, Vec3::ZERO, 0.0));

    let mut tree = Tree::new();
    tree.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::new(4.0, 0.0, 0.0), Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    let tree_id = world.add_tree(tree);
    world.add_geom(Geom::sphere_on_link(tree_id, 0, 0.25, Vec3::ZERO, 0.0));

    let mesh_id = world.add_mesh(ConvexMesh {
        vertices: vec![
            Vec3::new(-0.25, -0.25, -0.25),
            Vec3::new(0.25, -0.25, -0.25),
            Vec3::new(0.0, 0.25, -0.25),
            Vec3::new(0.0, 0.0, 0.25),
        ],
        faces: vec![[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
    });
    let mesh_body = world.add_body(Body::solid_sphere(
        1.0,
        0.25,
        Vec3::new(6.0, 0.0, 0.0),
        Quat::IDENTITY,
    ));
    world.add_geom(Geom::mesh(
        mesh_body,
        mesh_id,
        Vec3::ZERO,
        Quat::IDENTITY,
        0.0,
    ));

    let ray = Ray {
        origin: Vec3::ZERO,
        direction: Vec3::X,
    };
    reset_allocations();
    let _ = world.raycast(ray, 20.0, u32::MAX);
    assert_eq!(allocation_count(), 0);

    reset_allocations();
    let _ = world.shape_cast(
        ShapeDesc::Sphere { radius: 0.1 },
        newt::world::Pose {
            position: Vec3::ZERO,
            orientation: Quat::IDENTITY,
        },
        newt::world::Pose {
            position: Vec3::new(8.0, 0.0, 0.0),
            orientation: Quat::IDENTITY,
        },
        u32::MAX,
    );
    assert_eq!(allocation_count(), 0);

    for _ in 0..4 {
        let _ = world.raycast(ray, 20.0, u32::MAX);
        let _ = world.shape_cast(
            ShapeDesc::Sphere { radius: 0.1 },
            newt::world::Pose {
                position: Vec3::ZERO,
                orientation: Quat::IDENTITY,
            },
            newt::world::Pose {
                position: Vec3::new(8.0, 0.0, 0.0),
                orientation: Quat::IDENTITY,
            },
            u32::MAX,
        );
    }

    reset_allocations();
    for _ in 0..100 {
        let _ = world.raycast(ray, 20.0, u32::MAX);
        let _ = world.shape_cast(
            ShapeDesc::Sphere { radius: 0.1 },
            newt::world::Pose {
                position: Vec3::ZERO,
                orientation: Quat::IDENTITY,
            },
            newt::world::Pose {
                position: Vec3::new(8.0, 0.0, 0.0),
                orientation: Quat::IDENTITY,
            },
            u32::MAX,
        );
    }
    assert_eq!(allocation_count(), 0);
}
