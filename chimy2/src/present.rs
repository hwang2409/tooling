use crate::fb::Framebuffer;
use std::collections::HashSet;
use std::error::Error;
use std::num::NonZeroU32;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowAttributes, WindowId};

pub use winit::event::MouseButton as PresentMouseButton;
pub use winit::keyboard::KeyCode as PresentKeyCode;

#[derive(Clone, Copy, Debug, Default)]
pub struct FrameTiming {
    pub draw: std::time::Duration,
    pub present: std::time::Duration,
    pub total: std::time::Duration,
}

#[derive(Clone, Debug, Default)]
pub struct InputState {
    pressed: HashSet<KeyCode>,
    mouse_buttons: HashSet<MouseButton>,
    mouse_delta: (f32, f32),
    scroll_delta: f32,
    last_cursor: Option<(f32, f32)>,
}

impl InputState {
    pub fn is_down(&self, key: KeyCode) -> bool {
        self.pressed.contains(&key)
    }

    pub fn is_mouse_down(&self, button: PresentMouseButton) -> bool {
        self.mouse_buttons.contains(&button)
    }

    pub const fn mouse_delta(&self) -> (f32, f32) {
        self.mouse_delta
    }

    pub const fn scroll_delta(&self) -> f32 {
        self.scroll_delta
    }

    fn clear_frame_deltas(&mut self) {
        self.mouse_delta = (0.0, 0.0);
        self.scroll_delta = 0.0;
    }
}

trait Backend {
    fn resize(&mut self, width: u32, height: u32) -> Result<(), String>;
    fn present(&mut self, color: &[u32]) -> Result<(), String>;
}

fn to_softbuffer_color(color: &mut [u32]) {
    for pixel in color {
        *pixel &= 0x00ff_ffff;
    }
}

struct SoftbufferBackend<D, W>
where
    D: winit::raw_window_handle::HasDisplayHandle,
    W: winit::raw_window_handle::HasWindowHandle,
{
    surface: softbuffer::Surface<D, W>,
}

impl<D, W> Backend for SoftbufferBackend<D, W>
where
    D: winit::raw_window_handle::HasDisplayHandle,
    W: winit::raw_window_handle::HasWindowHandle,
{
    fn resize(&mut self, width: u32, height: u32) -> Result<(), String> {
        let width = NonZeroU32::new(width).ok_or_else(|| "window width is zero".to_string())?;
        let height = NonZeroU32::new(height).ok_or_else(|| "window height is zero".to_string())?;
        // The dimensions come from winit's physical window size.
        self.surface
            .resize(width, height)
            .map_err(|error| error.to_string())
    }

    fn present(&mut self, color: &[u32]) -> Result<(), String> {
        // softbuffer's macOS CG backend allocates in every buffer_mut call;
        // the zero-allocation invariant covers project-owned code only.
        let mut buffer = self
            .surface
            .buffer_mut()
            .map_err(|error| error.to_string())?;
        if buffer.len() != color.len() {
            return Err(format!(
                "surface has {} pixels, framebuffer has {}",
                buffer.len(),
                color.len()
            ));
        }
        buffer.copy_from_slice(color);
        to_softbuffer_color(&mut buffer);
        buffer.present().map_err(|error| error.to_string())
    }
}

trait EventLoopControl {
    fn exit(&mut self);
}

struct ActiveEventLoopControl<'a> {
    event_loop: &'a ActiveEventLoop,
}

impl EventLoopControl for ActiveEventLoopControl<'_> {
    fn exit(&mut self) {
        self.event_loop.exit();
    }
}

struct App<F> {
    title: String,
    logical_width: u32,
    logical_height: u32,
    window: Option<&'static Window>,
    backend: Option<Box<dyn Backend>>,
    framebuffer: Framebuffer,
    draw: F,
    started: Instant,
    frames: usize,
    max_frames: Option<usize>,
    max_seconds: Option<f32>,
    error: Option<String>,
    input: InputState,
    observer: Option<Box<dyn FnMut(FrameTiming)>>,
}

impl<F> App<F>
where
    F: FnMut(&mut Framebuffer, f32, &InputState),
{
    fn fail<C: EventLoopControl>(&mut self, event_loop: &mut C, error: impl Into<String>) {
        self.error = Some(error.into());
        event_loop.exit();
    }

    /// Returns true once the caller-requested frame budget has been drawn.
    ///
    /// Guards both the redraw arm and the request loop so a redraw already
    /// queued when `exit()` was called cannot draw an extra frame.
    fn frame_budget_exhausted(&self) -> bool {
        self.max_frames
            .is_some_and(|max_frames| self.frames >= max_frames)
    }

    fn time_budget_exhausted(&self) -> bool {
        self.max_seconds
            .is_some_and(|max_seconds| self.started.elapsed().as_secs_f32() >= max_seconds)
    }

    fn resize(&mut self, width: u32, height: u32) -> Result<(), String> {
        self.framebuffer.resize(width as usize, height as usize);
        if width != 0 && height != 0 {
            if let Some(backend) = self.backend.as_deref_mut() {
                backend.resize(width, height)?;
            }
        }
        Ok(())
    }

    fn handle_resize<C: EventLoopControl>(&mut self, event_loop: &mut C, width: u32, height: u32) {
        if let Err(error) = self.resize(width, height) {
            self.fail(event_loop, error);
        }
    }

    fn dispatch_window_event<C: EventLoopControl>(
        &mut self,
        event_loop: &mut C,
        window: Option<&Window>,
        event: WindowEvent,
    ) {
        match &event {
            WindowEvent::Focused(false) => {
                self.input.pressed.clear();
                self.input.mouse_buttons.clear();
                self.input.last_cursor = None;
                self.input.clear_frame_deltas();
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state,
                        ..
                    },
                ..
            } => {
                if *key == KeyCode::Escape && *state == ElementState::Pressed {
                    event_loop.exit();
                    return;
                }
                if *state == ElementState::Pressed {
                    self.input.pressed.insert(*key);
                } else {
                    self.input.pressed.remove(key);
                }
            }
            WindowEvent::MouseInput { button, state, .. } => {
                if *state == ElementState::Pressed {
                    self.input.mouse_buttons.insert(*button);
                } else {
                    self.input.mouse_buttons.remove(button);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let cursor = (position.x as f32, position.y as f32);
                if let Some(previous) = self.input.last_cursor {
                    self.input.mouse_delta.0 += cursor.0 - previous.0;
                    self.input.mouse_delta.1 += cursor.1 - previous.1;
                }
                self.input.last_cursor = Some(cursor);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.input.scroll_delta += match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y * 24.0,
                    MouseScrollDelta::PixelDelta(value) => value.y as f32,
                };
            }
            _ => {}
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => self.handle_resize(event_loop, size.width, size.height),
            WindowEvent::ScaleFactorChanged { .. } => {
                let Some(window) = window else { return };
                // inner_size is already in physical pixels after this event.
                let size = window.inner_size();
                self.handle_resize(event_loop, size.width, size.height);
            }
            WindowEvent::RedrawRequested => {
                if self.error.is_some()
                    || self.framebuffer.width == 0
                    || self.framebuffer.height == 0
                    || self.frame_budget_exhausted()
                {
                    return;
                }
                if self.time_budget_exhausted() {
                    event_loop.exit();
                    return;
                }
                let frame_started = Instant::now();
                (self.draw)(
                    &mut self.framebuffer,
                    self.started.elapsed().as_secs_f32(),
                    &self.input,
                );
                let draw = frame_started.elapsed();
                let present_started = Instant::now();
                if let Some(backend) = self.backend.as_mut() {
                    if let Err(error) = backend.present(&self.framebuffer.color) {
                        self.fail(event_loop, error);
                        return;
                    }
                }
                if let Some(observer) = self.observer.as_mut() {
                    observer(FrameTiming {
                        draw,
                        present: present_started.elapsed(),
                        total: frame_started.elapsed(),
                    });
                }
                self.input.clear_frame_deltas();
                self.frames += 1;
                if self.frame_budget_exhausted() || self.time_budget_exhausted() {
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }
}

impl<F> ApplicationHandler for App<F>
where
    F: FnMut(&mut Framebuffer, f32, &InputState),
{
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attributes = WindowAttributes::default()
            .with_title(self.title.clone())
            .with_inner_size(LogicalSize::new(self.logical_width, self.logical_height));
        let window: &'static Window = match event_loop.create_window(attributes) {
            Ok(window) => Box::leak(Box::new(window)),
            Err(error) => {
                let mut control = ActiveEventLoopControl { event_loop };
                self.fail(&mut control, error.to_string());
                return;
            }
        };
        let context = match softbuffer::Context::new(event_loop.owned_display_handle()) {
            Ok(context) => context,
            Err(error) => {
                let mut control = ActiveEventLoopControl { event_loop };
                self.fail(&mut control, error.to_string());
                return;
            }
        };
        let surface = match softbuffer::Surface::new(&context, window) {
            Ok(surface) => surface,
            Err(error) => {
                let mut control = ActiveEventLoopControl { event_loop };
                self.fail(&mut control, error.to_string());
                return;
            }
        };
        self.window = Some(window);
        self.backend = Some(Box::new(SoftbufferBackend { surface }));
        let size = window.inner_size();
        let mut control = ActiveEventLoopControl { event_loop };
        self.handle_resize(&mut control, size.width, size.height);
        if self.error.is_some() {
            return;
        }
        window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(window) = self.window else { return };
        let mut control = ActiveEventLoopControl { event_loop };
        self.dispatch_window_event(&mut control, Some(window), event);
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.frame_budget_exhausted() {
            return;
        }
        if self.time_budget_exhausted() {
            event_loop.exit();
            return;
        }
        if let Some(window) = self.window {
            window.request_redraw();
        }
    }
}

trait EventRunner<F>
where
    F: FnMut(&mut Framebuffer, f32, &InputState),
{
    fn run(&mut self, app: &mut App<F>) -> Result<(), Box<dyn Error>>;
}

struct WinitEventRunner;

impl<F> EventRunner<F> for WinitEventRunner
where
    F: FnMut(&mut Framebuffer, f32, &InputState),
{
    fn run(&mut self, app: &mut App<F>) -> Result<(), Box<dyn Error>> {
        let event_loop = EventLoop::new()?;
        event_loop.set_control_flow(ControlFlow::Poll);
        event_loop.run_app(app)?;
        Ok(())
    }
}

struct RunOptions {
    max_frames: Option<usize>,
    max_seconds: Option<f32>,
    observer: Option<Box<dyn FnMut(FrameTiming)>>,
}

fn run_with_runner<F, R>(
    title: &str,
    width: u32,
    height: u32,
    options: RunOptions,
    draw: F,
    runner: &mut R,
) -> Result<(), Box<dyn Error>>
where
    F: FnMut(&mut Framebuffer, f32, &InputState),
    R: EventRunner<F>,
{
    if options.max_frames == Some(0) {
        return Ok(());
    }
    let mut app = App {
        title: title.to_string(),
        logical_width: width,
        logical_height: height,
        window: None,
        backend: None,
        framebuffer: Framebuffer::new(width as usize, height as usize),
        draw,
        started: Instant::now(),
        frames: 0,
        max_frames: options.max_frames,
        max_seconds: options.max_seconds,
        error: None,
        input: InputState::default(),
        observer: options.observer,
    };
    runner.run(&mut app)?;
    app.error.map_or(Ok(()), |error| Err(error.into()))
}

/// Run a windowed framebuffer demo.
pub fn run<F>(
    title: &str,
    width: u32,
    height: u32,
    max_frames: Option<usize>,
    draw: F,
) -> Result<(), Box<dyn Error>>
where
    F: FnMut(&mut Framebuffer, f32),
{
    let mut draw = draw;
    let mut runner = WinitEventRunner;
    run_with_runner(
        title,
        width,
        height,
        RunOptions {
            max_frames,
            max_seconds: None,
            observer: None,
        },
        move |framebuffer, elapsed, _| draw(framebuffer, elapsed),
        &mut runner,
    )
}

/// Run a windowed framebuffer demo with physical-key state.
pub fn run_with_input<F>(
    title: &str,
    width: u32,
    height: u32,
    max_frames: Option<usize>,
    draw: F,
) -> Result<(), Box<dyn Error>>
where
    F: FnMut(&mut Framebuffer, f32, &InputState),
{
    let mut runner = WinitEventRunner;
    run_with_runner(
        title,
        width,
        height,
        RunOptions {
            max_frames,
            max_seconds: None,
            observer: None,
        },
        draw,
        &mut runner,
    )
}

/// Run a windowed framebuffer demo and report draw/present timings.
pub fn run_with_input_timed<F, O>(
    title: &str,
    width: u32,
    height: u32,
    max_seconds: f32,
    draw: F,
    observe: O,
) -> Result<(), Box<dyn Error>>
where
    F: FnMut(&mut Framebuffer, f32, &InputState),
    O: FnMut(FrameTiming) + 'static,
{
    let mut runner = WinitEventRunner;
    run_with_runner(
        title,
        width,
        height,
        RunOptions {
            max_frames: None,
            max_seconds: Some(max_seconds),
            observer: Some(Box::new(observe)),
        },
        draw,
        &mut runner,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FailingBackend;

    impl Backend for FailingBackend {
        fn resize(&mut self, _: u32, _: u32) -> Result<(), String> {
            Err("resize failed".to_string())
        }

        fn present(&mut self, _: &[u32]) -> Result<(), String> {
            Ok(())
        }
    }

    struct HeadlessEventRunner {
        exited: bool,
    }

    impl EventLoopControl for HeadlessEventRunner {
        fn exit(&mut self) {
            self.exited = true;
        }
    }

    impl<F> EventRunner<F> for HeadlessEventRunner
    where
        F: FnMut(&mut Framebuffer, f32, &InputState),
    {
        fn run(&mut self, app: &mut App<F>) -> Result<(), Box<dyn Error>> {
            app.backend = Some(Box::new(FailingBackend));
            app.handle_resize(self, 2, 2);
            Ok(())
        }
    }

    #[test]
    fn resize_failure_is_returned_to_run() {
        let mut runner = HeadlessEventRunner { exited: false };
        let result = run_with_runner(
            "",
            2,
            2,
            RunOptions {
                max_frames: Some(1),
                max_seconds: None,
                observer: None,
            },
            |_: &mut Framebuffer, _: f32, _: &InputState| {},
            &mut runner,
        );

        assert_eq!(
            result.map_err(|error| error.to_string()),
            Err("resize failed".to_string())
        );
        assert!(runner.exited);
    }

    #[test]
    fn presentation_clears_alpha_byte() {
        let mut color = [0xff12_3456, 0x8012_3456, 0x0012_3456];
        to_softbuffer_color(&mut color);
        assert_eq!(color, [0x0012_3456, 0x0012_3456, 0x0012_3456]);
    }

    struct CountingBackend;

    impl Backend for CountingBackend {
        fn resize(&mut self, _: u32, _: u32) -> Result<(), String> {
            Ok(())
        }

        fn present(&mut self, _: &[u32]) -> Result<(), String> {
            Ok(())
        }
    }

    #[test]
    fn redraw_over_limit_never_draws_more_than_max_frames() {
        use std::cell::Cell;
        use std::rc::Rc;

        let draw_count = Rc::new(Cell::new(0usize));
        let counter = draw_count.clone();
        let mut app = App {
            title: String::new(),
            logical_width: 2,
            logical_height: 2,
            window: None,
            backend: Some(Box::new(CountingBackend)),
            framebuffer: Framebuffer::new(2, 2),
            draw: move |_: &mut Framebuffer, _: f32, _: &InputState| {
                counter.set(counter.get() + 1);
            },
            started: Instant::now(),
            frames: 0,
            max_frames: Some(3),
            max_seconds: None,
            error: None,
            input: InputState::default(),
            observer: None,
        };

        let mut runner = HeadlessEventRunner { exited: false };
        for _ in 0..8 {
            app.dispatch_window_event(&mut runner, None, WindowEvent::RedrawRequested);
        }

        assert_eq!(
            draw_count.get(),
            3,
            "draw callback fired past --frames limit"
        );
        assert!(
            runner.exited,
            "event loop should exit once the budget is met"
        );
        assert_eq!(app.frames, 3);
    }

    #[test]
    fn expired_time_budget_exits_before_redraw() {
        use std::cell::Cell;
        use std::rc::Rc;

        let draw_count = Rc::new(Cell::new(0usize));
        let counter = draw_count.clone();
        let mut app = App {
            title: String::new(),
            logical_width: 2,
            logical_height: 2,
            window: None,
            backend: Some(Box::new(CountingBackend)),
            framebuffer: Framebuffer::new(2, 2),
            draw: move |_: &mut Framebuffer, _: f32, _: &InputState| {
                counter.set(counter.get() + 1);
            },
            started: Instant::now(),
            frames: 0,
            max_frames: None,
            max_seconds: Some(0.0),
            error: None,
            input: InputState::default(),
            observer: None,
        };

        let mut runner = HeadlessEventRunner { exited: false };
        app.dispatch_window_event(&mut runner, None, WindowEvent::RedrawRequested);

        assert_eq!(draw_count.get(), 0);
        assert!(runner.exited);
    }

    #[test]
    fn focus_loss_clears_pressed_keys_headlessly() {
        let mut app = App {
            title: String::new(),
            logical_width: 2,
            logical_height: 2,
            window: None,
            backend: None,
            framebuffer: Framebuffer::new(2, 2),
            draw: |_: &mut Framebuffer, _: f32, _: &InputState| {},
            started: Instant::now(),
            frames: 0,
            max_frames: None,
            max_seconds: None,
            error: None,
            input: InputState::default(),
            observer: None,
        };
        app.input.pressed.insert(KeyCode::KeyW);
        assert!(app.input.is_down(KeyCode::KeyW));

        let mut runner = HeadlessEventRunner { exited: false };
        app.dispatch_window_event(&mut runner, None, WindowEvent::Focused(false));

        assert!(app.input.pressed.is_empty());
    }
}
