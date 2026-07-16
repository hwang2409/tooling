use std::collections::BTreeMap;
#[cfg(target_os = "linux")]
use std::fs;
use std::sync::Mutex;
#[cfg(test)]
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use sysinfo::System;

use crate::{write_metric, write_type};

/// Collects process metrics aggregated by configured process-name pattern.
#[derive(Debug)]
pub struct ProcCollector {
    patterns: Vec<String>,
    state: Mutex<ProcState>,
    #[cfg(test)]
    panic_next: AtomicBool,
    #[cfg(test)]
    rebuild_count: AtomicUsize,
}

#[derive(Debug)]
struct ProcState {
    system: System,
    primed: bool,
}

impl ProcState {
    fn new() -> Self {
        Self {
            system: System::new(),
            primed: false,
        }
    }
}

impl ProcCollector {
    pub fn new(patterns: Vec<String>) -> Self {
        Self {
            patterns,
            state: Mutex::new(ProcState::new()),
            #[cfg(test)]
            panic_next: AtomicBool::new(false),
            #[cfg(test)]
            rebuild_count: AtomicUsize::new(0),
        }
    }

    pub fn patterns(&self) -> &[String] {
        &self.patterns
    }

    pub fn scrape(&self) -> String {
        let mut state = match self.state.lock() {
            Ok(state) => state,
            Err(poisoned) => {
                // A panic during a refresh can leave sysinfo's internal state
                // half-updated. Start with a clean baseline for the retry.
                let mut state = poisoned.into_inner();
                *state = ProcState::new();
                self.state.clear_poison();
                #[cfg(test)]
                self.rebuild_count.fetch_add(1, Ordering::SeqCst);
                state
            }
        };
        #[cfg(test)]
        if self.panic_next.swap(false, Ordering::SeqCst) {
            panic!("injected panic inside process collector state");
        }
        if !state.primed {
            // Establish a CPU baseline, then sample a real delta for the first
            // response. Later scrapes reuse this System's CPU history.
            state.system.refresh_all();
            std::thread::sleep(sysinfo::MINIMUM_CPU_UPDATE_INTERVAL);
            state.system.refresh_all();
            state.primed = true;
        } else {
            state.system.refresh_all();
        }

        let mut matches = BTreeMap::<&str, ProcessTotals>::new();

        for process in state.system.processes().values() {
            let process_name = process.name().to_string_lossy();
            for pattern in &self.patterns {
                if matches_pattern(&process_name, pattern) {
                    let entry = matches.entry(pattern).or_default();
                    entry.cpu_percent += f64::from(process.cpu_usage()).max(0.0);
                    entry.rss_bytes += process.memory();
                    if let Some(thread_count) = process_thread_count(process.pid().as_u32()) {
                        if entry.thread_count_known {
                            entry.thread_count += thread_count;
                        }
                    } else {
                        entry.thread_count_known = false;
                    }
                    if let Some(fd_count) = process.open_files() {
                        if entry.fd_count_known {
                            entry.fd_count += fd_count;
                        }
                    } else {
                        entry.fd_count_known = false;
                    }
                }
            }
        }

        let mut output = String::new();
        write_type(&mut output, "proc_cpu_percent", "gauge");
        write_type(&mut output, "proc_rss_bytes", "gauge");
        write_type(&mut output, "proc_thread_count", "gauge");
        write_type(&mut output, "proc_fd_count", "gauge");
        for (pattern, totals) in matches {
            write_process_metrics(&mut output, pattern, &totals);
        }
        output
    }

    #[cfg(test)]
    pub fn panic_inside_state_once(&self) {
        self.panic_next.store(true, Ordering::SeqCst);
    }

    #[cfg(test)]
    pub fn rebuild_count(&self) -> usize {
        self.rebuild_count.load(Ordering::SeqCst)
    }
}

fn write_process_metrics(output: &mut String, pattern: &str, totals: &ProcessTotals) {
    let labels = [("name", pattern)];
    write_metric(output, "proc_cpu_percent", &labels, totals.cpu_percent);
    write_metric(output, "proc_rss_bytes", &labels, totals.rss_bytes);
    if totals.thread_count_known {
        write_metric(output, "proc_thread_count", &labels, totals.thread_count);
    }
    if totals.fd_count_known {
        write_metric(output, "proc_fd_count", &labels, totals.fd_count);
    }
}

struct ProcessTotals {
    cpu_percent: f64,
    rss_bytes: u64,
    thread_count: usize,
    thread_count_known: bool,
    fd_count: usize,
    fd_count_known: bool,
}

impl Default for ProcessTotals {
    fn default() -> Self {
        Self {
            cpu_percent: 0.0,
            rss_bytes: 0,
            thread_count: 0,
            thread_count_known: true,
            fd_count: 0,
            fd_count_known: true,
        }
    }
}

#[cfg(target_os = "linux")]
fn process_thread_count(pid: u32) -> Option<usize> {
    let status = fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
    status
        .lines()
        .find_map(|line| line.strip_prefix("Threads:")?.trim().parse().ok())
}

#[cfg(target_os = "macos")]
fn process_thread_count(_: u32) -> Option<usize> {
    // sysinfo's tasks view is Linux-only. macOS thread counts require a
    // libproc FFI call; omit the series rather than export a false zero.
    None
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn process_thread_count(_: u32) -> Option<usize> {
    // No portable process-thread API is available through the minimal
    // dependency set; absence means unknown to Prometheus consumers.
    None
}

/// Plain patterns match substrings; `*` and `?` provide simple glob matching.
pub fn matches_pattern(name: &str, pattern: &str) -> bool {
    let name = name.to_lowercase();
    let pattern = pattern.to_lowercase();
    if !pattern.contains(['*', '?']) {
        return name.contains(&pattern);
    }
    glob_matches(name.as_bytes(), pattern.as_bytes())
}

fn glob_matches(name: &[u8], pattern: &[u8]) -> bool {
    let mut name_index = 0;
    let mut pattern_index = 0;
    let mut star = None;
    let mut star_name_index = 0;

    while name_index < name.len() {
        if pattern_index < pattern.len()
            && (pattern[pattern_index] == b'?' || pattern[pattern_index] == name[name_index])
        {
            name_index += 1;
            pattern_index += 1;
        } else if pattern_index < pattern.len() && pattern[pattern_index] == b'*' {
            star = Some(pattern_index);
            pattern_index += 1;
            star_name_index = name_index;
        } else if let Some(star_index) = star {
            pattern_index = star_index + 1;
            star_name_index += 1;
            name_index = star_name_index;
        } else {
            return false;
        }
    }
    while pattern_index < pattern.len() && pattern[pattern_index] == b'*' {
        pattern_index += 1;
    }
    pattern_index == pattern.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_patterns_match_substrings_case_insensitively() {
        assert!(matches_pattern("CodexWorker", "codex"));
        assert!(matches_pattern("codex", "CODEX"));
        assert!(!matches_pattern("python", "codex"));
    }

    #[test]
    fn glob_patterns_match_process_names() {
        assert!(matches_pattern("codex-worker", "codex-*"));
        assert!(matches_pattern("codex-worker", "c?dex*"));
        assert!(!matches_pattern("codex-worker", "claude-*"));
    }

    #[test]
    fn consecutive_scrapes_report_cpu_for_a_busy_process() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let running = Arc::new(AtomicBool::new(true));
        let worker_running = Arc::clone(&running);
        let worker = std::thread::spawn(move || {
            let mut value = 0_u64;
            while worker_running.load(Ordering::Relaxed) {
                value = value.wrapping_add(1);
                std::hint::black_box(value);
            }
        });
        let executable = std::env::current_exe().unwrap();
        let pattern = executable
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let collector = ProcCollector::new(vec![pattern.clone()]);
        let first = collector.scrape();
        let second = collector.scrape();
        running.store(false, Ordering::Relaxed);
        worker.join().unwrap();

        let metric = format!("proc_cpu_percent{{name=\"{pattern}\"}} ");
        let value = |output: &str| {
            output
                .lines()
                .find(|line| line.starts_with(&metric))
                .and_then(|line| line.split_whitespace().last())
                .and_then(|value| value.parse::<f64>().ok())
                .unwrap_or(0.0)
        };
        assert!(
            value(&first) > 0.0,
            "first scrape had no CPU delta: {first}"
        );
        assert!(
            value(&second) > 0.0,
            "second scrape had no CPU delta: {second}"
        );
    }

    #[test]
    fn current_test_process_has_nonzero_rss() {
        let executable = std::env::current_exe().unwrap();
        let pattern = executable
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let output = ProcCollector::new(vec![pattern.clone()]).scrape();
        let expected = format!("proc_rss_bytes{{name=\"{pattern}\"}} ");
        let line = output
            .lines()
            .find(|line| line.starts_with(&expected))
            .unwrap();
        let rss: u64 = line.split_whitespace().last().unwrap().parse().unwrap();
        assert!(rss > 0, "current test process should have RSS: {output}");
    }

    #[test]
    fn fd_series_are_present_when_known_and_omitted_when_unknown() {
        let known = ProcessTotals {
            cpu_percent: 1.0,
            rss_bytes: 2,
            thread_count: 3,
            thread_count_known: true,
            fd_count: 4,
            fd_count_known: true,
        };
        let unknown = ProcessTotals {
            fd_count_known: false,
            ..known
        };
        let mut output = String::new();
        write_process_metrics(&mut output, "worker", &known);
        assert!(output.contains("proc_fd_count{name=\"worker\"} 4"));
        output.clear();
        write_process_metrics(&mut output, "worker", &unknown);
        assert!(!output.contains("proc_fd_count"));
    }
}
