use sysinfo::{Disks, Networks, System};

use crate::{write_metric, write_type};

/// Collects host metrics for `gauge-node`.
pub struct NodeCollector {
    system: System,
    disks: Disks,
    networks: Networks,
}

impl NodeCollector {
    pub fn new() -> Self {
        let mut system = System::new_all();
        system.refresh_all();
        std::thread::sleep(sysinfo::MINIMUM_CPU_UPDATE_INTERVAL);
        system.refresh_all();
        Self {
            system,
            disks: Disks::new_with_refreshed_list(),
            networks: Networks::new_with_refreshed_list(),
        }
    }

    pub fn scrape(&mut self) -> String {
        self.system.refresh_all();
        self.disks.refresh(true);
        self.networks.refresh(true);

        let mut output = String::new();

        write_type(&mut output, "node_cpu_percent", "gauge");
        for (index, cpu) in self.system.cpus().iter().enumerate() {
            let usage = f64::from(cpu.cpu_usage()).clamp(0.0, 100.0);
            write_metric(
                &mut output,
                "node_cpu_percent",
                &[("cpu", &index.to_string())],
                usage,
            );
        }
        let total_cpu = f64::from(self.system.global_cpu_usage()).clamp(0.0, 100.0);
        write_metric(
            &mut output,
            "node_cpu_percent",
            &[("cpu", "total")],
            total_cpu,
        );

        write_type(&mut output, "node_memory_total_bytes", "gauge");
        write_metric(
            &mut output,
            "node_memory_total_bytes",
            &[],
            self.system.total_memory(),
        );
        write_type(&mut output, "node_memory_used_bytes", "gauge");
        write_metric(
            &mut output,
            "node_memory_used_bytes",
            &[],
            self.system.used_memory(),
        );
        write_type(&mut output, "node_memory_available_bytes", "gauge");
        write_metric(
            &mut output,
            "node_memory_available_bytes",
            &[],
            self.system.available_memory(),
        );
        write_type(&mut output, "node_memory_free_bytes", "gauge");
        write_metric(
            &mut output,
            "node_memory_free_bytes",
            &[],
            self.system.free_memory(),
        );

        write_type(&mut output, "node_swap_total_bytes", "gauge");
        write_metric(
            &mut output,
            "node_swap_total_bytes",
            &[],
            self.system.total_swap(),
        );
        write_type(&mut output, "node_swap_used_bytes", "gauge");
        write_metric(
            &mut output,
            "node_swap_used_bytes",
            &[],
            self.system.used_swap(),
        );
        write_type(&mut output, "node_swap_free_bytes", "gauge");
        write_metric(
            &mut output,
            "node_swap_free_bytes",
            &[],
            self.system.free_swap(),
        );

        let load = System::load_average();
        write_type(&mut output, "node_load1", "gauge");
        write_metric(&mut output, "node_load1", &[], load.one);
        write_type(&mut output, "node_load5", "gauge");
        write_metric(&mut output, "node_load5", &[], load.five);
        write_type(&mut output, "node_load15", "gauge");
        write_metric(&mut output, "node_load15", &[], load.fifteen);

        write_type(&mut output, "node_disk_size_bytes", "gauge");
        write_type(&mut output, "node_disk_free_bytes", "gauge");
        write_type(&mut output, "node_disk_used_bytes", "gauge");
        write_type(&mut output, "node_disk_read_bytes_total", "counter");
        write_type(&mut output, "node_disk_written_bytes_total", "counter");
        for disk in self.disks.list() {
            let device = disk.name().to_string_lossy().into_owned();
            let mountpoint = disk.mount_point().to_string_lossy().into_owned();
            let labels = [
                ("device", device.as_str()),
                ("mountpoint", mountpoint.as_str()),
            ];
            let total = disk.total_space();
            let free = disk.available_space();
            write_metric(&mut output, "node_disk_size_bytes", &labels, total);
            write_metric(&mut output, "node_disk_free_bytes", &labels, free);
            write_metric(
                &mut output,
                "node_disk_used_bytes",
                &labels,
                total.saturating_sub(free),
            );
            let usage = disk.usage();
            write_metric(
                &mut output,
                "node_disk_read_bytes_total",
                &labels,
                usage.total_read_bytes,
            );
            write_metric(
                &mut output,
                "node_disk_written_bytes_total",
                &labels,
                usage.total_written_bytes,
            );
        }

        write_type(&mut output, "node_network_receive_bytes_total", "counter");
        write_type(&mut output, "node_network_transmit_bytes_total", "counter");
        for (interface, network) in self.networks.list() {
            let labels = [("device", interface.as_str())];
            write_metric(
                &mut output,
                "node_network_receive_bytes_total",
                &labels,
                network.total_received(),
            );
            write_metric(
                &mut output,
                "node_network_transmit_bytes_total",
                &labels,
                network.total_transmitted(),
            );
        }

        output
    }
}

impl Default for NodeCollector {
    fn default() -> Self {
        Self::new()
    }
}
