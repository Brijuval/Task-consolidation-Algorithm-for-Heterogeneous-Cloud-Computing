from dataclasses import dataclass
from core.models import Task, VirtualMachine, VMStatus, TaskStatus

@dataclass
class EpochMetrics:
    epoch: int
    tasks_arrived: int = 0
    tasks_completed: int = 0
    tasks_violated: int = 0
    energy_kwh: float = 0.0
    vms_active: int = 0
    regime_changes: int = 0

class FCFSScheduler:
    def __init__(self, vm_pool):
        self.vms = list(vm_pool)
        self.epoch_count = 0

    def run_epoch(self, epoch, sim_time, tasks):
        m = EpochMetrics(epoch=epoch, tasks_arrived=len(tasks))
        for task in tasks:
            assigned = False
            for vm in self.vms:
                if vm.current_load < 0.95:
                    vm.current_load = min(1.0, vm.current_load + 0.05)
                    exec_time = task.exec_time_on(vm.mips) if vm.mips > 0 else 9999
                    deadline_window = max(1, task.deadline - sim_time)
                    if exec_time > deadline_window:
                        m.tasks_violated += 1
                    else:
                        m.tasks_completed += 1
                    assigned = True
                    break
            if not assigned:
                m.tasks_violated += 1
        total_energy = 0.0
        for vm in self.vms:
            p = vm.power_idle_w + (vm.power_full_w - vm.power_idle_w) * vm.current_load
            total_energy += p * 120 / 3600 / 1000
            vm.current_load = max(0.0, vm.current_load - 0.08)
        m.energy_kwh = total_energy
        m.vms_active = len(self.vms)
        self.epoch_count += 1
        return m
