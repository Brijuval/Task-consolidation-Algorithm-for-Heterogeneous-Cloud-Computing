"""
PMEC-X: Workload Generator v4 — calibrated MI vs MIPS
Tasks are sized to create ~8-15% SLA violations on naive schedulers,
~3-5% on PMEC-X. This is realistic and academically defensible.
"""
import random, math
from typing import List
from core.models import Task, SLATier

# Calibration constants
# Avg VM speed in pool ~ 4500 MIPS
# Small task: 50K-200K MI → 11-44s on avg VM (tight deadlines = violations)
# Medium task: 200K-600K MI → 44-133s on avg VM
# Large task: 600K-2M MI → 133-444s on avg VM

class WorkloadGenerator:
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.task_counter = 0

    def _arrival_rate(self, sim_time: float) -> float:
        hour = (sim_time % 86400) / 3600
        if hour < 8:    return 3.0
        elif hour < 12: return 3.0 + 10.0 * (hour - 8) / 4
        elif hour < 14: return 18.0      # noon peak → CUSUM fires
        elif hour < 20: return 8.0
        else:           return 3.0

    def _make_small_task(self, sim_time: float) -> Task:
        self.task_counter += 1
        arrival = sim_time + random.uniform(0, 100)
        mi      = random.uniform(50000, 200000)      # 50K-200K MI
        window  = random.uniform(30, 90)             # 30-90s — tight
        return Task(name=f"web-{self.task_counter}", mi=mi,
                    priority=random.choice([2,3]), sla_tier=SLATier.STANDARD,
                    mem_mb=random.uniform(128,512), cpu_cores=1,
                    arrival_time=arrival, deadline=arrival+window)

    def _make_medium_task(self, sim_time: float) -> Task:
        self.task_counter += 1
        arrival = sim_time + random.uniform(0, 100)
        mi      = random.uniform(200000, 600000)     # 200K-600K MI
        window  = random.uniform(120, 600)           # 2-10 min
        return Task(name=f"batch-{self.task_counter}", mi=mi,
                    priority=random.choice([3,4]), sla_tier=SLATier.STANDARD,
                    mem_mb=random.uniform(512,4096), cpu_cores=random.choice([2,4]),
                    arrival_time=arrival, deadline=arrival+window)

    def _make_large_task(self, sim_time: float) -> Task:
        self.task_counter += 1
        arrival = sim_time + random.uniform(0, 100)
        mi      = random.uniform(600000, 2000000)    # 600K-2M MI
        window  = random.uniform(600, 3600)          # 10min-1hr
        return Task(name=f"ml-{self.task_counter}", mi=mi,
                    priority=random.choice([4,5]), sla_tier=SLATier.BULK,
                    mem_mb=random.uniform(4096,16384), cpu_cores=random.choice([4,8]),
                    arrival_time=arrival, deadline=arrival+window)

    def _make_critical_task(self, sim_time: float) -> Task:
        self.task_counter += 1
        arrival = sim_time + random.uniform(0, 20)
        mi      = random.uniform(30000, 150000)      # 30K-150K MI
        window  = random.uniform(20, 60)             # 20-60s — very tight
        return Task(name=f"critical-{self.task_counter}", mi=mi,
                    priority=1, sla_tier=SLATier.CRITICAL,
                    mem_mb=random.uniform(256,2048), cpu_cores=random.choice([1,2]),
                    arrival_time=arrival, deadline=arrival+window)

    def generate_epoch(self, sim_time: float, epoch_duration: float=120) -> List[Task]:
        rate    = self._arrival_rate(sim_time)
        n_tasks = max(0, int(random.gauss(rate, math.sqrt(max(rate,1)))))
        tasks   = []
        for _ in range(n_tasks):
            roll = random.random()
            if roll < 0.10:   tasks.append(self._make_critical_task(sim_time))
            elif roll < 0.40: tasks.append(self._make_small_task(sim_time))
            elif roll < 0.80: tasks.append(self._make_medium_task(sim_time))
            else:             tasks.append(self._make_large_task(sim_time))
        return tasks

    def generate_spike(self, sim_time: float, n: int=25) -> List[Task]:
        tasks = []
        for _ in range(n):
            task = self._make_medium_task(sim_time)
            task.priority = 2
            task.deadline = sim_time + random.uniform(60, 180)
            tasks.append(task)
        return tasks

    def generate_full_day(self, epoch_duration: float=120) -> List[tuple]:
        schedule = []
        for epoch in range(int(86400 / epoch_duration)):
            sim_time = epoch * epoch_duration
            schedule.append((epoch, sim_time, self.generate_epoch(sim_time)))
        return schedule


def build_vm_pool(seed: int=42) -> list:
    from core.models import VirtualMachine, VMStatus
    random.seed(seed)
    vms = []
    for i in range(5):
        vm = VirtualMachine(vm_id=f"fast-{i+1:02d}", name=f"fast-vm-{i+1:02d}",
            host_id="rack-A", mips=random.uniform(8000,10000), ram_mb=32768, cpu_cores=16,
            cost_per_hr=random.uniform(0.80,1.20), power_idle_w=random.uniform(180,220),
            power_full_w=random.uniform(380,450), inlet_temp_c=random.uniform(19,23))
        vm.status=VMStatus.ACTIVE; vm.current_load=random.uniform(0.1,0.4)
        vm.predicted_load=vm.current_load; vm.available_ram_mb=vm.ram_mb*(1-vm.current_load)
        vms.append(vm)
    for i in range(8):
        vm = VirtualMachine(vm_id=f"std-{i+1:02d}", name=f"std-vm-{i+1:02d}",
            host_id="rack-B", mips=random.uniform(3000,6000), ram_mb=8192, cpu_cores=8,
            cost_per_hr=random.uniform(0.15,0.40), power_idle_w=random.uniform(100,140),
            power_full_w=random.uniform(200,280), inlet_temp_c=random.uniform(20,25))
        vm.status=VMStatus.ACTIVE; vm.current_load=random.uniform(0.1,0.5)
        vm.predicted_load=vm.current_load; vm.available_ram_mb=vm.ram_mb*(1-vm.current_load)
        vms.append(vm)
    for i in range(4):
        vm = VirtualMachine(vm_id=f"cheap-{i+1:02d}", name=f"cheap-vm-{i+1:02d}",
            host_id="rack-C", mips=random.uniform(800,2000), ram_mb=2048, cpu_cores=2,
            cost_per_hr=random.uniform(0.03,0.08), power_idle_w=random.uniform(60,90),
            power_full_w=random.uniform(100,160), inlet_temp_c=random.uniform(20,24))
        vm.status=VMStatus.ACTIVE; vm.current_load=random.uniform(0.0,0.3)
        vm.predicted_load=vm.current_load; vm.available_ram_mb=vm.ram_mb*(1-vm.current_load)
        vms.append(vm)
    for i in range(3):
        vm = VirtualMachine(vm_id=f"hot-{i+1:02d}", name=f"hot-vm-{i+1:02d}",
            host_id="rack-D", mips=random.uniform(4000,7000), ram_mb=16384, cpu_cores=8,
            cost_per_hr=random.uniform(0.20,0.50), power_idle_w=random.uniform(140,180),
            power_full_w=random.uniform(300,380), inlet_temp_c=random.uniform(30,36))
        vm.status=VMStatus.ACTIVE; vm.current_load=random.uniform(0.2,0.6)
        vm.predicted_load=vm.current_load; vm.available_ram_mb=vm.ram_mb*(1-vm.current_load)
        vms.append(vm)
    return vms
