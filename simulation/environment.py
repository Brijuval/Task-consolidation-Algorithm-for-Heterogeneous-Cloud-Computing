"""
PMEC-X: Main Simulation Environment
=====================================
environment.py

WHAT THIS FILE DOES:
  This is the full PMEC-X scheduling loop.
  Every 120 seconds (one epoch) it:
    1. Collects observed VM loads → feeds EWMA+CUSUM predictor
    2. Sorts arriving tasks by urgency
    3. Scores every (task, VM) pair → assigns best VM
    4. Runs consolidation → migrates or queues tasks from underloaded VMs
    5. Shuts down empty VMs → saves energy
    6. Records everything for the paper results

  Also runs all 4 baseline schedulers on the SAME workload
  for fair comparison.
"""

import random
import time
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from core.models import (
    Task, VirtualMachine, TaskStatus, VMStatus,
    SchedulingDecision, SLATier
)
from core.predictor import HybridPredictor
from core.scorer import PMECXScorer, WeightVector, NormalizationContext


# ─────────────────────────────────────────────────────────────────────────────
# METRICS COLLECTOR
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EpochMetrics:
    """Everything we measure in one epoch — becomes a row in result tables."""
    epoch:            int   = 0
    sim_time:         float = 0.0
    tasks_arrived:    int   = 0
    tasks_assigned:   int   = 0
    tasks_queued:     int   = 0
    tasks_completed:  int   = 0
    tasks_violated:   int   = 0
    vms_active:       int   = 0
    vms_shutdown:     int   = 0
    vms_migrated:     int   = 0
    energy_kwh:       float = 0.0
    avg_score:        float = 0.0
    regime_changes:   int   = 0
    carbon_g:         float = 0.0     # gCO₂ emitted this epoch


@dataclass
class SimulationResults:
    """Aggregated results across the full simulation run."""
    scheduler_name:     str
    total_tasks:        int   = 0
    completed_tasks:    int   = 0
    violated_tasks:     int   = 0
    total_energy_kwh:   float = 0.0
    total_carbon_g:     float = 0.0
    avg_active_vms:     float = 0.0
    total_migrations:   int   = 0
    total_shutdowns:    int   = 0
    regime_changes:     int   = 0
    epoch_metrics:      List[EpochMetrics] = field(default_factory=list)

    @property
    def sla_violation_rate(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return self.violated_tasks / self.total_tasks * 100

    @property
    def completion_rate(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return self.completed_tasks / self.total_tasks * 100

    def summary(self) -> str:
        return (
            f"\n{'='*55}\n"
            f"  Scheduler:        {self.scheduler_name}\n"
            f"{'='*55}\n"
            f"  Total tasks:      {self.total_tasks}\n"
            f"  Completed:        {self.completed_tasks} "
            f"({self.completion_rate:.1f}%)\n"
            f"  SLA violations:   {self.violated_tasks} "
            f"({self.sla_violation_rate:.1f}%)\n"
            f"  Total energy:     {self.total_energy_kwh:.3f} kWh\n"
            f"  Carbon emitted:   {self.total_carbon_g:.1f} gCO₂\n"
            f"  Avg active VMs:   {self.avg_active_vms:.1f}\n"
            f"  VM shutdowns:     {self.total_shutdowns}\n"
            f"  Migrations:       {self.total_migrations}\n"
            f"  Regime changes:   {self.regime_changes}\n"
            f"{'='*55}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# PMEC-X SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────

class PMECXScheduler:
    """
    The full PMEC-X scheduling algorithm.

    HOW ONE EPOCH WORKS:
      Step 1 — PREDICT:  Update EWMA+CUSUM for every VM
      Step 2 — SORT:     Order tasks by urgency score (highest first)
      Step 3 — ASSIGN:   For each task, score all VMs, pick best
      Step 4 — CONSOLIDATE: Migrate tasks off underloaded VMs
      Step 5 — SHUTDOWN: Power down empty VMs
      Step 6 — COMPLETE: Mark finished tasks, record metrics
    """

    def __init__(self, vm_pool: List[VirtualMachine],
                 weights: Optional[WeightVector] = None,
                 alpha: float = 0.3):

        self.vms        = {vm.vm_id: vm for vm in vm_pool}
        self.scorer     = PMECXScorer(weights or WeightVector())
        self.predictor  = HybridPredictor(alpha=alpha)
        self.results    = SimulationResults(scheduler_name="PMEC-X")

        # Queues
        self.wait_queue: List[Task] = []
        self.all_tasks:  Dict[str, Task] = {}

        # Carbon scores (mock — Day 2 plugs in real API)
        # Different VMs on different racks get different carbon scores
        self._carbon_scores = self._init_carbon_scores()

    def _init_carbon_scores(self) -> Dict[str, float]:
        """
        Mock carbon intensity per VM.
        In Day 2 this gets replaced with live WattTime API data.
        Rack A = renewable heavy (0.2), Rack D = coal heavy (0.8)
        """
        scores = {}
        for vm_id, vm in self.vms.items():
            if "fast" in vm_id:
                scores[vm_id] = 0.25    # Rack A — mostly renewable
            elif "std" in vm_id:
                scores[vm_id] = 0.50    # Rack B — grid average
            elif "cheap" in vm_id:
                scores[vm_id] = 0.45    # Rack C — slightly better
            else:
                scores[vm_id] = 0.75    # Rack D — hot + dirty
        return scores

    # ── Step 1: Predict ───────────────────────────────────────────────────────

    def _predict(self, epoch: int) -> int:
        """
        Update EWMA+CUSUM for all VMs.
        Returns number of VMs with regime change alarms.
        """
        vm_loads = {vm_id: vm.current_load
                    for vm_id, vm in self.vms.items()
                    if vm.status != VMStatus.SHUTDOWN}

        results = self.predictor.update_all(vm_loads)

        # Push predictions back into VM objects
        for vm_id, r in results.items():
            if vm_id in self.vms:
                self.vms[vm_id].predicted_load = r['prediction']
                self.vms[vm_id].ewma_load       = r['ewma_estimate']

        return len(self.predictor.get_regime_change_vms())

    # ── Step 2: Sort ─────────────────────────────────────────────────────────

    def _sort_tasks(self, tasks: List[Task]) -> List[Task]:
        """Sort by urgency score descending — most urgent first."""
        return sorted(tasks, key=lambda t: t.urgency_score, reverse=True)

    # ── Step 3: Assign ────────────────────────────────────────────────────────

    def _assign(self, tasks: List[Task], sim_time: float) -> Tuple[int, int]:
        """
        Core assignment loop — O(n × m).
        Returns (assigned_count, queued_count).
        """
        ctx = NormalizationContext.from_vm_pool(list(self.vms.values()))
        active_vms = [vm for vm in self.vms.values()
                      if vm.status not in (VMStatus.SHUTDOWN, VMStatus.OVERLOADED)]

        assigned = 0
        queued   = 0

        for task in tasks:
            # Skip if dependencies not met
            if not task.is_ready:
                task.status = TaskStatus.WAITING
                self.wait_queue.append(task)
                queued += 1
                continue

            best_vm, best_score, factors = self.scorer.best_vm(
                task, active_vms, ctx, self._carbon_scores
            )

            # Fallback: if feasibility filter blocked all VMs,
            # assign to fastest available VM as last resort.
            # Task may still violate but at least it gets executed.
            if best_vm is None and active_vms:
                best_vm = max(active_vms, key=lambda v: v.mips)
                best_score = 0.01

            if best_vm is None:
                # No VM can fit this task right now — queue it
                task.status = TaskStatus.QUEUED
                self.wait_queue.append(task)
                queued += 1
                continue

            # Assign task to best VM
            exec_time = task.exec_time_on(best_vm.mips)
            task.status         = TaskStatus.RUNNING
            task.assigned_vm_id = best_vm.vm_id
            task.start_time     = sim_time
            task.finish_time    = sim_time + exec_time

            # Update VM state
            load_delta = min(0.4, task.mi / (best_vm.mips * 200))
            best_vm.current_load   = min(0.95, best_vm.current_load + load_delta)
            best_vm.available_ram_mb = max(0, best_vm.available_ram_mb - task.mem_mb)
            best_vm.status         = VMStatus.ACTIVE
            best_vm.running_tasks.append(task.task_id)

            self.all_tasks[task.task_id] = task
            assigned += 1

        return assigned, queued

    # ── Step 4: Consolidate ───────────────────────────────────────────────────

    def _consolidate(self, sim_time: float) -> Tuple[int, int]:
        """
        Hybrid consolidation — migrate or queue tasks off underloaded VMs.
        Returns (migrations, shutdowns).
        """
        migrations = 0
        shutdowns  = 0

        ctx = NormalizationContext.from_vm_pool(list(self.vms.values()))
        active_vms = [vm for vm in self.vms.values()
                      if vm.status not in (VMStatus.SHUTDOWN,)]

        for vm in list(self.vms.values()):
            if not vm.is_underloaded:
                continue
            if not vm.running_tasks:
                # VM is underloaded AND empty → shut it down
                vm.status = VMStatus.SHUTDOWN
                shutdowns += 1
                continue

            # VM is underloaded but has tasks — try to migrate them off
            all_migrated = True
            for task_id in list(vm.running_tasks):
                task = self.all_tasks.get(task_id)
                if task is None:
                    continue

                # SLA-tier-aware consolidation (documented in paper Section 4.4)
                # CRITICAL and STANDARD tasks are never migrated or queued —
                # their deadlines are too tight to survive the overhead.
                # Only BULK tasks are eligible for consolidation.
                if task.sla_tier != SLATier.BULK:
                    all_migrated = False
                    continue

                slack_budget = task.slack_budget_on(vm.mips, sim_time)
                mig_cost     = vm.ram_mb / 1000    # simplified migration cost (seconds)

                if slack_budget >= mig_cost:
                    # Enough slack to migrate
                    other_vms = [v for v in active_vms if v.vm_id != vm.vm_id]
                    dest_vm, score, _ = self.scorer.best_vm(
                        task, other_vms, ctx, self._carbon_scores
                    )
                    if dest_vm and score > 0:
                        # Migrate
                        vm.running_tasks.remove(task_id)
                        load_delta = min(0.4, task.mi / (vm.mips * 200))
                        vm.current_load = max(0, vm.current_load - load_delta)
                        vm.available_ram_mb += task.mem_mb

                        task.assigned_vm_id = dest_vm.vm_id
                        task.migrations    += 1
                        dest_vm.running_tasks.append(task_id)
                        dest_vm.current_load = min(0.95, dest_vm.current_load + load_delta)
                        dest_vm.available_ram_mb = max(0, dest_vm.available_ram_mb - task.mem_mb)
                        migrations += 1
                    else:
                        all_migrated = False

                elif slack_budget >= 0:
                    # Queue the task (SLA still safe)
                    vm.running_tasks.remove(task_id)
                    task.status = TaskStatus.QUEUED
                    self.wait_queue.append(task)
                    task.queue_entries += 1
                else:
                    # SLA critical — must stay
                    all_migrated = False

            # If all tasks migrated off, shut this VM down
            if all_migrated and not vm.running_tasks:
                vm.status = VMStatus.SHUTDOWN
                shutdowns += 1

        return migrations, shutdowns

    # ── Step 5: Complete tasks ────────────────────────────────────────────────

    def _complete_tasks(self, sim_time: float) -> Tuple[int, int]:
        """
        Mark tasks that have finished by this sim_time.
        Returns (completed, violated).
        """
        completed = 0
        violated  = 0

        for task in list(self.all_tasks.values()):
            if task.status != TaskStatus.RUNNING:
                continue
            if task.finish_time and sim_time >= task.finish_time:
                if task.finish_time > task.deadline:
                    task.status = TaskStatus.VIOLATED
                    violated += 1
                else:
                    task.status = TaskStatus.COMPLETED
                    completed += 1

                # Free VM resources
                vm = self.vms.get(task.assigned_vm_id)
                if vm:
                    load_delta = min(0.4, task.mi / (vm.mips * 200))
                    vm.current_load = max(0, vm.current_load - load_delta)
                    vm.available_ram_mb = min(vm.ram_mb,
                                              vm.available_ram_mb + task.mem_mb)
                    if task.task_id in vm.running_tasks:
                        vm.running_tasks.remove(task.task_id)

                    # Idle VM if no more tasks
                    if not vm.running_tasks and vm.status == VMStatus.ACTIVE:
                        vm.status = VMStatus.IDLE

        return completed, violated

    # ── Main epoch runner ─────────────────────────────────────────────────────

    def run_epoch(self, epoch: int, sim_time: float,
                  new_tasks: List[Task]) -> EpochMetrics:
        """Run one complete scheduling epoch. Returns metrics."""
        m = EpochMetrics(epoch=epoch, sim_time=sim_time,
                         tasks_arrived=len(new_tasks))

        # Include waited tasks from previous epoch
        all_tasks_this_epoch = self.wait_queue + new_tasks
        self.wait_queue = []
        for t in new_tasks:
            self.all_tasks[t.task_id] = t

        # Purge tasks from queue whose deadline has already passed
        fresh_tasks = []
        for t in all_tasks_this_epoch:
            if t.finish_time is None and sim_time > t.deadline:
                t.status = TaskStatus.VIOLATED
                m.tasks_violated += 1
            else:
                fresh_tasks.append(t)
        all_tasks_this_epoch = fresh_tasks

        # Step 1: Predict
        m.regime_changes = self._predict(epoch)

        # Step 2: Sort by urgency
        sorted_tasks = self._sort_tasks(all_tasks_this_epoch)

        # Step 3: Assign
        m.tasks_assigned, m.tasks_queued = self._assign(sorted_tasks, sim_time)

        # Step 4: Consolidate
        m.vms_migrated, m.vms_shutdown = self._consolidate(sim_time)

        # Step 5: Complete tasks
        m.tasks_completed, m.tasks_violated = self._complete_tasks(sim_time)

        # Step 6: Measure energy
        epoch_hours = 120 / 3600   # 120-second epoch in hours
        active_vms  = [vm for vm in self.vms.values()
                       if vm.status != VMStatus.SHUTDOWN]
        m.vms_active  = len(active_vms)
        m.energy_kwh  = sum(vm.current_power_w * epoch_hours / 1000
                            for vm in active_vms)

        # Carbon: energy × grid intensity (average 450 gCO₂/kWh)
        m.carbon_g = m.energy_kwh * 450

        # Update totals
        self.results.total_tasks      += m.tasks_arrived
        self.results.completed_tasks  += m.tasks_completed
        self.results.violated_tasks   += m.tasks_violated
        self.results.total_energy_kwh += m.energy_kwh
        self.results.total_carbon_g   += m.carbon_g
        self.results.total_migrations += m.vms_migrated
        self.results.total_shutdowns  += m.vms_shutdown
        self.results.regime_changes   += m.regime_changes
        self.results.epoch_metrics.append(m)

        return m

    def run_full_simulation(self, workload_schedule: List[tuple],
                             verbose: bool = True) -> SimulationResults:
        """
        Run the complete simulation on a pre-generated workload.

        Args:
            workload_schedule: List of (epoch, sim_time, tasks) from WorkloadGenerator
            verbose:           Print progress every 60 epochs

        Returns:
            SimulationResults with all metrics
        """
        total_epochs = len(workload_schedule)
        active_vm_counts = []

        for epoch, sim_time, tasks in workload_schedule:
            metrics = self.run_epoch(epoch, sim_time, tasks)
            active_vm_counts.append(metrics.vms_active)

            if verbose and epoch % 60 == 0:
                hour = sim_time / 3600
                print(f"  [{self.results.scheduler_name}] "
                      f"Epoch {epoch:3d}/720 | "
                      f"Hour {hour:4.1f} | "
                      f"Tasks: +{metrics.tasks_arrived} "
                      f"done={metrics.tasks_completed} "
                      f"viol={metrics.tasks_violated} | "
                      f"VMs: {metrics.vms_active} active | "
                      f"Energy: {metrics.energy_kwh:.4f} kWh")

        self.results.avg_active_vms = (sum(active_vm_counts) /
                                        len(active_vm_counts) if active_vm_counts else 0)
        return self.results


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE SCHEDULERS (for comparison)
# ─────────────────────────────────────────────────────────────────────────────

class RoundRobinScheduler:
    """
    Baseline 1: Round Robin
    Assigns tasks to VMs in a fixed rotating order.
    Ignores energy, SLA, thermal, carbon — everything.
    Simplest possible scheduler.
    """
    def __init__(self, vm_pool: List[VirtualMachine]):
        self.vms     = [vm for vm in vm_pool]
        self.index   = 0
        self.results = SimulationResults(scheduler_name="Round Robin")
        self.all_tasks: Dict[str, Task] = {}

    def run_epoch(self, epoch: int, sim_time: float,
                  new_tasks: List[Task]) -> EpochMetrics:
        m = EpochMetrics(epoch=epoch, sim_time=sim_time,
                         tasks_arrived=len(new_tasks))
        active = [vm for vm in self.vms if vm.status != VMStatus.SHUTDOWN]
        if not active:
            return m

        for task in new_tasks:
            vm = active[self.index % len(active)]
            self.index += 1
            exec_time           = task.exec_time_on(vm.mips)
            task.status         = TaskStatus.RUNNING
            task.assigned_vm_id = vm.vm_id
            task.start_time     = sim_time
            task.finish_time    = sim_time + exec_time
            vm.current_load     = min(0.95, vm.current_load + 0.05)
            self.all_tasks[task.task_id] = task
            m.tasks_assigned += 1

        # Complete tasks
        for task in self.all_tasks.values():
            if task.status == TaskStatus.RUNNING and task.finish_time:
                if sim_time >= task.finish_time:
                    if task.finish_time > task.deadline:
                        task.status = TaskStatus.VIOLATED
                        m.tasks_violated += 1
                    else:
                        task.status = TaskStatus.COMPLETED
                        m.tasks_completed += 1

        epoch_hours   = 120 / 3600
        m.vms_active  = len(active)
        m.energy_kwh  = sum(vm.current_power_w * epoch_hours / 1000
                            for vm in active)
        m.carbon_g    = m.energy_kwh * 450

        self.results.total_tasks      += m.tasks_arrived
        self.results.completed_tasks  += m.tasks_completed
        self.results.violated_tasks   += m.tasks_violated
        self.results.total_energy_kwh += m.energy_kwh
        self.results.total_carbon_g   += m.carbon_g
        self.results.epoch_metrics.append(m)
        return m

    def run_full_simulation(self, workload_schedule, verbose=True):
        active_counts = []
        for epoch, sim_time, tasks in workload_schedule:
            m = self.run_epoch(epoch, sim_time, tasks)
            active_counts.append(m.vms_active)
            if verbose and epoch % 60 == 0:
                hour = sim_time / 3600
                print(f"  [Round Robin]  Epoch {epoch:3d}/720 | "
                      f"Hour {hour:4.1f} | "
                      f"Tasks: +{m.tasks_arrived} "
                      f"done={m.tasks_completed} viol={m.tasks_violated} | "
                      f"VMs: {m.vms_active} active")
        self.results.avg_active_vms = (sum(active_counts) /
                                        len(active_counts) if active_counts else 0)
        return self.results


class MinMinScheduler:
    """
    Baseline 2: MinMin
    Always assigns task to the VM that gives minimum completion time.
    Greedy speed-only optimization. No energy awareness.
    """
    def __init__(self, vm_pool: List[VirtualMachine]):
        self.vms     = {vm.vm_id: vm for vm in vm_pool}
        self.results = SimulationResults(scheduler_name="MinMin")
        self.all_tasks: Dict[str, Task] = {}

    def run_epoch(self, epoch: int, sim_time: float,
                  new_tasks: List[Task]) -> EpochMetrics:
        m = EpochMetrics(epoch=epoch, sim_time=sim_time,
                         tasks_arrived=len(new_tasks))
        active = [vm for vm in self.vms.values()
                  if vm.status != VMStatus.SHUTDOWN]
        if not active:
            return m

        for task in new_tasks:
            # Pick VM with minimum execution time
            best_vm = min(active, key=lambda vm: task.exec_time_on(vm.mips))
            exec_time           = task.exec_time_on(best_vm.mips)
            task.status         = TaskStatus.RUNNING
            task.assigned_vm_id = best_vm.vm_id
            task.start_time     = sim_time
            task.finish_time    = sim_time + exec_time
            best_vm.current_load = min(0.95, best_vm.current_load + 0.06)
            self.all_tasks[task.task_id] = task
            m.tasks_assigned += 1

        for task in self.all_tasks.values():
            if task.status == TaskStatus.RUNNING and task.finish_time:
                if sim_time >= task.finish_time:
                    if task.finish_time > task.deadline:
                        task.status = TaskStatus.VIOLATED
                        m.tasks_violated += 1
                    else:
                        task.status = TaskStatus.COMPLETED
                        m.tasks_completed += 1

        epoch_hours   = 120 / 3600
        m.vms_active  = len(active)
        m.energy_kwh  = sum(vm.current_power_w * epoch_hours / 1000
                            for vm in active)
        m.carbon_g    = m.energy_kwh * 450

        self.results.total_tasks      += m.tasks_arrived
        self.results.completed_tasks  += m.tasks_completed
        self.results.violated_tasks   += m.tasks_violated
        self.results.total_energy_kwh += m.energy_kwh
        self.results.total_carbon_g   += m.carbon_g
        self.results.epoch_metrics.append(m)
        return m

    def run_full_simulation(self, workload_schedule, verbose=True):
        active_counts = []
        for epoch, sim_time, tasks in workload_schedule:
            m = self.run_epoch(epoch, sim_time, tasks)
            active_counts.append(m.vms_active)
            if verbose and epoch % 60 == 0:
                hour = sim_time / 3600
                print(f"  [MinMin]       Epoch {epoch:3d}/720 | "
                      f"Hour {hour:4.1f} | "
                      f"Tasks: +{m.tasks_arrived} "
                      f"done={m.tasks_completed} viol={m.tasks_violated} | "
                      f"VMs: {m.vms_active} active")
        self.results.avg_active_vms = (sum(active_counts) /
                                        len(active_counts) if active_counts else 0)
        return self.results


class FCFSScheduler:
    """
    Baseline 3: First Come First Served
    Tasks assigned in arrival order to first available VM.
    No priority, no energy, no SLA awareness.
    """
    def __init__(self, vm_pool: List[VirtualMachine]):
        self.vms     = list(vm_pool)
        self.results = SimulationResults(scheduler_name="FCFS")
        self.all_tasks: Dict[str, Task] = {}

    def run_epoch(self, epoch: int, sim_time: float,
                  new_tasks: List[Task]) -> EpochMetrics:
        m = EpochMetrics(epoch=epoch, sim_time=sim_time,
                         tasks_arrived=len(new_tasks))
        active = [vm for vm in self.vms if vm.status != VMStatus.SHUTDOWN]
        if not active:
            return m

        for task in new_tasks:
            # First VM with headroom
            assigned = False
            for vm in active:
                if vm.current_load < 0.85 and vm.can_fit(task):
                    exec_time           = task.exec_time_on(vm.mips)
                    task.status         = TaskStatus.RUNNING
                    task.assigned_vm_id = vm.vm_id
                    task.start_time     = sim_time
                    task.finish_time    = sim_time + exec_time
                    vm.current_load     = min(0.95, vm.current_load + 0.05)
                    self.all_tasks[task.task_id] = task
                    m.tasks_assigned += 1
                    assigned = True
                    break
            if not assigned:
                m.tasks_queued += 1

        for task in self.all_tasks.values():
            if task.status == TaskStatus.RUNNING and task.finish_time:
                if sim_time >= task.finish_time:
                    if task.finish_time > task.deadline:
                        task.status = TaskStatus.VIOLATED
                        m.tasks_violated += 1
                    else:
                        task.status = TaskStatus.COMPLETED
                        m.tasks_completed += 1

        epoch_hours   = 120 / 3600
        m.vms_active  = len(active)
        m.energy_kwh  = sum(vm.current_power_w * epoch_hours / 1000
                            for vm in active)
        m.carbon_g    = m.energy_kwh * 450

        self.results.total_tasks      += m.tasks_arrived
        self.results.completed_tasks  += m.tasks_completed
        self.results.violated_tasks   += m.tasks_violated
        self.results.total_energy_kwh += m.energy_kwh
        self.results.total_carbon_g   += m.carbon_g
        self.results.epoch_metrics.append(m)
        return m

    def run_full_simulation(self, workload_schedule, verbose=True):
        active_counts = []
        for epoch, sim_time, tasks in workload_schedule:
            m = self.run_epoch(epoch, sim_time, tasks)
            active_counts.append(m.vms_active)
            if verbose and epoch % 60 == 0:
                hour = sim_time / 3600
                print(f"  [FCFS]         Epoch {epoch:3d}/720 | "
                      f"Hour {hour:4.1f} | "
                      f"Tasks: +{m.tasks_arrived} "
                      f"done={m.tasks_completed} viol={m.tasks_violated} | "
                      f"VMs: {m.vms_active} active")
        self.results.avg_active_vms = (sum(active_counts) /
                                        len(active_counts) if active_counts else 0)
        return self.results
