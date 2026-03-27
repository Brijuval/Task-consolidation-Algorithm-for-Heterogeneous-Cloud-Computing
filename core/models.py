"""
PMEC-X: Predictive Multifactor Energy Consolidation with Cross-layer Scheduling
=================================================================================
models.py — Core data models for Tasks, Containers, VMs, and Hosts

WHAT THIS FILE DOES:
  Defines the fundamental data structures that every other module in PMEC-X
  uses. Think of this as the "blueprint" — before we can schedule anything,
  we need to agree on what a Task IS and what a VM IS.

WHY THESE FIELDS MATTER:
  Every field in Task and VM maps to one factor in our 7-factor scoring
  function. If the field doesn't contribute to a scheduling decision, it
  doesn't belong here.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set
from enum import Enum
import uuid
import time


# ─────────────────────────────────────────────────────────────────────────────
# ENUMERATIONS
# ─────────────────────────────────────────────────────────────────────────────

class TaskStatus(Enum):
    """Lifecycle states of a task through the scheduler."""
    PENDING    = "pending"       # Arrived, waiting to be scheduled
    WAITING    = "waiting"       # Dependencies not yet met
    QUEUED     = "queued"        # Consolidation deferred it (SLA still safe)
    RUNNING    = "running"       # Assigned to a VM, executing
    MIGRATING  = "migrating"     # Being moved to a different VM
    COMPLETED  = "completed"     # Finished within deadline
    VIOLATED   = "violated"      # Finished AFTER deadline (SLA violation)
    FAILED     = "failed"        # Could not be scheduled (no VM fit)


class VMStatus(Enum):
    """Operational states of a Virtual Machine."""
    ACTIVE     = "active"        # Powered on, accepting tasks
    IDLE       = "idle"          # Powered on but no tasks (costly — we shut these down)
    MIGRATING  = "migrating"     # Currently involved in a live migration
    SHUTDOWN   = "shutdown"      # Powered off (energy saved!)
    OVERLOADED = "overloaded"    # Above θ_high threshold — no new tasks


class SLATier(Enum):
    """
    SLA tier determines how aggressively PMEC-X protects a task's deadline.
    
    CRITICAL  → Never migrate, never queue. Always assign immediately.
    STANDARD  → Can queue if slack_budget is sufficient.
    BULK      → Can migrate and queue freely. Background jobs.
    """
    CRITICAL  = 1
    STANDARD  = 2
    BULK      = 3


# ─────────────────────────────────────────────────────────────────────────────
# TASK MODEL
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Task:
    """
    Represents a single computational workload unit.
    
    In a real cloud: this maps to a VM instance request, a batch job,
    a container workload, or a serverless function invocation.
    
    FIELDS EXPLAINED:
      task_id     → Unique identifier (auto-generated UUID)
      mi          → Million Instructions — how much computation this task needs.
                    Execution time on a VM = mi / vm.mips  (in seconds)
      priority    → 1 (critical) to 5 (background). Maps to urgency scoring.
      deadline    → Unix timestamp by which this task MUST finish.
                    If finish_time > deadline → SLA violation → paper metric goes up.
      arrival_time→ When this task entered the system (unix timestamp).
      sla_tier    → CRITICAL/STANDARD/BULK — controls consolidation aggressiveness.
      dependencies→ Set of task_ids that must COMPLETE before this task can start.
                    This is what makes scheduling NP-hard in the general case.
      mem_mb      → Memory requirement in MB. Used for VM headroom check.
      cpu_cores   → Number of vCPUs needed. Constrains dual-layer container placement.
    
    TRACKING FIELDS (filled in by scheduler, not by workload generator):
      status         → Current lifecycle state
      assigned_vm_id → Which VM this task is running on
      start_time     → When execution actually began
      finish_time    → When execution actually completed
      migrations     → How many times this task was migrated (we track this)
    """
    # Identity
    task_id:        str            = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name:           str            = ""

    # Workload characteristics
    mi:             float          = 1000.0    # Million Instructions
    priority:       int            = 3         # 1=highest, 5=lowest
    sla_tier:       SLATier        = SLATier.STANDARD
    mem_mb:         float          = 512.0     # Memory requirement
    cpu_cores:      int            = 1         # vCPU requirement

    # Time constraints
    arrival_time:   float          = field(default_factory=time.time)
    deadline:       float          = field(default_factory=lambda: time.time() + 3600)

    # Dependencies (task_ids that must finish first)
    dependencies:   Set[str]       = field(default_factory=set)

    # Scheduler tracking (do not set manually)
    status:         TaskStatus     = TaskStatus.PENDING
    assigned_vm_id: Optional[str]  = None
    assigned_container_id: Optional[str] = None
    start_time:     Optional[float]= None
    finish_time:    Optional[float]= None
    migrations:     int            = 0
    queue_entries:  int            = 0

    # ── Computed properties ──────────────────────────────────────────────────

    @property
    def deadline_slack(self) -> float:
        """
        Fraction of total deadline window remaining.
        
        Formula: (deadline - now) / (deadline - arrival)
        Range:   1.0 = just arrived, 0.0 = at deadline, <0 = violated
        
        WHY THIS MATTERS: This is the 's' in our urgency formula.
        A task with slack=0.1 is nearly at its deadline → high urgency.
        A task with slack=0.9 just arrived → low urgency.
        """
        total_window = self.deadline - self.arrival_time
        if total_window <= 0:
            return 0.0
        remaining = self.deadline - time.time()
        return max(0.0, remaining / total_window)

    @property
    def urgency_score(self) -> float:
        """
        U(τᵢ) = 0.6 × (1 − priority/P_max) + 0.4 × (1 − slack)
        
        Range: 0.0 (no urgency) to 1.0 (maximum urgency)
        
        Priority component: priority=1 → score=0.8, priority=5 → score=0.0
        Slack component:    slack=0.0  → score=0.4, slack=1.0  → score=0.0
        """
        P_MAX = 5
        priority_score = 1.0 - (self.priority / P_MAX)
        slack_score    = 1.0 - self.deadline_slack
        return 0.6 * priority_score + 0.4 * slack_score

    @property
    def is_ready(self) -> bool:
        """True if all dependencies are met — task can be scheduled now."""
        return len(self.dependencies) == 0

    @property
    def sla_violated(self) -> bool:
        """True if task has finished but missed its deadline."""
        if self.finish_time is None:
            return False
        return self.finish_time > self.deadline

    def exec_time_on(self, vm_mips: float) -> float:
        """
        Expected execution time of this task on a VM with given MIPS.
        
        Formula: exec_time = MI / MIPS  (in seconds, since MI = MIPS × seconds)
        Example: 10,000 MI on a 2,000 MIPS VM = 5 seconds
        """
        if vm_mips <= 0:
            return float('inf')
        return (self.mi * 1e6) / (vm_mips * 1e6)  # both in instructions/sec

    def slack_budget_on(self, vm_mips: float, current_time: float) -> float:
        """
        How much time is left AFTER executing on this VM before deadline.
        
        slack_budget = deadline - now - exec_time
        
        If slack_budget >= migration_cost → safe to migrate
        If slack_budget >= 0             → safe to queue
        If slack_budget < 0              → must assign NOW (SLA critical)
        """
        return self.deadline - current_time - self.exec_time_on(vm_mips)

    def __repr__(self):
        return (f"Task({self.task_id} | {self.mi:.0f}MI | "
                f"P{self.priority} | slack={self.deadline_slack:.2f} | "
                f"{self.status.value})")


# ─────────────────────────────────────────────────────────────────────────────
# CONTAINER MODEL (for dual-layer scheduling — Novelty #1)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Container:
    """
    Represents a container slot WITHIN a VM.
    
    WHY THIS EXISTS (Novelty #1):
      Existing papers schedule at VM level OR container level. Never both.
      PMEC-X schedules at BOTH levels simultaneously:
        - Outer layer: which VM gets this workload?
        - Inner layer: which container slot within that VM?
      
      This matters because a VM might have 8 vCPUs, and we can pack
      multiple containers into it, each running independent tasks.
      PMEC-X tracks container-level resource usage to make tighter
      consolidation decisions than VM-level schedulers can.
    
    FIELDS:
      container_id  → Unique ID
      vm_id         → Which VM this container runs on
      cpu_limit     → vCPU cores allocated to this container
      mem_limit_mb  → Memory allocated
      task_id       → Currently running task (None if idle)
    """
    container_id:   str           = field(default_factory=lambda: str(uuid.uuid4())[:8])
    vm_id:          str           = ""
    cpu_limit:      int           = 1
    mem_limit_mb:   float         = 512.0
    task_id:        Optional[str] = None
    created_at:     float         = field(default_factory=time.time)

    @property
    def is_idle(self) -> bool:
        return self.task_id is None

    def __repr__(self):
        task = self.task_id or "idle"
        return f"Container({self.container_id} on VM={self.vm_id} | task={task})"


# ─────────────────────────────────────────────────────────────────────────────
# VIRTUAL MACHINE MODEL
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VirtualMachine:
    """
    Represents a Virtual Machine in the heterogeneous cloud pool.
    
    FIELDS EXPLAINED:
      vm_id         → Unique identifier
      mips          → Processing speed (Million Instructions Per Second).
                      A 10,000 MIPS VM finishes a 10,000 MI task in 1 second.
                      Heterogeneous pool means VMs have VERY different MIPS.
      
      ram_mb        → Total RAM. Tasks need mem_mb ≤ vm.available_ram_mb.
      cpu_cores     → Total vCPUs. Container allocations consume these.
      
      cost_per_hr   → Operational cost in $/hr. Premium VMs are faster but
                      cost more. PMEC-X balances speed vs. cost.
      
      power_idle_w  → Watts consumed at idle state. The "energy-proportionality
                      gap" — even an idle server burns 60-70% of peak power.
                      This is what consolidation attacks: shut idle VMs down.
      
      power_full_w  → Watts at 100% utilization. Linear model between idle/full.
      
      inlet_temp_c  → NOVELTY #5: Inlet air temperature of the physical host
                      this VM runs on (°C). Hot-aisle servers get penalized in
                      scoring even if CPU headroom looks fine — because scheduling
                      there raises cooling cost. Range typically 18–35°C.
                      18–22°C = cool zone = good. 28–35°C = hot zone = penalized.
      
      host_id       → Physical host this VM is on. Multiple VMs share a host.
                      Used for thermal grouping and anti-affinity rules.
    
    TRACKING FIELDS (updated every epoch by simulation):
      current_load      → Fraction of capacity in use (0.0 to 1.0)
      available_ram_mb  → RAM currently free
      containers        → Container slots within this VM
      ewma_load         → EWMA-smoothed load prediction (updated by predictor)
      free_at           → Timestamp when VM will next be free
    """
    # Identity
    vm_id:          str           = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name:           str           = ""
    host_id:        str           = "host-0"

    # Capacity
    mips:           float         = 2000.0
    ram_mb:         float         = 4096.0
    cpu_cores:      int           = 4
    bandwidth_mbps: float         = 1000.0

    # Economics
    cost_per_hr:    float         = 0.10     # $/hr

    # Energy (Fan et al. 2007 linear model)
    power_idle_w:   float         = 120.0    # Watts at idle
    power_full_w:   float         = 250.0    # Watts at full load

    # NOVELTY #5 — Thermal state
    inlet_temp_c:   float         = 22.0     # °C inlet air temperature

    # Status
    status:         VMStatus      = VMStatus.IDLE

    # Runtime tracking (updated by scheduler)
    current_load:   float         = 0.0
    available_ram_mb: float       = field(init=False)
    containers:     List[Container] = field(default_factory=list)
    running_tasks:  List[str]     = field(default_factory=list)  # task_ids
    free_at:        float         = field(default_factory=time.time)

    # EWMA state (updated by predictor every epoch)
    ewma_load:      float         = 0.0      # Smoothed load estimate
    predicted_load: float         = 0.0      # Next-epoch forecast

    def __post_init__(self):
        self.available_ram_mb = self.ram_mb

    # ── Energy model ─────────────────────────────────────────────────────────

    @property
    def current_power_w(self) -> float:
        """
        P(vⱼ, uⱼ) = P_idle + (P_full - P_idle) × utilization
        
        This is the Fan et al. (2007) linear power model.
        A completely idle VM still draws power_idle_w watts.
        This idle power is what PMEC-X eliminates by shutting VMs down.
        """
        if self.status == VMStatus.SHUTDOWN:
            return 0.0
        return self.power_idle_w + (self.power_full_w - self.power_idle_w) * self.current_load

    def energy_for_task(self, task: Task) -> float:
        """
        Estimated energy (Watt-hours) to run a task on this VM.
        
        Formula: energy = power × time = current_power_w × exec_time / 3600
        """
        exec_t = task.exec_time_on(self.mips)
        return self.current_power_w * exec_t / 3600.0

    # ── Capacity checks ───────────────────────────────────────────────────────

    @property
    def predicted_headroom(self) -> float:
        """
        1 - predicted_load → how much capacity is predicted to be FREE next epoch.
        This is what PMEC-X scores against — not current load, PREDICTED load.
        That is the difference between reactive and predictive scheduling.
        """
        return max(0.0, 1.0 - self.predicted_load)

    @property
    def is_underloaded(self, threshold: float = 0.12) -> bool:
        """
        Below consolidation threshold θ_low. Candidate for shutdown.
        Threshold is dynamic — passed in by the scheduler based on time of day.
        Night: higher threshold (aggressive). Day: lower threshold (preserve capacity).
        """
        return self.current_load < threshold and self.status == VMStatus.ACTIVE

    @property
    def is_overloaded(self) -> bool:
        """Above safety threshold θ_high (default 0.85). No new tasks."""
        return self.current_load > 0.85

    def can_fit(self, task: Task) -> bool:
        """
        Check if this VM can physically accept this task.
        Checks: RAM, CPU cores, and predicted headroom.
        """
        if self.status in (VMStatus.SHUTDOWN, VMStatus.OVERLOADED):
            return False
        if task.mem_mb > self.available_ram_mb:
            return False
        if task.cpu_cores > self.cpu_cores:
            return False
        # Safety margin: 1.05 means we allow 5% over-commitment buffer
        load_demand = task.mi / (self.mips * 100)
        if self.predicted_load + load_demand > 0.85 * 1.05:
            return False
        return True

    # ── Thermal scoring ───────────────────────────────────────────────────────

    @property
    def thermal_score(self) -> float:
        """
        NOVELTY #5: Normalized thermal penalty.
        
        Range: 1.0 = cool (18°C) → good score
               0.0 = hot  (38°C) → bad score
        
        Formula: 1 - (temp - MIN_TEMP) / (MAX_TEMP - MIN_TEMP)
        Clamped to [0, 1].
        
        Physical reasoning: scheduling on a hot-aisle host increases
        cooling energy expenditure even if CPU utilization looks fine.
        """
        MIN_TEMP = 18.0   # Ideal cold-aisle temperature
        MAX_TEMP = 38.0   # Emergency threshold
        normalized = (self.inlet_temp_c - MIN_TEMP) / (MAX_TEMP - MIN_TEMP)
        return max(0.0, 1.0 - normalized)

    def __repr__(self):
        return (f"VM({self.vm_id} | {self.mips:.0f}MIPS | "
                f"load={self.current_load:.2f} | {self.status.value} | "
                f"{self.inlet_temp_c:.1f}°C)")


# ─────────────────────────────────────────────────────────────────────────────
# PHYSICAL HOST MODEL
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PhysicalHost:
    """
    Represents a physical server in the data center.
    
    WHY THIS EXISTS:
      Multiple VMs run on one physical host. The host's thermal state
      (inlet temperature) affects ALL VMs on it. PMEC-X uses host-level
      thermal data to make smarter placement decisions.
    
    FIELDS:
      host_id       → Unique identifier
      total_mips    → Aggregate MIPS of all CPUs on this server
      total_ram_mb  → Total physical RAM
      inlet_temp_c  → Air inlet temperature (°C) — the key thermal metric
      pue           → Power Usage Effectiveness (1.0 = perfect, typical 1.4–1.6)
                      PUE accounts for cooling overhead. A host with PUE=1.5
                      uses 1.5W of total facility power for every 1W of compute.
      vms           → List of VM IDs running on this host
    """
    host_id:        str           = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name:           str           = ""
    total_mips:     float         = 50000.0
    total_ram_mb:   float         = 65536.0
    total_cores:    int           = 32
    inlet_temp_c:   float         = 22.0
    pue:            float         = 1.4
    vms:            List[str]     = field(default_factory=list)

    @property
    def effective_power_factor(self) -> float:
        """
        Accounts for PUE in energy calculations.
        A VM drawing 100W on a host with PUE=1.4 actually costs 140W
        at the facility level.
        """
        return self.pue

    def __repr__(self):
        return (f"Host({self.host_id} | {len(self.vms)} VMs | "
                f"{self.inlet_temp_c:.1f}°C | PUE={self.pue})")


# ─────────────────────────────────────────────────────────────────────────────
# SCHEDULING RESULT MODEL
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SchedulingDecision:
    """
    Records what PMEC-X decided for a single task in a single epoch.
    
    WHY THIS EXISTS:
      We need an audit trail of every decision for:
        1. Generating paper result tables (energy, SLA, makespan)
        2. Feeding the Bayesian weight tuner (did this decision work out?)
        3. Debugging why a task got migrated or queued
    """
    task_id:        str
    action:         str           # "assign" | "queue" | "migrate" | "retain"
    vm_id:          Optional[str] = None
    container_id:   Optional[str] = None
    score:          float         = 0.0
    epoch:          int           = 0
    timestamp:      float         = field(default_factory=time.time)
    reason:         str           = ""

    # Factor breakdown (for debugging and paper analysis)
    score_speed:    float         = 0.0
    score_avail:    float         = 0.0
    score_cost:     float         = 0.0
    score_energy:   float         = 0.0
    score_carbon:   float         = 0.0
    score_thermal:  float         = 0.0
    score_headroom: float         = 0.0

    def __repr__(self):
        return (f"Decision({self.action.upper()} task={self.task_id} "
                f"→ vm={self.vm_id} | score={self.score:.3f})")
