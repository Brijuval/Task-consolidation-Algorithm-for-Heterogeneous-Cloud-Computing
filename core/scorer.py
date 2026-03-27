"""
PMEC-X: Predictive Multifactor Energy Consolidation
=====================================================
scorer.py — 7-Factor Normalized VM Scoring Engine

WHAT THIS FILE DOES:
  For every (task, VM) pair, computes a single score S ∈ [0, 1].
  The VM with the highest score gets the task.

THE 7 FACTORS:
  f1: Speed      — How fast will this VM run the task?
  f2: Headroom   — How much predicted capacity does the VM have?
  f3: Cost       — How cheap is this VM?
  f4: Energy     — How energy-efficient is this VM?
  f5: Carbon     — How green is the electricity powering this VM? (NOVELTY #3)
  f6: Thermal    — How cool is the server this VM is on?         (NOVELTY #5)
  f7: Urgency    — How urgent is this task? (blended in per task)

FORMULA:
  S(τᵢ, vⱼ) = Σₖ wₖ × fₖ(τᵢ, vⱼ)   where Σwₖ = 1

WHY NORMALIZATION MATTERS:
  All factors are on different scales:
    - MIPS: 500 to 10,000
    - Cost: $0.05 to $2.00/hr
    - Watts: 80 to 500W
    - Temperature: 18°C to 38°C
  
  Without normalization, MIPS would dominate just because its numbers are big.
  We normalize each factor to [0, 1] before weighting, so every factor
  gets a fair contribution proportional to its weight.

WEIGHT VECTOR W:
  Default equal weights (1/7 each) — meaning all factors are equally
  important. The Adaptive Weight Tuner (Day 2) will learn better weights
  from actual scheduling outcomes.
  
  Operators can also manually tune:
    - Energy-focused: w_energy=0.4, w_carbon=0.3, rest share 0.3
    - SLA-focused:    w_headroom=0.4, w_speed=0.3, rest share 0.3
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from core.models import Task, VirtualMachine, SchedulingDecision


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHT VECTOR
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WeightVector:
    """
    The 7 weights that control PMEC-X's priorities.
    
    Must sum to 1.0. The Adaptive Weight Tuner updates these every epoch
    based on observed scheduling outcomes (energy saved, SLA compliance).
    
    INTERPRETATION:
      w_speed=0.3 means "30% of the score comes from how fast the VM is"
    """
    w_speed:    float = 1/7     # Factor 1: VM processing speed
    w_headroom: float = 1/7     # Factor 2: Predicted available capacity
    w_cost:     float = 1/7     # Factor 3: Operational cost efficiency
    w_energy:   float = 1/7     # Factor 4: Energy efficiency
    w_carbon:   float = 1/7     # Factor 5: Carbon intensity (novelty)
    w_thermal:  float = 1/7     # Factor 6: Thermal state (novelty)
    w_urgency:  float = 1/7     # Factor 7: Task urgency blend

    def normalize(self):
        """Ensure weights sum to 1.0 after any manual adjustment."""
        total = (self.w_speed + self.w_headroom + self.w_cost +
                 self.w_energy + self.w_carbon + self.w_thermal + self.w_urgency)
        if total > 0:
            self.w_speed    /= total
            self.w_headroom /= total
            self.w_cost     /= total
            self.w_energy   /= total
            self.w_carbon   /= total
            self.w_thermal  /= total
            self.w_urgency  /= total

    def as_list(self) -> List[float]:
        return [self.w_speed, self.w_headroom, self.w_cost,
                self.w_energy, self.w_carbon, self.w_thermal, self.w_urgency]

    def __repr__(self):
        return (f"W(speed={self.w_speed:.3f} head={self.w_headroom:.3f} "
                f"cost={self.w_cost:.3f} energy={self.w_energy:.3f} "
                f"carbon={self.w_carbon:.3f} thermal={self.w_thermal:.3f} "
                f"urgency={self.w_urgency:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZATION CONTEXT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NormalizationContext:
    """
    Pool-wide maximums needed to normalize each factor to [0, 1].
    
    WHY A SEPARATE OBJECT:
      Normalization depends on the WHOLE VM pool (e.g., "this VM costs
      $0.10/hr, which is good if the max is $2.00/hr, but bad if the
      max is $0.12/hr"). We compute pool-wide stats once per epoch,
      not per (task, VM) pair. This keeps scoring O(n·m) total.
    
    UPDATED EVERY EPOCH by the scorer before any task is assigned.
    """
    max_mips:       float = 1.0
    max_cost:       float = 1.0
    max_power_full: float = 1.0
    max_inlet_temp: float = 38.0
    min_inlet_temp: float = 18.0
    max_carbon:     float = 1.0     # gCO₂/kWh (normalized 0-1)

    @classmethod
    def from_vm_pool(cls, vms: List[VirtualMachine]) -> 'NormalizationContext':
        """
        Compute normalization context from the current VM pool.
        Call this once per epoch before scoring any tasks.
        """
        if not vms:
            return cls()

        active_vms = [v for v in vms if v.status.value != 'shutdown']
        if not active_vms:
            active_vms = vms

        return cls(
            max_mips        = max(v.mips for v in active_vms),
            max_cost        = max(v.cost_per_hr for v in active_vms),
            max_power_full  = max(v.power_full_w for v in active_vms),
            max_inlet_temp  = max(v.inlet_temp_c for v in active_vms),
            min_inlet_temp  = min(v.inlet_temp_c for v in active_vms),
            max_carbon      = 1.0,  # carbon scores already normalized in module
        )


# ─────────────────────────────────────────────────────────────────────────────
# SCORING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class PMECXScorer:
    """
    The heart of PMEC-X — computes S(τᵢ, vⱼ) for every (task, VM) pair.
    
    USAGE:
        scorer = PMECXScorer(weights=WeightVector())
        context = NormalizationContext.from_vm_pool(vm_pool)
        score, breakdown = scorer.score(task, vm, context, carbon_score=0.7)
    
    The scorer is stateless between calls — all state lives in WeightVector
    and NormalizationContext. This makes it easy to swap weights for the
    Bayesian tuner and for baseline comparison experiments.
    """

    def __init__(self, weights: Optional[WeightVector] = None):
        self.weights = weights or WeightVector()
        self.scoring_history: List[SchedulingDecision] = []

    # ── Individual factor computations ────────────────────────────────────────

    def f1_speed(self, task: Task, vm: VirtualMachine, ctx: NormalizationContext) -> float:
        """
        Factor 1: Speed
        
        Formula: MIPS_j / max_k(MIPS_k)
        
        A faster VM gets a higher score. But speed alone is not enough —
        a fast VM that's already at 90% load is worse than a slower VM
        with plenty of headroom. That's why we have Factor 2.
        
        Range: 0.0 (slowest VM in pool) to 1.0 (fastest VM in pool)
        """
        if ctx.max_mips <= 0:
            return 0.0
        return min(1.0, vm.mips / ctx.max_mips)

    def f2_headroom(self, task: Task, vm: VirtualMachine, ctx: NormalizationContext) -> float:
        """
        Factor 2: Predicted headroom
        
        Formula: 1 - predicted_load
        
        This uses the EWMA+CUSUM prediction, NOT the current load.
        Assigning against predicted load prevents over-commitment:
        if a VM is at 40% now but EWMA predicts 70% next epoch,
        we don't assign a heavy task to it.
        
        Range: 0.0 (fully loaded) to 1.0 (completely free)
        """
        return max(0.0, vm.predicted_headroom)

    def f3_cost(self, task: Task, vm: VirtualMachine, ctx: NormalizationContext) -> float:
        """
        Factor 3: Cost efficiency
        
        Formula: 1 - (cost_j / max_k(cost_k))
        
        Inverted so that LOWER cost → HIGHER score.
        A $0.05/hr VM gets score=0.97 if max is $2.00/hr.
        A $2.00/hr VM gets score=0.0.
        
        Range: 0.0 (most expensive) to 1.0 (cheapest)
        """
        if ctx.max_cost <= 0:
            return 1.0
        return max(0.0, 1.0 - (vm.cost_per_hr / ctx.max_cost))

    def f4_energy(self, task: Task, vm: VirtualMachine, ctx: NormalizationContext) -> float:
        """
        Factor 4: Energy efficiency
        
        Formula: 1 - (power_full_j / max_k(power_full_k))
        
        Inverted so that LOWER power draw → HIGHER score.
        A 150W VM gets score=0.7 if max is 500W.
        A 500W VM gets score=0.0.
        
        Note: We use power_full_w (peak power) as the proxy for energy
        efficiency of the hardware design. A VM that draws 150W at full
        load is intrinsically more efficient than one that draws 500W.
        
        Range: 0.0 (highest power) to 1.0 (lowest power)
        """
        if ctx.max_power_full <= 0:
            return 1.0
        return max(0.0, 1.0 - (vm.power_full_w / ctx.max_power_full))

    def f5_carbon(self, task: Task, vm: VirtualMachine,
                  ctx: NormalizationContext, carbon_score: float = 0.5) -> float:
        """
        Factor 5: Carbon intensity (NOVELTY #3)
        
        Formula: 1 - carbon_intensity_normalized
        
        carbon_score is provided by the Carbon Module (carbon.py).
        It represents how green the electricity grid is RIGHT NOW
        at the location of this VM's physical host.
        
        A VM running on solar/wind-heavy grid → carbon_score=0.1 → f5=0.9
        A VM running on coal-heavy grid       → carbon_score=0.9 → f5=0.1
        
        This is what no existing scheduler does: we actually shift load
        toward the greenest VMs based on live grid data.
        
        Range: 0.0 (dirtiest grid) to 1.0 (cleanest grid)
        """
        return max(0.0, 1.0 - carbon_score)

    def f6_thermal(self, task: Task, vm: VirtualMachine, ctx: NormalizationContext) -> float:
        """
        Factor 6: Thermal state (NOVELTY #5)
        
        Uses vm.thermal_score which is already computed in the VM model:
        Formula: 1 - (inlet_temp - MIN_TEMP) / (MAX_TEMP - MIN_TEMP)
        
        A VM on a cool server (20°C) → thermal_score=0.9 → good
        A VM on a hot server  (34°C) → thermal_score=0.2 → penalized
        
        Physical reasoning: scheduling tasks on hot servers increases
        cooling energy expenditure even if compute headroom looks fine.
        
        Range: 0.0 (hottest servers) to 1.0 (coolest servers)
        """
        return vm.thermal_score

    def f7_urgency(self, task: Task, vm: VirtualMachine, ctx: NormalizationContext) -> float:
        """
        Factor 7: Task urgency blend
        
        Formula: U(τᵢ) = 0.6×(1 - priority/P_max) + 0.4×(1 - slack)
        
        Note: Urgency is a TASK property, not a VM property.
        It's the same value for all VMs when scoring task τᵢ.
        Including it in the scoring function means urgency influences
        the composite score AND the task ordering.
        
        This is the right place to include it: an urgent task should
        not just be scheduled first — it should also be scheduled on
        the best VM available, not just the next free one.
        
        Range: 0.0 (low urgency) to 1.0 (maximum urgency)
        """
        return task.urgency_score

    # ── Main scoring function ─────────────────────────────────────────────────

    def score(
        self,
        task:         Task,
        vm:           VirtualMachine,
        ctx:          NormalizationContext,
        carbon_score: float = 0.5
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute S(τᵢ, vⱼ) = Σₖ wₖ × fₖ
        
        Returns:
            (composite_score, factor_breakdown)
            
            composite_score: float ∈ [0, 1]
            factor_breakdown: dict of individual factor scores for debugging
        
        FEASIBILITY GUARD:
            Returns (-1, {}) if the VM cannot physically fit the task.
            Infeasible VMs are excluded before the argmax selection.
        """
        # Hard feasibility check 1 — if VM can't fit task, score = -1
        if not vm.can_fit(task):
            return -1.0, {}

        # Hard feasibility check 2 — deadline feasibility
        # If exec_time > remaining_slack, this VM physically cannot
        # finish the task before its deadline. Never assign it.
        import time as _time
        remaining_slack = task.deadline - task.arrival_time
        exec_time_on_vm = task.exec_time_on(vm.mips)
        if exec_time_on_vm > remaining_slack * 0.95:
            return -1.0, {}

        w = self.weights

        # Compute each factor
        factors = {
            'speed':    self.f1_speed(task, vm, ctx),
            'headroom': self.f2_headroom(task, vm, ctx),
            'cost':     self.f3_cost(task, vm, ctx),
            'energy':   self.f4_energy(task, vm, ctx),
            'carbon':   self.f5_carbon(task, vm, ctx, carbon_score),
            'thermal':  self.f6_thermal(task, vm, ctx),
            'urgency':  self.f7_urgency(task, vm, ctx),
        }

        # Weighted sum
        composite = (
            w.w_speed    * factors['speed']    +
            w.w_headroom * factors['headroom'] +
            w.w_cost     * factors['cost']     +
            w.w_energy   * factors['energy']   +
            w.w_carbon   * factors['carbon']   +
            w.w_thermal  * factors['thermal']  +
            w.w_urgency  * factors['urgency']
        )

        return composite, factors

    def best_vm(
        self,
        task:         Task,
        vms:          List[VirtualMachine],
        ctx:          NormalizationContext,
        carbon_scores: Dict[str, float] = None,
        exclude_vms:   List[str] = None
    ) -> Tuple[Optional[VirtualMachine], float, Dict]:
        """
        Find the best VM for a task using argmax scoring.
        
        Args:
            task:          The task to place
            vms:           Pool of candidate VMs
            ctx:           Normalization context (computed once per epoch)
            carbon_scores: Dict of vm_id → carbon score from Carbon Module
            exclude_vms:   VM IDs to skip (e.g. currently migrating)
        
        Returns:
            (best_vm, best_score, factor_breakdown)
            Returns (None, -1, {}) if no VM can fit the task.
        
        TIME COMPLEXITY: O(m) where m = number of VMs
        """
        carbon_scores = carbon_scores or {}
        exclude_vms   = set(exclude_vms or [])

        best_vm_obj   = None
        best_score    = -1.0
        best_factors  = {}

        for vm in vms:
            if vm.vm_id in exclude_vms:
                continue

            carbon = carbon_scores.get(vm.vm_id, 0.5)
            score, factors = self.score(task, vm, ctx, carbon)

            if score > best_score:
                best_score   = score
                best_vm_obj  = vm
                best_factors = factors

        return best_vm_obj, best_score, best_factors

    def rank_vms(
        self,
        task:          Task,
        vms:           List[VirtualMachine],
        ctx:           NormalizationContext,
        carbon_scores: Dict[str, float] = None
    ) -> List[Tuple[VirtualMachine, float, Dict]]:
        """
        Return all VMs ranked by score for a task.
        Useful for migration: pick the #2 VM as migration destination.
        """
        carbon_scores = carbon_scores or {}
        scored = []
        for vm in vms:
            carbon = carbon_scores.get(vm.vm_id, 0.5)
            score, factors = self.score(task, vm, ctx, carbon)
            if score >= 0:
                scored.append((vm, score, factors))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
