"""
PMEC-X: Run Simulation — Fixed Version
"""
import copy, csv, os, time
from simulation.workload import WorkloadGenerator, build_vm_pool
from simulation.environment import (PMECXScheduler, RoundRobinScheduler,
                                     MinMinScheduler, FCFSScheduler)
from core.scorer import WeightVector

def run_all(n_epochs: int=720, verbose: bool=True):
    print("\n" + "="*60)
    print("   PMEC-X: Predictive Multifactor Energy Consolidation")
    print("="*60)
    print(f"\nGenerating {n_epochs}-epoch workload (24-hour simulation)...")

    gen      = WorkloadGenerator(seed=42)
    schedule = gen.generate_full_day()[:n_epochs]
    total    = sum(len(t) for _,_,t in schedule)
    print(f"Total tasks: {total} | Epochs: {n_epochs} | "
          f"Simulated time: {n_epochs*2/60:.1f} hours\n")

    base_pool = build_vm_pool(seed=42)

    schedulers = [
        PMECXScheduler(copy.deepcopy(base_pool), WeightVector()),
        RoundRobinScheduler(copy.deepcopy(base_pool)),
        MinMinScheduler(copy.deepcopy(base_pool)),
        FCFSScheduler(copy.deepcopy(base_pool)),
    ]

    all_results = []
    for s in schedulers:
        name = s.results.scheduler_name
        print(f"Running: {name}")
        print("-"*45)
        t0 = time.time()
        r  = s.run_full_simulation(schedule, verbose=verbose)
        print(f"  Done in {time.time()-t0:.1f}s\n")
        all_results.append(r)

    # ── Results table ──────────────────────────────────────────────────
    print("="*80)
    print("  RESULTS — 24-HOUR SIMULATION")
    print("="*80)
    print(f"  {'Scheduler':<14} {'Tasks':>6} {'Completed':>10} "
          f"{'Violated':>9} {'Viol%':>7} {'Energy(kWh)':>12} "
          f"{'AvgVMs':>7} {'Shutdowns':>10}")
    print("  " + "-"*76)

    for r in all_results:
        marker = " <<" if r.scheduler_name == "PMEC-X" else ""
        print(f"  {r.scheduler_name:<14} {r.total_tasks:>6} "
              f"{r.completed_tasks:>10} {r.violated_tasks:>9} "
              f"{r.sla_violation_rate:>6.1f}% "
              f"{r.total_energy_kwh:>12.2f} "
              f"{r.avg_active_vms:>7.1f} "
              f"{r.total_shutdowns:>10}{marker}")
    print("="*80)

    pmecx     = all_results[0]
    baselines = all_results[1:]
    best_b    = min(baselines, key=lambda r: r.total_energy_kwh)

    e_save  = (1 - pmecx.total_energy_kwh / best_b.total_energy_kwh) * 100
    sla_imp = best_b.sla_violation_rate - pmecx.sla_violation_rate
    vm_red  = best_b.avg_active_vms - pmecx.avg_active_vms

    print(f"\n  PMEC-X vs best baseline ({best_b.scheduler_name}):")
    print(f"    Energy saved:        {e_save:+.1f}%")
    print(f"    SLA improvement:     {sla_imp:+.1f} percentage points")
    print(f"    Avg VMs reduced:     {vm_red:+.1f} VMs")
    print(f"    VM shutdowns:        {pmecx.total_shutdowns}")
    print(f"    Migrations:          {pmecx.total_migrations}")
    print(f"    CUSUM regime changes:{pmecx.regime_changes}")

    os.makedirs("results", exist_ok=True)
    with open("results/summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scheduler","total_tasks","completed","violated",
                    "sla_violation_pct","energy_kwh","carbon_g",
                    "avg_active_vms","shutdowns","migrations","regime_changes"])
        for r in all_results:
            w.writerow([r.scheduler_name, r.total_tasks, r.completed_tasks,
                        r.violated_tasks, f"{r.sla_violation_rate:.2f}",
                        f"{r.total_energy_kwh:.3f}", f"{r.total_carbon_g:.1f}",
                        f"{r.avg_active_vms:.2f}", r.total_shutdowns,
                        r.total_migrations, r.regime_changes])

    print(f"\n  Saved: results/summary.csv")
    print("  Ready for dashboard (Day 3) and paper (Day 5)\n")
    return all_results

if __name__ == "__main__":
    run_all(n_epochs=720, verbose=True)
