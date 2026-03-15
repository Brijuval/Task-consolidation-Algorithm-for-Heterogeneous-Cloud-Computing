package org.cloudbus.cloudsim.examples;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Set;

import org.cloudbus.cloudsim.Cloudlet;
import org.cloudbus.cloudsim.DatacenterBroker;
import org.cloudbus.cloudsim.Log;
import org.cloudbus.cloudsim.Vm;
import org.cloudbus.cloudsim.lists.VmList;

/**
 * EATCBroker: Energy Aware Task Consolidation Broker.
 * * FIX: This version overrides bindCloudletsToVms() properly 
 * and removes the manual sendNow() call causing the error.
 */
public class EATCBroker extends DatacenterBroker {

    public EATCBroker(String name) throws Exception {
        super(name);
    }

    /**
     * This method is called automatically by CloudSim when it is time
     * to decide which Task goes to which VM.
     */
    protected void bindCloudletsToVms() {
        Log.printLine("EATC Algorithm: Optimizing Task Allocation...");

        List<Cloudlet> cloudletList = getCloudletList();
        List<Vm> vmList = getVmList();

        // Safety Check: If no VMs, we can't schedule
        if (vmList.isEmpty()) {
            Log.printLine("EATC Error: No VMs created. Cannot schedule tasks.");
            return;
        }

        // --- STEP 1: Setup Min-Heap & ETC Matrix ---
        // We use a PriorityQueue to simulate the Min-Heap logic from your paper
        PriorityQueue<MappingPair> minHeap = new PriorityQueue<>();

        // Generate ETC (Expected Time to Compute) for every Task-VM pair
        for (Cloudlet c : cloudletList) {
            for (Vm v : vmList) {
                // ETC = Task Length / VM Speed (MIPS)
                double estimatedTime = c.getCloudletLength() / v.getMips();
                
                // Add to Heap
                minHeap.add(new MappingPair(c, v, estimatedTime));
            }
        }

        // --- STEP 2: Allocation Strategy ---
        Set<Integer> assignedCloudletIds = new HashSet<>();
        int tasksAssigned = 0;

        // Process the Heap
        while (!minHeap.isEmpty()) {
            // Extract the BEST option (Lowest Time) -> O(1)
            MappingPair bestPair = minHeap.poll();

            // Check if this cloudlet is already assigned
            if (!assignedCloudletIds.contains(bestPair.cloudlet.getCloudletId())) {
                
                // --- THE CRITICAL FIX ---
                // We use the parent method bindCloudletToVm. 
                // This updates the internal mapping. We DO NOT send data manually.
                bindCloudletToVm(bestPair.cloudlet.getCloudletId(), bestPair.vm.getId());
                
                // Mark as assigned
                assignedCloudletIds.add(bestPair.cloudlet.getCloudletId());
                tasksAssigned++;
            }
            
            // Optimization: If all tasks are assigned, stop the loop early
            if (tasksAssigned == cloudletList.size()) {
                break;
            }
        }
        
        Log.printLine("EATC Algorithm: Successfully bound " + tasksAssigned + " Cloudlets to VMs.");
        
        // Note: The parent DatacenterBroker class will automatically 
        // submit these cloudlets to the Datacenter after this method finishes.
    }

    // --- Helper Class for the Heap ---
    // Stores: (Task, VM, Time) and sorts by Time
    class MappingPair implements Comparable<MappingPair> {
        Cloudlet cloudlet;
        Vm vm;
        double time;

        public MappingPair(Cloudlet c, Vm v, double t) {
            this.cloudlet = c;
            this.vm = v;
            this.time = t;
        }

        @Override
        public int compareTo(MappingPair other) {
            // Min-Heap Logic: Smaller time comes first
            return Double.compare(this.time, other.time);
        }
    }
}