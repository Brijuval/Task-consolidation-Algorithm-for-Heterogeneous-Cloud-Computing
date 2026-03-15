package org.cloudbus.cloudsim.examples;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.LinkedList;
import java.util.List;

import org.cloudbus.cloudsim.*;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.provisioners.BwProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.PeProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.RamProvisionerSimple;

public class EATCSimulation {

    private static List<Cloudlet> cloudletList;
    private static List<Vm> vmList;

    public static void main(String[] args) {
        Log.printLine("Starting EATC Simulation...");

        try {
            // 1. Initialize CloudSim
            int num_user = 1;
            Calendar calendar = Calendar.getInstance();
            boolean trace_flag = false;
            CloudSim.init(num_user, calendar, trace_flag);

            // 2. Create the Heterogeneous Datacenter
            Datacenter datacenter0 = createDatacenter("Datacenter_0");

            // 3. Create our Custom EATC Broker
            EATCBroker broker = new EATCBroker("EATC_Broker");
            int brokerId = broker.getId();

            // 4. Create Heterogeneous VMs
            vmList = new ArrayList<Vm>();
            // Type 1: High Performance VM
            for (int i = 0; i < 5; i++) {
                // 1000 MIPS, 1 CPU, 2GB RAM
                vmList.add(new Vm(i, brokerId, 1000, 1, 2048, 10000, 10000, "Xen", new CloudletSchedulerSpaceShared()));
            }
            // Type 2: Low Power VM
            for (int i = 5; i < 10; i++) {
                // 500 MIPS, 1 CPU, 1GB RAM
                vmList.add(new Vm(i, brokerId, 500, 1, 1024, 10000, 10000, "Xen", new CloudletSchedulerSpaceShared()));
            }
            broker.submitVmList(vmList);

            // 5. Create Heterogeneous Cloudlets (Tasks)
            cloudletList = new ArrayList<Cloudlet>();
            for (int i = 0; i < 100; i++) {
                // Randomize Length to simulate heterogeneity (1000 to 10,000 MI)
                long length = 1000 + (long)(Math.random() * 9000); 
                long fileSize = 300;
                long outputSize = 300;
                int pesNumber = 1;
                UtilizationModel utilizationModel = new UtilizationModelFull();

                Cloudlet cloudlet = new Cloudlet(i, length, pesNumber, fileSize, outputSize, utilizationModel, utilizationModel, utilizationModel);
                cloudlet.setUserId(brokerId);
                cloudletList.add(cloudlet);
            }
            broker.submitCloudletList(cloudletList);

            // 6. Start Simulation
            CloudSim.startSimulation();

            // 7. Get Results
            List<Cloudlet> newList = broker.getCloudletReceivedList();
            CloudSim.stopSimulation();

            // 8. Print Output & Calculate Energy
            printCloudletList(newList);
            calculateEnergy(newList);

            Log.printLine("EATC Simulation Finished!");

        } catch (Exception e) {
            e.printStackTrace();
            Log.printLine("The simulation has been terminated due to an unexpected error");
        }
    }

    private static Datacenter createDatacenter(String name) {
        List<Host> hostList = new ArrayList<Host>();

        // Create Heterogeneous Hosts
        // Machine Type 1: High End (3000 MIPS)
        List<Pe> peList1 = new ArrayList<Pe>();
        peList1.add(new Pe(0, new PeProvisionerSimple(3000)));
        peList1.add(new Pe(1, new PeProvisionerSimple(3000))); // Dual Core
        
        for (int i = 0; i < 5; i++) {
            hostList.add(new Host(
                    i,
                    new RamProvisionerSimple(16384),
                    new BwProvisionerSimple(100000),
                    1000000,
                    peList1,
                    new VmSchedulerTimeShared(peList1)
            ));
        }

        // Machine Type 2: Efficiency (1500 MIPS)
        List<Pe> peList2 = new ArrayList<Pe>();
        peList2.add(new Pe(0, new PeProvisionerSimple(1500))); // Single Core
        
        for (int i = 5; i < 10; i++) {
            hostList.add(new Host(
                    i,
                    new RamProvisionerSimple(8192),
                    new BwProvisionerSimple(100000),
                    1000000,
                    peList2,
                    new VmSchedulerTimeShared(peList2)
            ));
        }

        String arch = "x86";
        String os = "Linux";
        String vmm = "Xen";
        double time_zone = 10.0;
        double cost = 3.0;
        double costPerMem = 0.05;
        double costPerStorage = 0.1;
        double costPerBw = 0.1;
        LinkedList<Storage> storageList = new LinkedList<Storage>();

        DatacenterCharacteristics characteristics = new DatacenterCharacteristics(
                arch, os, vmm, hostList, time_zone, cost, costPerMem, costPerStorage, costPerBw);

        try {
            return new Datacenter(name, characteristics, new VmAllocationPolicySimple(hostList), storageList, 0);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private static void printCloudletList(List<Cloudlet> list) {
        int size = list.size();
        Cloudlet cloudlet;

        String indent = "    ";
        Log.printLine();
        Log.printLine("========== OUTPUT ==========");
        Log.printLine("Cloudlet ID" + indent + "STATUS" + indent + "Data center ID" + indent + "VM ID" + indent + "Time" + indent + "Start Time" + indent + "Finish Time");

        DecimalFormat dft = new DecimalFormat("###.##");
        for (int i = 0; i < size; i++) {
            cloudlet = list.get(i);
            Log.print(indent + cloudlet.getCloudletId() + indent + indent);

            if (cloudlet.getCloudletStatus() == Cloudlet.SUCCESS) {
                Log.print("SUCCESS");
                Log.printLine(indent + indent + cloudlet.getResourceId() + indent + indent + indent + cloudlet.getVmId() +
                        indent + indent + dft.format(cloudlet.getActualCPUTime()) + indent + indent + dft.format(cloudlet.getExecStartTime()) +
                        indent + indent + dft.format(cloudlet.getFinishTime()));
            }
        }
    }

    // --- ENERGY CALCULATION (From your Methodology) ---
    private static void calculateEnergy(List<Cloudlet> list) {
        double totalEnergy = 0;
        double maxPower = 250; // Watts (Standard Server Peak)
        double idlePower = 0.6 * maxPower; // 60% Idle power
        
        // Energy = Power * Time
        // For EATC, we sum the energy used during the active time of every task
        for (Cloudlet c : list) {
            double executionTime = c.getActualCPUTime();
            // E = (P_busy - P_idle) * utilization + P_idle
            // Since task is running, utilization is 1.0 (100% for that core)
            double energyForTask = maxPower * executionTime; 
            totalEnergy += energyForTask;
        }
        
        Log.printLine("\n===========================================");
        Log.printLine("TOTAL ENERGY CONSUMED: " + new DecimalFormat("#.##").format(totalEnergy / 1000) + " kWh");
        Log.printLine("===========================================");
    }
}