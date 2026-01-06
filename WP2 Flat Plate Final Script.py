import h5py
import numpy as np
import re

k_B = 1.380649e-23  # Boltzmann constant [J/K]
m_molecular = 100.0 * 1.66054e-27  
hdf5_file = r"C:\Users\sarah\Documents\MEng Project Systema\Models\flat plate outgas_Flat plate outgas\Flat plate outgas.md.h5"
normals_file = r"C:\Users\sarah\Documents\MEng Project Systema\Models\flat plate thermica_Flat plate Thermica\flat plate thermica.nod.nwk"

print("="*70)
print("OUTGASSING THRUST CALCULATION")
print("="*70)

with h5py.File(hdf5_file, 'r') as f:
    print("Top-level groups:")
    for key in f.keys():
        print(f"  /{key}")
    
    if '/Model/' in f:
        print("\nContents of /Model/:")
        for key in f['/Model/'].keys():
            dataset = f['/Model/'][key]
            print(f"  {key}: shape = {dataset.shape if hasattr(dataset, 'shape') else 'N/A'}")

with h5py.File(hdf5_file, 'r') as f:
    emittable_mass_all = f['/Results/Outgassing/Emittable mass'][:]  
    emittable_mass_density_all = f['/Results/Outgassing/Emittable mass density'][:]  
    area_all = f['/Results/Properties/Area'][:]
    temperature_all = f['/Results/Outgassing/Temperature'][:]
    
    print(f"  Emittable mass shape: {emittable_mass_all.shape}")
    print(f"  Emittable mass density shape: {emittable_mass_density_all.shape}")
    print(f"  Temperature shape: {temperature_all.shape}")
    print(f"  Area shape: {area_all.shape}")

n_nodes = 40
start_idx = 3  # Skip first 3 nodes

# Extract data for nodes 4-43 (indices 3-42)
emittable_mass = np.squeeze(emittable_mass_all[start_idx:start_idx+n_nodes, :, :])
emittable_mass_density = np.squeeze(emittable_mass_density_all[start_idx:start_idx+n_nodes, :, :])  
area = area_all[start_idx:start_idx+n_nodes]  
temperature = temperature_all[start_idx:start_idx+n_nodes, :]  

n_timesteps = emittable_mass.shape[1]

print(f"  Using nodes {start_idx+1} to {start_idx+n_nodes} (40 nodes total)")

normals = []
target_nodes = range(100001, 100041)

with open(normals_file, 'r') as nf:
    for line in nf:
        line_stripped = line.strip()
        
        if not line_stripped or line_stripped.startswith('#') or line_stripped.startswith('$'):
            continue
        
        if line_stripped.startswith('D'):
            # Extract node ID
            node_match = re.search(r'D\s+(\d+)', line)
            if node_match:
                node_id = int(node_match.group(1))
                
                if node_id in target_nodes:
                    # Extract NX, NY, NZ
                    nx_match = re.search(r'NX\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', line)
                    ny_match = re.search(r'NY\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', line)
                    nz_match = re.search(r'NZ\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', line)
                    
                    if nx_match and ny_match and nz_match:
                        nx = float(nx_match.group(1))
                        ny = float(ny_match.group(1))
                        nz = float(nz_match.group(1))
                        normals.append([node_id, nx, ny, nz])

normals = sorted(normals, key=lambda x: x[0])
normals_array = np.array([n[1:] for n in normals])  


if len(normals_array) != n_nodes:
    n_nodes = min(len(normals_array), n_nodes)
    emittable_mass = emittable_mass[:n_nodes, :]
    emittable_mass_density = emittable_mass_density[:n_nodes, :]
    area = area[:n_nodes]
    temperature = temperature[:n_nodes, :]
    normals_array = normals_array[:n_nodes, :]


final_timestep_idx = -1
m_final = emittable_mass[:, final_timestep_idx]
rho_final = emittable_mass_density[:, final_timestep_idx]
T_final = temperature[:, final_timestep_idx]
dt = 1.0  
m_current = emittable_mass[:, -2]
m_next = emittable_mass[:, -1]  
delta_m = m_current - m_next  

m_dot = delta_m / dt  

v_exhaust = np.sqrt(8 * k_B * T_final / (np.pi * m_molecular))  # [m/s]

F_element = m_dot * v_exhaust  # [N]

F_net = np.zeros(3)
for i in range(n_nodes):
    F_net += F_element[i] * normals_array[i, :]

F_magnitude = np.sqrt(np.sum(F_net**2))
F_direction = F_net / (F_magnitude + 1e-30)

print(f"\n{'='*70}")
print("RESULTS SUMMARY")
print(f"{'='*70}")
print(f"Net Thrust Vector: [{F_net[0]:.6e}, {F_net[1]:.6e}, {F_net[2]:.6e}] N")
print(f"Thrust Magnitude: {F_magnitude:.6e} N")
print(f"Thrust Direction: [{F_direction[0]:.6f}, {F_direction[1]:.6f}, {F_direction[2]:.6f}]")
print(f"Total Mass Flow Rate: {np.sum(m_dot):.6e} kg/s")
print(f"Mean Exhaust Velocity: {np.mean(v_exhaust):.2f} m/s")

print(f"\n{'='*70}")
print(f"{'='*70}")

output_file = r"C:\Users\sarah\Documents\MEng Project Systema\Models\flat plate outgas_Flat plate outgas\outgassing_thrust_results.csv"

node_ids = np.arange(100001, 100001 + n_nodes)

with open(output_file, 'w') as f:
    f.write("OUTGASSING THRUST CALCULATION RESULTS\n")
    f.write(f"Net Thrust Magnitude [N]:,{F_magnitude:.6e}\n")
    f.write(f"Net Thrust Direction X:,{F_direction[0]:.6f}\n")
    f.write(f"Net Thrust Direction Y:,{F_direction[1]:.6f}\n")
    f.write(f"Net Thrust Direction Z:,{F_direction[2]:.6f}\n")
    f.write(f"Net Thrust Vector X [N]:,{F_net[0]:.6e}\n")
    f.write(f"Net Thrust Vector Y [N]:,{F_net[1]:.6e}\n")
    f.write(f"Net Thrust Vector Z [N]:,{F_net[2]:.6e}\n")
    f.write(f"Total Mass Flow Rate [kg/s]:,{np.sum(m_dot):.6e}\n")
    f.write(f"Mean Exhaust Velocity [m/s]:,{np.mean(v_exhaust):.2f}\n")
    f.write(f"Timestep dt [s]:,{dt}\n")
    f.write(f"Molecular Mass [g/mol]:,100.0\n")
    f.write(f"Molecular Mass [kg]:,{m_molecular:.6e}\n")
    f.write("\n")
    
    f.write("Node_ID,")
    f.write("Area[m2],")
    f.write("Emittable_Mass[kg],")
    f.write("Emittable_Mass_Density[kg/m2],")
    f.write("Temperature[K],")
    f.write("Mass_Flow_Rate[kg/s],")
    f.write("Exhaust_Velocity[m/s],")
    f.write("Thrust[N],")
    f.write("Normal_X,")
    f.write("Normal_Y,")
    f.write("Normal_Z\n")
    
    # Per-node data
    for i in range(n_nodes):
        f.write(f"{node_ids[i]},")
        f.write(f"{area[i]:.6e},")
        f.write(f"{m_final[i]:.6e},")
        f.write(f"{rho_final[i]:.6e},")
        f.write(f"{T_final[i]:.2f},")
        f.write(f"{m_dot[i]:.6e},")
        f.write(f"{v_exhaust[i]:.6e},")
        f.write(f"{F_element[i]:.6e},")
        f.write(f"{normals_array[i,0]:.6f},")
        f.write(f"{normals_array[i,1]:.6f},")
        f.write(f"{normals_array[i,2]:.6f}\n")

print(f"Results saved to: {output_file}")
print("\n" + "="*70)
print("CALCULATION COMPLETE")
print("="*70)