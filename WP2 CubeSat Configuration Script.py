import h5py
import numpy as np
import re

# Constants
k_B = 1.380649e-23 
m_molecular = 100.0 * 1.66054e-27  
# File paths
hdf5_file = r"C:\Users\sarah\Documents\MEng Project Systema\Models\Cube Box Outgas_Cube Box outgas\Cube Box outgas.md.h5"
normals_file = r"C:\Users\sarah\Documents\MEng Project Systema\Models\Cube Box_Cube Box\Cube Box.nod.nwk"

# Target node range
TARGET_NODE_START = 100001
TARGET_NODE_END = 100040

print("="*70)
print("CUBE BOX OUTGASSING THRUST CALCULATION")
print(f"Using nodes {TARGET_NODE_START}-{TARGET_NODE_END}")
print("="*70)

print("\nStep 1: Reading surface normals from .nod.nwk file...")
all_normals = []

with open(normals_file, 'r') as nf:
    for line in nf:
        line_stripped = line.strip()
        
        if not line_stripped or line_stripped.startswith('#') or line_stripped.startswith('$'):
            continue
        
        if line_stripped.startswith('D'):
            node_match = re.search(r'D\s+(\d+)', line)
            if node_match:
                node_id = int(node_match.group(1))
                
                nx_match = re.search(r'NX\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', line)
                ny_match = re.search(r'NY\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', line)
                nz_match = re.search(r'NZ\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', line)
                
                if nx_match and ny_match and nz_match:
                    nx = float(nx_match.group(1))
                    ny = float(ny_match.group(1))
                    nz = float(nz_match.group(1))
                    all_normals.append([node_id, nx, ny, nz])

all_normals = sorted(all_normals, key=lambda x: x[0])
print(f"  Total normals loaded: {len(all_normals)}")
print(f"  Node ID range: {all_normals[0][0]} to {all_normals[-1][0]}")

target_indices = []
target_normals = []
for idx, normal in enumerate(all_normals):
    node_id = normal[0]
    if TARGET_NODE_START <= node_id <= TARGET_NODE_END:
        target_indices.append(idx)
        target_normals.append(normal)

print(f"  Found {len(target_indices)} nodes in range {TARGET_NODE_START}-{TARGET_NODE_END}")
if len(target_indices) > 0:
    print(f"  Indices in file: {target_indices[0]} to {target_indices[-1]}")
    print(f"  Node IDs: {target_normals[0][0]} to {target_normals[-1][0]}")
else:
    print("  ERROR: No nodes found in target range!")
    exit(1)

normals_array = np.array([n[1:] for n in target_normals]) 
node_ids = np.array([n[0] for n in target_normals])  

print("\nStep 2: Reading HDF5 data for target nodes...")
with h5py.File(hdf5_file, 'r') as f:
    emittable_mass_all = f['/Results/Outgassing/Emittable mass'][target_indices, :, :]  
    emittable_mass_density_all = f['/Results/Outgassing/Emittable mass density'][target_indices, :, :]  
    area_all = f['/Results/Properties/Area'][target_indices]
    temperature_all = f['/Results/Outgassing/Temperature'][target_indices, :]
    
    print(f"  Emittable mass shape: {emittable_mass_all.shape}")
    print(f"  Emittable mass density shape: {emittable_mass_density_all.shape}")
    print(f"  Temperature shape: {temperature_all.shape}")
    print(f"  Area shape: {area_all.shape}")

# Extract data
emittable_mass = np.squeeze(emittable_mass_all) 
emittable_mass_density = np.squeeze(emittable_mass_density_all)  
area = area_all  
temperature = temperature_all  

n_nodes = len(target_indices)
n_timesteps = emittable_mass.shape[1] if len(emittable_mass.shape) > 1 else 1
print(f"  Number of nodes: {n_nodes}")
print(f"  Number of timesteps: {n_timesteps}")

print("\nStep 3: Calculating thrust parameters (using final timestep)...")

final_timestep_idx = -1

m_final = emittable_mass[:, final_timestep_idx] if len(emittable_mass.shape) > 1 else emittable_mass

rho_final = emittable_mass_density[:, final_timestep_idx] if len(emittable_mass_density.shape) > 1 else emittable_mass_density

T_final = temperature[:, final_timestep_idx] if len(temperature.shape) > 1 else temperature


dt = 1.0  
if n_timesteps > 1:
    m_current = emittable_mass[:, -2]  
    m_next = emittable_mass[:, -1] 
    delta_m = m_current - m_next  
    

    delta_m = np.zeros(n_nodes)

m_dot = delta_m / dt  # [kg/s]


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
print("SAVING TO CSV")
print(f"{'='*70}")

output_file = r"C:\Users\sarah\Documents\MEng Project Systema\Models\Cube Box Outgas_Cube Box outgas\cube_box_thrust_results.csv"

with open(output_file, 'w') as f:
    f.write("CUBE BOX OUTGASSING THRUST CALCULATION RESULTS\n")
    f.write(f"Node Range:,{TARGET_NODE_START}-{TARGET_NODE_END}\n")
    f.write(f"Number of Nodes:,{n_nodes}\n")
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