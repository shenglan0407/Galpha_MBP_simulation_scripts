# remove water molecules
import mdtraj as md

for ii in range(17)[1:]:
    traj=md.load_pdb('MBP_1gfi_wlig_MD/1MBP_1gfi_%d.pdb'%ii)
    atoms_idx_to_keep=[a.index for a in traj.topology.atoms if not (str(a.residue.name) \
        in ['POT','HOH','CLA'])]
    new_traj=traj.atom_slice(atoms_idx_to_keep)
    new_traj.save('MBP_1gfi_wlig_MD/MBP_1gfi_nowater_%d.pdb'%ii)