import os

thickness = 5.0
for idx in range(94)[1:]:
    with open('make_water_shell.pml', 'w') as the_file:
        the_file.write('load MBP_1gp2_%s.pdb\n'%idx)
        the_file.write('select w,resn HOH\n')
        the_file.write('alter (w),chain="B"\n')
        the_file.write('select (p),resn POT\n')
        the_file.write('alter (p),chain="C"\n')

        # the_file.write('select (c),resn CLA\n')
        # the_file.write('alter (c),chain="D"\n')

        the_file.write('select chain A\n')
        the_file.write('select s1, chain B within %.1f of chain A\n'%thickness)
        the_file.write('selec s12, (sele) or (s1)\n')
        the_file.write('save MBP_1gp2_water_shell_%gA_%s.pdb, s12\n'%(int(thickness), idx) )

    os.system("pymol -c make_water_shell.pml")