Simulate particle interaction to augment networkx node position spring model. Added repulsive forces.
You supply particle interaction pairs. it can be symmetric i->j ~ j->i or non-symmetric.
Since for symm case forces are same by magnitude, its enough to calc i->j and then invert it for j->i.
Its done by forming a particle interaction matrix, in which entry i,j marks interaction between particle i and j.
By transposing and stuff you can extract symmetric and non-symm entries.