import numpy as np,  pickle, os
from scipy.sparse import coo_matrix, triu
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from tqdm import tqdm


def gen_sparse_matrices_symm_and_non(edges, num_particles):
    # sparse matrix holds entries 1 on indicies (i,j), which shows that particles i and j interact though spectific force
    # generate such matrix from edges (connections)

    if not edges:   rows, cols = np.array([]), np.array([])
    else:           rows, cols = np.array(edges).T

    sm = coo_matrix((np.ones(len(edges)), (rows, cols)), shape=(num_particles, num_particles), dtype=int)
    # entries that are symmetric under matrix transpose dont give new information since F(i,j) = -F(j,i) and F(j,i) = - |F(i,j)|
    # 0 entries are also symmetric, but we are not interested in them, since they are not particle interaction markers. can do this:
    sm_symm         = (sm + sm.T) == 2       
    # and by same logic:
    sm_non_symm     = coo_matrix((sm - sm.T) == 1)
    # get half of entries, by isolating symmetric top triangle
    sm_symm_u       = triu(sm_symm)
    
    return sm_symm_u, sm_non_symm

def prep_matrices(edges_spring, edges_repel, num_particles):
    # generate two matrices for spring and repulsive interactions
    # m2_X contains upper symmetric matrix and non-symmetic matrix for this interaction
    m2_spring   = gen_sparse_matrices_symm_and_non(edges_spring , num_particles)
    m2_repel    = gen_sparse_matrices_symm_and_non(edges_repel  , num_particles)

    return (m2_spring, m2_repel)


def lj_repel(dist_n, dist, k_r):
    # F_r = 1/norm(d)^7 * d / norm(d) = d / norm(d)^8
    return -24 * k_r**6 * (1 / dist_n**8)[:, np.newaxis] * dist / dist_n[:, np.newaxis]

def integration(positions, velocities, forces, sm_su_spring, sm_ns_spring, sm_su_repel, sm_ns_repel, positions_t_OG, 
                k_s, k_t, k_r, i_start, i_num, dt , k_t_ramp = [], k_s_ramp = [], k_r_ramp = [], doPlot = False, ax = None, ids_set = [], plot_every = 50, path = '', case = ''):

    # simulation of particle movement under spring, columb-like and friction forces using Verlet intergration method.
    # forces are pair-wise and superposition holds. they can be anti-symmetric -> F(i,j) = -F(j,i) (for particles i and j)
    # and magnitudes |F(i,j)| = |F(j,i)| are symmetric (s), so only half of them have to be calculated and other half is obtained by inverting direction.
    # there exist cases where |F(i,j)| != |F(j,i)|, these are non-symmetric cases (ns) and have to all be calculated.
    # cases can be determined from connections matrices of size (num_particles, num_particles)
    # matrix is blank, with indicies (i,j) = 1 represent that particles i and j interact. in most cases it is a sparse matrix (few non zero entries)
    # symmetric and non-symmetric connections are stored in sparse matrices. see 'gen_sparse_matrices_symm_and_non()' for more details.
    # main forces are:
    #   1) spring force which is prop to distance between particles (i,j)
    #   2) repulsive force, which is acquired from leonard-jones potential repulsive term, which is prop to r^(-7)
    #   3) horizontal constraint force where force is parabolic for particle displacement from equilibruim point
    #   4) friction force which is proportional to velocity
    # generally i want ot constraint nodes to move only vertically, so they stay at respective vertical time slices

    # values can ramp up or down during intial stage of iteration. L = length of iterations from start; a & b are from - to values to ramp.
    if len(k_s_ramp) > 0:
        a, b, L     = k_s_ramp
        s_k_vals    = np.concatenate([np.linspace(a, b, min(L, i_num)), np.full(i_num - L, b)])  

    if len(k_t_ramp) > 0:
        a, b, L     = k_t_ramp
        t_k_vals    = np.concatenate([np.linspace(a, b, min(L, i_num)), np.full(i_num - L, b)])  

    if len(k_r_ramp) > 0:
        a, b, L     = k_r_ramp
        r_k_vals    = np.concatenate([np.linspace(a, b, min(L, i_num)), np.full(i_num - L, b)])  

    if doPlot:
        # for force arrow patches during plot
        edges_spring    =  (    list(zip(sm_su_spring.row, sm_su_spring.col))   + 
                                list(zip(sm_su_spring.col, sm_su_spring.row))   +
                                list(zip(sm_ns_spring.col, sm_ns_spring.row))   )

        edges_repel     =  (    list(zip(sm_su_repel.row, sm_su_repel.col))     + 
                                list(zip(sm_su_repel.col, sm_su_repel.row))     +
                                list(zip(sm_ns_repel.col, sm_ns_repel.row))     )

    for k in tqdm( range(i_num) ):
        i = k + i_start
        if len(k_s_ramp) > 0: k_s = s_k_vals[k]
        if len(k_t_ramp) > 0: k_t = t_k_vals[k]
        if len(k_r_ramp) > 0: k_r = r_k_vals[k]

        # UPDATE POSITIONS FROM VELOCITY AND FORCE FIELDS
        positions += (velocities*dt + 0.5 * forces * dt**2)

        # GET UPDATED FORCES FOR NEW POSITIONS, REMEMBER OLD FORCES FOR OLD POSITIONS
        forces_new              = np.zeros_like(positions) 
        
        # spring forces:
        d_spring_s  = positions[sm_su_spring.col]   -   positions[sm_su_spring.row]
        d_spring_ns = positions[sm_ns_spring.col]   -   positions[sm_ns_spring.row]
        
        # F_s = k_s * norm(d) * d_dir; d_dir = d/norm(d)
        force_attractive_s      = k_s * d_spring_s      
        force_attractive_ns     = k_s * d_spring_ns

        # drop horizontal component to make particles move mostly vertically
        force_attractive_s[ :,0]    = 0.0   
        force_attractive_ns[:,0]    = 0.0

        np.add.at(forces_new, sm_su_spring.row  , force_attractive_s    )
        np.add.at(forces_new, sm_su_spring.col  , -force_attractive_s   )
        np.add.at(forces_new, sm_ns_spring.row  , 2*force_attractive_ns )   # one sided force, double it

        # columb -like force
        d_repel_s   = positions[sm_su_repel.col ]   -   positions[sm_su_repel.row ]
        d_repel_ns  = positions[sm_ns_repel.col ]   -   positions[sm_ns_repel.row ]

        # F_r = 1/norm(d)^7 * d / norm(d) = d / norm(d)^8
        force_repulsive_s       = lj_repel(np.linalg.norm(d_repel_s     , axis = 1),    d_repel_s   , k_r)
        force_repulsive_ns      = lj_repel(np.linalg.norm(d_repel_ns    , axis = 1),    d_repel_ns  , k_r)

        np.add.at(forces_new, sm_su_repel.row   , force_repulsive_s     )
        np.add.at(forces_new, sm_su_repel.col   , -force_repulsive_s    )
        np.add.at(forces_new, sm_ns_repel.row   , 2*force_repulsive_ns  )

        # time constraints -> parabolic returning force
        drs = positions[:,0] - positions_t_OG

        # F_t = norm(d)^2 * d / norm(d) = norm(d) * d, its only 1D here
        forces_x = -1 * k_t * np.abs(drs) * drs
        forces_new[:,0] += forces_x

        # UPDATE VELOCTIES BASED ON MEAN OLD-NEW FORCE FIELDS AND TAKE INTO ACCOUNT FRICTION
        velocities += (0.5 * (forces + forces_new) * dt - 0.1* velocities)

        # END CYCLE 
        forces = forces_new

        if doPlot and i % plot_every == 0 :
            ax.clear()  # Clear the previous data from the figure
            draw_plot(ax, positions, ids_set, edges_spring, edges_repel, x_min, x_max, y_min, y_max, i)
            fig.savefig(os.path.join(path, f'{case}_{i}.png'))

    return positions 


#fig, ax = plt.subplots()

def draw_plot(ax, positions, pos_set, edges_spring, edges_repel, x_min, x_max, y_min, y_max, i):
    for IDs in edges_spring:
        arrow = FancyArrowPatch(positions[IDs[0]], positions[IDs[1]], color='red', arrowstyle='->', mutation_scale=15, linewidth=2)
        ax.add_patch(arrow)

    # Plot arrows for edges_repel
    for IDs in edges_repel:
        arrow = FancyArrowPatch(positions[IDs[0]], positions[IDs[1]], color='blue', arrowstyle='->', mutation_scale=15, linewidth=1)
        ax.add_patch(arrow)

    ax.scatter(positions[:, 0], positions[:, 1])  # Assuming a scatter plot for 2D positions
    if len(pos_set) > 0:
        ax.scatter(positions[pos_set, 0], positions[pos_set, 1], marker = 'x', color = 'green')
    ax.set_title(f'Iteration {i}')
    ax.set_xlim((x_min,x_max))
    ax.set_ylim((y_min,y_max))
    ax.grid(True)


if 1 == -1:
    import time

    with open('modules/graphs_node_position_spring_lj_mode_iterator.pickle', 'rb') as handle:
        [stray_nodes, seg_nodes, positions, positions_t_OG, conns_spring, conn_repel] = pickle.load(handle)
    save_path = 'particle_movement'

    fig, ax = plt.subplots()

    nodes_all = stray_nodes + seg_nodes

    node_enum = {n:i for i,n in enumerate(nodes_all)} # node -> order in nodes_all

    ids_set     = range(len(stray_nodes),len(stray_nodes) + len(seg_nodes))

    velocities      = np.zeros_like(positions, float)   # ordered same as nodes_all
    forces          = np.zeros_like(positions, float)
    

    edges_spring    = [(node_enum[a],node_enum[b]) for a,b in conns_spring] # connections between nodes ->
    edges_repel     = [(node_enum[a],node_enum[b]) for a,b in conn_repel]   # -> connections of indicies of nodes_all

    num_particles   = len(positions)
    (m_spr, m_rep) = prep_matrices(edges_spring, edges_repel, num_particles)

    dt = 0.01
    #x_min,x_max = min(positions[:,0]) - 1, max(positions[:,0]) + 1
    x_min,x_max = 369,386
    y_min,y_max = min(positions[:,1]) - 1, max(positions[:,1]) + 1

    start_time  = time.time()

    start       = 0
    num         = 2000 #(positions,velocities,forces, sm_su_spring, sm_ns_spring, sm_su_repel, sm_ns_repel, positions_t_OG, k_s, k_t, k_r, i_start, i_num, k_t_ramp = [], k_s_ramp = [], k_r_ramp = [])
    positions   =   integration(positions, velocities, forces, *m_spr, *m_rep, positions_t_OG,                                                 
                            k_s = 2, k_t = 50, k_r = 0.1, i_start = start, i_num = num, dt = dt,
                            k_t_ramp = [], k_s_ramp = [], k_r_ramp = [], doPlot = False, ax = ax, ids_set = ids_set, path = save_path)

    draw_plot(ax, positions, ids_set, edges_spring, edges_repel, x_min, x_max, y_min, y_max, -1)
    plt.show()

    start       += num    # start of new counter
    num         = 2000

    k_t_ramp    = [50,200,int(3/4*num) ]

    positions   =   integration(positions, velocities, forces, *m_spr, *m_rep, positions_t_OG,                                                 
                                k_s = 2, k_t = 50, k_r = 0.1, i_start = start, i_num = num, dt = dt,
                                k_t_ramp = k_t_ramp, k_s_ramp = [], k_r_ramp = [], doPlot = False, ax = ax, ids_set = ids_set, path = save_path)

    #draw_plot(ax, positions, ids_set, x_min, x_max, y_min, y_max, -1)
    #plt.show()
    start       += num    # start of new counter
    num         = 2000

    k_t_ramp    = [200, 300, int(3/4*num)  ]
    k_r_ramp    = [0.1, 0.2, int(num/2)    ]

    positions   =   integration(positions, velocities, forces, *m_spr, *m_rep, positions_t_OG, 
                                k_s = 1, k_t = 10, k_r = 0.2, i_start = start, i_num = num, dt = dt,                                                
                                k_t_ramp = k_t_ramp, k_s_ramp = [], k_r_ramp = k_r_ramp , doPlot = False, ax = ax, ids_set = ids_set, path = save_path)
                                                

    end_time = time.time()

    elapsed_time = end_time - start_time

    print(f"\nTime elapsed: {elapsed_time:.6f} seconds")