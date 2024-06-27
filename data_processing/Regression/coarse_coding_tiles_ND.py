# ========= maps 
import numpy as np
class int_map():
    def __init__(self, d_min, d_max) -> None:
        self.d_min = d_min
        self.num_actions = (d_max - d_min) + 1
        #print(self.num_actions)


    def __call__(self, x) -> int:
        return x - self.d_min
    
    @property
    def n(self):
        return self.num_actions 
    
class lin_map():
    def __init__(self, d_min, d_max, num_seg) -> None:
        self.num_seg = num_seg
        half_seg = (d_max-d_min) / (num_seg ) / 2
        self.min_max_mod  = [d_min + half_seg, d_max - half_seg*(0.999)]
        self.minmax = [d_min, d_max]
        self.calc_coefficients(*self.min_max_mod, num_seg-1)

    def calc_coefficients(self,mn,mx,ns):
        a = (ns )/(mx - mn)
        b =  - a*mn
        self.a, self.b = a,b

    def __call__(self, x) -> int:
        return int(self.a*x + self.b + 0.5)
    
    @property
    def n(self):
        return self.num_seg
    
#========= main class    
class multi_tile():
    def __init__(self, num_tiles, dtypes, domain) -> None:
        self.num_tiles = num_tiles
        self.domain = domain
        # self.dims   = len(self.domain[0])
        self.dtypes  = dtypes
        self.dims   = len(domain)
        self.tiles = []
        self.cr8_maps()
        #print(self.l_maps);#print(self.ni)
        self.node_ids()
        
    def node_ids(self):
        self.ni = []
        i = 0
        for layer in self.l_maps:
            dims = []
            for dir in layer:
                dims.append(dir.n)
            a = np.arange(np.prod(dims)).reshape(dims) + i
            self.ni.append(a)
            i  = a.max() + 1
        self.max_tiling_ID = i
    
    def extended_domain(self, mn, mx, tile, offset, num_segments):
        domain_length = (mx-mn)
        dx = domain_length*offset # offset for each layer 
        Dx_left = (self.num_tiles - (tile + 1))*dx # - offset2 # tile indexing from 0
        Dx_right = tile * dx 
        segment_size = domain_length / num_segments
        extended_domain = [mn - Dx_left  , mx + Dx_right]
        additional_segments = np.round((Dx_left + Dx_right)/segment_size + 0.5).astype(int)

        return extended_domain, num_segments + additional_segments

    def cr8_maps(self):
        a = []
        for i in range(self.num_tiles):
            a.append([])
            for j in range(self.dims):
                mn,mx = self.domain[j][:2]
                if self.dtypes[j] == 'float':
                    segs  =  self.domain[j][2][i]
                    offset = self.domain[j][3]
                    (mn,mx), segs = self.extended_domain(mn,mx, i, offset, segs)
                    a[i].append(lin_map(mn, mx, segs))
                else:
                    a[i].append(int_map(mn,mx))
        self.l_maps = a
    
    def get_idx(self, pt):
        a =[]
        for i in range(self.num_tiles):
            a.append([])
            for j in range(self.dims):
                k = self.l_maps[i][j](pt[j])
                a[i].append(k)
            
        return [self.ni[i][tuple(a[i])] for i in range(self.num_tiles)]
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    tiles_num = 5;
    nt = 40
    dt  = 'float'
    dt2 = 'float'
    dom1 = [0, 2.0 * np.pi, [nt]*tiles_num, 0.1]
    dom2 = [0, 2.0 * np.pi, [nt]*tiles_num, 0.1]
    dtypes = [dt,dt2]; params = [dom1,dom2]
    mt = multi_tile(tiles_num,dtypes,params)

    def target_fn(x, y):
        return np.sin(x) + np.cos(y)

    def target_fn_noisy(x, y):
        return target_fn(x, y) + 0.1 * np.random.randn()

    perf = []
    w = np.zeros(mt.max_tiling_ID)
    # step size for SGD
    alpha = 0.25

    for i in range(20000):
        x, y = 2.0 * np.pi * np.random.rand(2)
        target = target_fn_noisy(x, y)
        active_tiles = mt.get_idx((x,y))
        pred = np.sum(w[active_tiles])
        w[active_tiles] += alpha * (target - pred)
        perf.append(np.abs(target - pred))

    plt.plot(perf)


    res = 300
    x = np.arange(0.0, 2.0 * np.pi, 2.0 * np.pi / res)
    y = np.arange(0.0, 2.0 * np.pi, 2.0 * np.pi / res)
    z = np.zeros([len(x), len(y)])
    
    for i in range(len(x)):
        for j in range(len(y)):
            active_tiles = mt.get_idx((x[i],y[j]))
            z[j,i] = np.sum(w[active_tiles])
        
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(1, 1, 1, projection='3d',computed_zorder=False)
    X, Y = np.meshgrid(x, y)
    surf = ax.plot_surface(X, Y, z, cmap=plt.get_cmap('hot'), linewidth=1.5, antialiased=False, alpha = 0.9, label='Approximation')
    Z = target_fn(X,Y)
    ax.plot_wireframe(X, Y, Z, rstride=30, cstride=30, linewidth = 0.7, label='True surface')#, zorder = 100)
    ax.view_init(elev=30., azim=45)
    plt.legend()
    plt.show()