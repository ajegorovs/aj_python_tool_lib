import numpy as np
class tiles():
    def __init__(self, tiles_x, tiles_y, layers, offset, max_entries_side = 1000) -> None:
        self.Nx     = np.array([tiles_x]*layers) if type(tiles_x) == int else np.array(tiles_x)
        self.Ny     = np.array([tiles_y]*layers) if type(tiles_y) == int else np.array(tiles_y)
        self.Nxy    = np.vstack((self.Nx,self.Ny)).T
        self.Nt     = layers
        self.D      = (np.array(offset)*max_entries_side).astype(int)
        self.Nn     = max_entries_side
        self.Dtot   = self.D*(layers-1)
        self.M      = None
        
    def fill_layer(self,nxy):
        # create local tile indexing
        nxy = np.array(nxy) 
        bp = np.arange(np.prod(nxy)).reshape(nxy) 
        # split segments regions into almost equal parts
        Nxs = [0] + [len(a) for a in np.array_split(range(self.Nn), nxy[0])]
        Nys = [0] + [len(a) for a in np.array_split(range(self.Nn), nxy[1])]
        Nxs,Nys = np.cumsum(Nxs),np.cumsum(Nys)    # start + right borders if i-th sector

        M = np.zeros(shape=(self.Nn,self.Nn), dtype = int)
        for (x,y), ID in np.ndenumerate(bp):
            M[Nxs[x]:Nxs[x+1], Nys[y]:Nys[y+1]] = ID
        return M
    
    def tiles(self):
        M = np.zeros((self.Nt, *([self.Nn]*2 +self.Dtot)), dtype = int)
        dx,dy,first_ID = *self.D,0
        for i in range(self.Nt):
            sl_x,sl_y       = slice(i*dx,i*dx+self.Nn), slice(i*dy,i*dy+self.Nn)
            M[i,sl_x,sl_y]  = self.fill_layer(self.Nxy[i]) + first_ID
            first_ID        += np.prod(self.Nxy[i])    # ID offset from local to global
        sl_x = slice((self.Nt -1 )*self.D[0],-(self.Nt - 1)*self.D[0])
        sl_y = slice((self.Nt -1 )*self.D[1],-(self.Nt - 1)*self.D[1])
        self.M = M[:,sl_x,sl_y]
        return self.M
    
    def embedding(self, dom_x, dom_y):
        """ Create mapping from domain to index and embedding matrix"""
        DOMAIN_X_MIN,DOMAIN_X_MAX = dom_x
        DOMAIN_Y_MIN,DOMAIN_Y_MAX = dom_y
        M = self.tiles()
        Mxmax, Mymax = np.array(M[0].shape) - 1
        k0 = (Mxmax - 0)/(DOMAIN_X_MAX - DOMAIN_X_MIN)
        k1 = (Mymax - 0)/(DOMAIN_Y_MAX - DOMAIN_Y_MIN)
        m0 = -k0*DOMAIN_X_MIN
        m1 = -k1*DOMAIN_Y_MIN
        self.kx,self.mx = k0,m0
        self.ky,self.my = k1,m1

        M_embed = np.zeros(shape=(*M[0].shape, M.max() + 1), dtype=int)
        for (x, y), _ in np.ndenumerate(M[0]):
            where_ones = M[:,x,y]           # slice across tiles dim
            M_embed[x,y][where_ones] = 1    # one-hot encode

        self.M_embed = M_embed

    def get_embed(self, x, y, oneHot = True):
        """retrieve one-hot encoded embedding from embedding matrix"""
        id_x, id_y = int(self.kx*x + self.mx),int(self.ky*y + self.my)
        enc = self.M_embed[id_x,id_y]
        if oneHot:
            return enc
        else:
            return np.argwhere(enc != 0).flatten()
    
    def plot(self):
        import matplotlib.pyplot as plt
        from scipy import ndimage
        fig,ax = plt.subplots(1,self.Nt, figsize = (5*self.Nt,5), facecolor = 'black')
        M = self.M
        for a,m in zip(ax, M):
            m[np.bitwise_or(ndimage.sobel(m, 0) > 0, ndimage.sobel(m, 1) > 0)] *= 0
            a.matshow(m)
            a.set_xticks([])
            a.set_yticks([])

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    num_layers = 3
    offsets_xy = (0.15,0.1)
    N_segments_x = [3,6,3]
    N_segments_y = [6,2,8]
    t = tiles(N_segments_x,N_segments_y,num_layers, offsets_xy,max_entries_side = 400)
    
        
    dom_x = (0.0, 2.0 * np.pi)
    dom_y = (0.0, 2.0 * np.pi)
    t.embedding(dom_x,dom_y)
    t.plot()
    print(t.get_embed(0,0, oneHot=False))
    plt.show()
