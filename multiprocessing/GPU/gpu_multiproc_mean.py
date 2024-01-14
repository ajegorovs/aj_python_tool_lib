import  time, datetime, torch, numpy as np
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from gpu_torch_cuda_functions import (batch_axis0_mean)
"""
example on how to process batches of images on cuda compatible GPUs
general consideration is that theres more RAM than VRAM, so data is held on CPU.
PREP:
    here we generate a large array of random floats (N_images, H, W) either:
        1) on CPU using np.random.randn()
        2) on GPU slice by slice using torch.randn(..., device = 'cuda') and fill arrray on CPU
this approach is tested due to limited VRAM. Still GPU aproach is faster
EXEC:
    array is fed onto GPU using DataLoader in batches.
    from each batch axis = 0 mean is calcualted
    total array mean is reconstructed by weighted average. weights by slice length.
"""

def gen_slices(data_length, slice_width):
    # input:    data_length, slice_width = 551, 100
    # output:   [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500), (500, 551)] 
    return [(i,min(i+slice_width, data_length)) for i in range(0, data_length, slice_width)]

def track_time(reset = False):
    # tracks time between track_time() executions
    if reset:
        track_time.last_time = time.time()
        return '(Initializing time counter)'
    else:
        current_time = time.time()
        time_passed = current_time - track_time.last_time
        track_time.last_time = current_time
        return f'({time_passed:.2f} s)'
    
def print2(inp):
    print(f'{datetime.datetime.now().strftime("%H-%M-%S")} {inp}')

class dataset_create(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample

print2(f'generating data {track_time(True)}')
ref_type = torch.float16                                             # only from size considerations
data_size, img_dims = 1200, (800,1200)
dims                = (data_size, *img_dims)
slice_size          = 200
slices              = gen_slices(data_size, slice_size)

# test CPU/GPU random float generation
mode = 'gpu'

if mode == 'cpu_np':                                                # full arr on CPU
    arr = np.random.randn(*dims).astype(np.float16)                 # 
elif mode in 'gpu':                                                 # slices on GPU -> CPU
    #device = 'cpu'                                                 # switch to do torch calc on CPU
    arr = np.zeros(dims, np.float16)

    for i, (fr, to) in enumerate(slices):                                   
        temp = torch.randn((to - fr, *img_dims), dtype=ref_type, device=device)
        arr[fr:to, ...] = temp.to('cpu').numpy()                             #GPU (1.32 s) CPU (223.06 s)
else:                                                                        # full on GPU -> CPU
    arr = torch.randn(dims, dtype=ref_type, device=device).to('cpu').numpy() # (0.82 s)

# ===== MEASURE CPU MEAN =====
print2(f'generating data... done {track_time()}')
mean_cpu = np.mean(arr, axis= 0)                                             # mean CPU (2.61 s)  
print2(f'calcualte mean CPU... done {track_time()}')

# ===== MEASURE GPU MEAN =====
t0_GPU          = time.time()
data            = torch.from_numpy(arr)                                      # wrapper, not a copy
dataset_here    = dataset_create(data)
dataloader      = DataLoader(dataset_here, batch_size=150, shuffle=False)

mean_gpu        = batch_axis0_mean(dataloader, img_dims, ref_type = ref_type, device= device)
mean_gpu        = mean_gpu.squeeze(0).to('cpu').numpy()

print2(f'calculate mean GPU... done ({time.time() - t0_GPU:.2f} s)')         # mean GPU (1.08 s)  

print2(f'Is mean_cpu == mean_gpu? :{np.allclose(mean_cpu, mean_gpu, rtol=1e-03, atol=1e-03)}')
# reduced tolerances, maybe fluctuations accumulate