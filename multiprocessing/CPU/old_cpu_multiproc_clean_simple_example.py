from multiprocessing import Manager, Pool, Event, shared_memory
import os, time, datetime, numpy as np

def track_time(reset = False):
    if reset:
        track_time.last_time = time.time()
        return '(Initializing time counter)'
    else:
        current_time = time.time()
        time_passed = current_time - track_time.last_time
        track_time.last_time = current_time
        return f'({time_passed:.2f} s)'
    
def print2(inp):
    t = datetime.datetime.now().strftime("%H-%M-%S")
    if __name__ == "__main__":  pid = f'PRNT:{os.getpid():>{5}}'
    else:                       pid = f'WRKR:{os.getpid():>{5}}'
    return print(f'{t}:[{pid}] {inp}')  

def gen_slices(data_length, slice_width):
    # input:    data_length, slice_width = 551, 100
    # output:   [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500), (500, 551)] 
    return [(i,min(i+slice_width, data_length)) for i in range(0, data_length, slice_width)]

def intialize_workers(worker_ready_event, arr_shape, arr_type):
    global first_iter, dataArchive, shp, tp
    first_iter = True
    shp = arr_shape; tp = arr_type
    existing_shm = shared_memory.SharedMemory(name='dataArchive')
    dataArchive = np.ndarray(arr_shape, dtype=arr_type, buffer=existing_shm.buf)
    #print2(dataArchive[0,:2,:2])
    worker_ready_event.set()
    track_time(True)

def mean_slice(from_to, queue):
    global first_iter, shp, tp#, dataArchive
    
    existing_shm = shared_memory.SharedMemory(name='dataArchive')
    dataArchive = np.ndarray(shp, dtype=tp, buffer=existing_shm.buf)
    #print2(f'\n{dataArchive[0,:2,:2]}')

    if not first_iter:  print2(f'time between iterations: {track_time()}')
    else:               first_iter = False

    s_from, s_to = from_to
    result = np.mean(dataArchive[s_from: s_to,...], axis = 0)
    queue.put((s_to - s_from , result))
    print2(f'{from_to} iteration time: {track_time()}')
    return 

if __name__ == "__main__":

    manager         = Manager()
    result_queue    = manager.Queue()
    
    data_size, img_dims = 1200, (800, 1200)
    dt = np.float16
    print2(f'generating data {track_time(True)}')
    
    if 1 == -1: # single core test for ref. 
        dataArchive     = np.random.randint(0, 256, size=(data_size, *img_dims), dtype=np.uint8).astype(dt)
        print2(f'generating data... done {track_time()}')
        np.mean(dataArchive, axis = 0)
        print2(f'mean calc... done {track_time()}')
        del dataArchive
    
    slice_size      = 150
    slices          = gen_slices(data_size, slice_size)
    dims            = (data_size, *img_dims)
    size_in_bytes   = int(np.prod(dims)) * np.dtype(dt).itemsize  
    shared_data     = shared_memory.SharedMemory(create=True, size=size_in_bytes, name="dataArchive")
    np_buff         = np.ndarray(dims, dtype=dt, buffer=shared_data.buf)
    for i in range(data_size):
        np_buff[i,...] = np.random.randint(0, 256, size=img_dims, dtype=np.uint8).astype(dt)

    #print2(f'\n{np_buff[0,:2,:2]}')
    #np_buff[...] = np.random.rand(*np_buff.shape)
    #np.copyto(np.ndarray(dataArchive.shape, dtype=dataArchive.dtype, buffer=shared_data.buf), dataArchive)
    print2(f'generating data... done {track_time()}')

    if 1 == -1:
        np.mean(np_buff, axis = 0)
        print2(f'mean calc... done {track_time()}')

    print2(f'spawning workers... {track_time()}')
    kernels_ready = Event()
    num_processors  = 4
    with Pool(processes = num_processors, initializer = intialize_workers, initargs = (kernels_ready, dims, dt)) as pool:
        while not kernels_ready.is_set():
            time.sleep(0.1)
        else: print2(f'kernels are ready {track_time()}')
        t0_CPU_MP = time.time()
        async_result = pool.starmap_async(mean_slice, ((s, result_queue) for s in slices))
        async_result.wait() 
        if not async_result.successful():
            print(async_result.get())

        # (calc weighted average here) ~insta
        
        print(f'CPU_MP time = {time.time() - t0_CPU_MP:.2f} s')

        # ~other tasks for the pool
    shared_data.close()
    shared_data.unlink()
