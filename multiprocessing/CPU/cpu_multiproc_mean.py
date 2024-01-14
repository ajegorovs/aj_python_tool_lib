from multiprocessing import shared_memory, Manager, Pool, Event
import os, time, datetime,  numpy as np
from threading import Thread

"""
It is a general pipeline for ansync (non-blocking) processing using multiple worker processes (kernels)
kernels are intialized using dummy function, they persist within 'with Pool() as pool:' context manager.
so you may assign them different tasks. Initialization takes time, and state is tracked by an event 'kernels_ready'.
We can do other work meanwhile, and we passively check status of kernels on separate thread.
when it comes using kernels, they all must be started and all execution is blocked until.
When kernels are ready we assign pool tasks using result_async = starmap_async(...). Again while kernels are running 
we are free to do other tasks but once we need data from result_async, we have to do result_async.wait()
to make sure all tasks are finished.
note: code wont crash if there is an error on kernel, you must check async_result.success() or if not, retrieve an error.
"""

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
    

def gen_slices(data_length, slice_width):
    # input:    data_length, slice_width = 551, 100
    # output:   [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500), (500, 551)] 
    return [(i,min(i+slice_width, data_length)) for i in range(0, data_length, slice_width)]


def check_ready(event):
    # timer which checks how long did it take to launch kernels.
    # runs concurrently on main thread.
    t = 0.0
    while not event.is_set():
        time.sleep(0.1)
        t += 0.1
    else:
        print2(f'Kernels are ready ({t:0.1f} s)')
    return

def check_ready_blocking(object, time_init = time.time(), wait_msg = '', done_msg = ''):
    # once you stuck waiting for kernels, can generate updating info line with time passed
    t = time.time() - time_init # time offset if counting began earlier than f-n launch

    if str(type(object)) == "<class 'multiprocessing.synchronize.Event'>" :
        check_func = lambda x: x.is_set()
    else:
        check_func = lambda x: x.ready()

    while not check_func(object):
        print2(f'{wait_msg} ({t:0.1f} s)', end='\r', flush=True)
        time.sleep(0.1)
        t += 0.1
    print('\n')
    print2(f'{done_msg} ({t:0.1f} s)', flush=True)
    track_time()

    return

def print2(inp, end='\n', flush=False):
    pid = os.getpid()
    if __name__ == "__main__":
        pid_str = f'PRNT:{pid:>{5}}'
    else:
        pid_str = f'WRKR:{pid:>{5}}'
    
    return print(f'({datetime.datetime.now().strftime("%H-%M-%S")})[{pid_str}] {inp}', end = end, flush = flush)  


def intialize_workers(worker_ready_event, arr_shape, arr_type):
    global first_iter, mean_arr_shape, mean_arr_type
    first_iter      = True          # define globals
    mean_arr_shape  = arr_shape        
    mean_arr_type   = arr_type
    worker_ready_event.set()        # signal that worker has done init.
    track_time(True)                # initialize time counter for each worker

def mean_slice(from_to, queue, report = False):
    global first_iter

    buf_mean    = shared_memory.SharedMemory(name='buf_mean')
    mean_np     = np.ndarray(mean_arr_shape, dtype = mean_arr_type, buffer=buf_mean.buf)
    if report:
        if not first_iter:  print2(f'time between iterations: {track_time()}')
        else:               first_iter = False

    s_from, s_to    = from_to
    res             = np.mean(mean_np[s_from: s_to,...], axis = 0)
    queue.put((s_to - s_from , res))
    if report: print2(f'{from_to} iteration time: {track_time()}')
    return 


def mean_slice_finish(num_elems, img_dims, queue):
    weights = np.zeros(num_elems)
    buffer  = np.zeros((num_elems, *img_dims))
    #print2(f'reserved mem {track_time()}')
    i = 0
    while not queue.empty():
        weight, image = queue.get()
        weights[i] = weight
        buffer[i,:,:] = image
        #print2(f'retrieved queue {i} w: {weight} {track_time()}')
        i += 1
    #print2(f'w: {weights}')
    return np.average(buffer, axis=0, weights=weights)

if __name__ == "__main__":

    manager         = Manager()
    result_queue    = manager.Queue()
    print2(f'generating data {track_time(True)}')
    data_size, img_dims = 1200, (800,1200)
    dt = np.float16
    #dataArchive     = np.random.randn(data_size, *img_dims).astype(dt)
    dataArchive     = np.random.randint(0, 256, size=(data_size, *img_dims), dtype=np.uint8).astype(dt)

    print2(f'generating data... done {track_time()}')
    np.mean(dataArchive, axis = 0)
    print2(f'mean calc... done {track_time()}')
    del dataArchive
    
    print2(f'generating data {track_time()}')
    # define a buffer big enough to fit all data 
    dims            = (data_size, *img_dims)
    size_in_bytes   = int(np.prod(dims)) * np.dtype(dt).itemsize #2304000000 
    shared_data     = shared_memory.SharedMemory(create=True, size=size_in_bytes, name="buf_mean")

    # fill buffer with random integers. not generating it fully because buffer + array might not fit into RAM. otherwise:
    #np_buff[...] = np.random.rand(*np_buff.shape)
    #np.copyto(np.ndarray(dataArchive.shape, dtype=dataArchive.dtype, buffer=shared_data.buf), dataArchive)

    np_buff         = np.ndarray(dims, dtype=dt, buffer=shared_data.buf)
    
    slice_size      = 150
    slices          = gen_slices(data_size, slice_size)

    for fr,to in slices:
        np_buff[fr:to,...] = np.random.randint(0, 256, size=(to - fr, *img_dims), dtype=np.uint8).astype(dt)

    print2(f'generating data... done {track_time()}')
    t0_CPU = time.time()
    mean_cpu = np.mean(np_buff, axis = 0)
    t_CPU = time.time() - t0_CPU
    print2(f'mean calc... done {track_time()}')

    print2(f'spawning workers... {track_time()}')
    kernels_ready = Event()
    report_kernels_ready = 0
    if report_kernels_ready:
        check_ready_thread = Thread(target=check_ready, args=(kernels_ready,))

    num_processors  = 4
    slice_size      = 150
    slices          = gen_slices(data_size, slice_size)
    with Pool(processes = num_processors, initializer = intialize_workers, initargs = (kernels_ready, dims, dt)) as pool:
        # processes/kernels have to be launched, it takes time. you can do other tasks meanwhile on this processor.
        # concurrent check_ready_thread will track how long it takes to launch kernels and output a message when ready.
        if report_kernels_ready: check_ready_thread.start()
        # can simulate work by sleeping while kernes are loading
        t0 = time.time()
        slp = 1
        print2(f'doing unrelated work for {slp} s...') 
        time.sleep(slp)
        # if work is finished but kernels are not loaded block all execution until they are ready. with print timer.
        if not kernels_ready.is_set():  #event_or_queue 0 or 1
            check_ready_blocking(kernels_ready, t0, 'Waiting until kernels are ready...', 'Kernels are ready!')

        t0_CPU_MP = time.time()
        track_time()
        async_result = pool.starmap_async(mean_slice, ((s, result_queue) for s in slices))
        #async_result.wait()                 # wait until all workers are done. can do other stuff before.
        if not async_result.ready(): 
            check_ready_blocking(async_result, t0, wait_msg='Waiting results ...', done_msg= 'Work completed! ')
        if not async_result.successful():
            print2(f'shit failed: {async_result.get()}')
        
        
        print2(f'end first parallel {track_time()}')
        
        mean_gpu = mean_slice_finish(len(slices), img_dims, result_queue)
        print2(f'end mean calc... {track_time()}')
        print(f'CPU_MP time =({time.time() - t0_CPU_MP:.2f} s)')

        print2(f'Is mean_cpu == mean_gpu? :{np.allclose(mean_cpu, mean_gpu, rtol=1e-03, atol=1e-03)}')
        a = 1
        # async_result = pool.starmap_async(mean_slice, [(s, dataArchive, result_queue) for s in slices])
        # async_result.wait()  

        # mean_slice_finish(len(slices), img_dims, result_queue)

        # print(f'{timeHMS()}: {os.getpid()} end second parallel {track_time()}')

        # async_result = pool.starmap_async(foo, [(s, dataArchive, result_queue) for s in slices])
        # async_result.wait()  # This line might be causing the issue, try commenting it out

        # # Process the results from the second calculation
        # mean_slice_finish(len(slices), img_dims, result_queue)

        # print(f'{timeHMS()}: {os.getpid()} end third parallel {track_time()}')
    a = 1
    shared_data.close()
    shared_data.unlink()
