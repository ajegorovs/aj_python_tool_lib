import torch, cv2

def morph_erode_dilate(im_tensor, kernel, mode):
    """
    mode = 0 = erosion; mode = 1 = dilation; im_tensor is 1s and 0s
    convolve image (stack) with a kernel. 
    dilate  = keep partial/full overlap between image and kernel.   means masked sum > 0
    erosion = keep only full overlap.                               means masked sum  = kernel sum.
    erode: subtract full overlap area from result, pixels with full overlap will be 0. rest will go below 0.
    erode:add 1 to bring full overlap pixels to value of 1. partial overlap will be below 1 and will be clamped to 0.
    dilate: just clamp
    """
    padding = (kernel.shape[-1]//2,)*2
    torch_result0   = torch.nn.functional.conv2d(im_tensor, kernel, padding = padding, groups = 1)
    if mode == 0:
        full_area = torch.sum(kernel)
        torch_result0.add_(-full_area + 1)
    return torch_result0.clamp_(0, 1)


def kernel_circular(width, dtype, normalize = False, device = 'cuda'):
    """
    import circle kernel from openCV, transform it to (1,1,width,width) shape for convolution
    for mean operation should be normalized, for moprhology not.
    """
    ker = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(width,width))).unsqueeze_(0).unsqueeze_(0)
    ker = ker.to(dtype = dtype, device = device)
    if not normalize:
        return ker
    else:
        return ker/torch.sum(ker)
    
# kernel_size = 5
# kernel_mean = kernel_circular(kernel_size, normalize = True, dtype = ref_type, device = device)

def torch_blur(image, kernel_size, kernel_mean, mode = 0, device = 'cuda'):
    """
    applies blur on image using given kernel. kernel should be normalized
    """
    if mode == 0:
        blur_conv = torch.nn.Conv2d(1, 1, kernel_size = kernel_size, bias = False, padding = 'same', padding_mode ='reflect').to(device)
        blur_conv.weight = torch.nn.Parameter(kernel_mean) 
        return blur_conv(image).detach()
    else:
        padding = (kernel_size//2,)*2
        blurred = torch.nn.functional.conv2d(image, kernel_mean, padding = padding, groups = 1)
        return blurred

def batch_axis0_mean(dataloader, shape_img, ref_type, device='cuda'):
    """
    array is split into batches using suppiled data loader. because full array might not fit into VRAM
    total array mean is reconstructed from means of batches using batch lengths as weights.
    all intermediate batch means are kept.
    NOTE: everything is reshaped into (X,1,H,W) shape for torch to work.
    """
    batch_means     = torch.zeros((len(dataloader), 1, *shape_img)  , dtype = ref_type, device= device)
    batch_weights   = torch.zeros((len(dataloader), 1, 1, 1)        , dtype = ref_type, device= device)

    for k,batch in enumerate(dataloader):
        batch = batch.to(device, dtype = ref_type)                              # Move the batch to GPU
        batch = batch.unsqueeze_(1)                                             # (N,h,w) -> (N,1,h,w)
        batch_means[    k,  ...]  = torch.mean(batch, dim = 0).unsqueeze(0)     # bring back to shape (1,1,h,w)
        batch_weights[  k,  ...]  = batch.size(0)
            
    #del batch                                              # Free up GPU memory
    return torch.sum(batch_means * batch_weights, dim=0) / torch.sum(batch_weights)
    
def batch_axis0_mean_rolling(dataloader, ref_type, device='cuda'):
    """
    array is split into batches using suppiled data loader. because full array might not fit into VRAM
    total array mean is reconstructed from means of batches using batch lengths as weights.
    here 'rolling/moving' method is employed, where two images are averaged: all prev and latest batch mean.
    NOTE: everything is reshaped into (X,1,H,W) shape for torch to work.
    """
    weighted_mean   = None
    total_weight    = 0
    
    for batch in dataloader:
        batch = batch.to(device, dtype = ref_type)              # Move the batch to GPU
        batch = batch.unsqueeze_(1)                             # (N,h,w) -> (N,1,h,w)
        batch_mean = torch.mean(batch, dim = 0).unsqueeze(0)    # mean dim 0, bring back to shape (1,1,h,w)
        batch_weight = batch.size(0)                            # note: you dont need squeeze/unsqueeze here, 
                                                                # note: only if using convolutions after
        if weighted_mean is None:                         # Update the weighted mean
            weighted_mean = batch_mean.clone()
        else:
            weighted_mean = (weighted_mean * total_weight + batch_mean * batch_weight) / (total_weight + batch_weight)

        total_weight += batch_weight
    #del batch                                              # Free up GPU memory
    return weighted_mean