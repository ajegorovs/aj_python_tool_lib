import torch

def conv2D_dims(HW, kernel_size, stride = 1, padding = 0, output_padding = 0, dilation = 1):
    """ 
        Calculate dimensions of convolution output given parameters 
    """
    HW = HW[-2:]
    a = torch.tensor([(x)*2 if type(x)!= int else x for x in [kernel_size,stride,padding,output_padding,dilation]])
    kernel_size, stride, padding, output_padding, dilation = a

    HW_out = 1/stride*(HW + 2*padding - dilation * (kernel_size - 1) - 1) + 1
    return torch.floor(HW_out).to(int)

def conv2DT_dims(HW, kernel_size, stride= 1, padding= 0, output_padding= 0, dilation= 1):
    """ 
        Calculate dimensions of Transposed convolution output given parameters 
    """
    HW = HW[-2:]
    a = torch.tensor([(x)*2 if type(x)!= int else x for x in [kernel_size,stride,padding,output_padding,dilation]])
    kernel_size, stride, padding, output_padding, dilation = a
    HW_out = (HW - 1) * stride - 2* padding + dilation * (kernel_size - 1) + output_padding + 1
    return HW_out