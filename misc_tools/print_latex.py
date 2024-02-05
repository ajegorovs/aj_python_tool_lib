
import numpy as np
from IPython.display import display, Math
from latexifier import latexify

def print_tex(*args,column = True, style_fraction = 'inline'):
    a = ''
    for arg in args:
        if type(arg) != str:
            arg = latexify(arg, newline=False, arraytype="bmatrix", column = column, style_fraction = style_fraction)
        else:
            arg = r'\text{' + arg + '} '
        a += arg
    display(Math(a))


print_tex('(Latex print example) I = : ', np.diag(range(3)), column=False)

