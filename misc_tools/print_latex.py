import numpy as np
from IPython.display import display, Math
from latexifier import latexify
import warnings

def string_matrix(matrix):
    """Format a matrix of strings using LaTeX syntax"""
    body_rows = [' & '.join(map(str,row)) for row in matrix]
    body_rows_join  =    r' \\ '.join(body_rows)
    return r"\begin{bmatrix}" + body_rows_join + r"\end{bmatrix}"

def print_tex(*args,column = False, style_fraction = 'inline', frmt = '{:3.6f}'):

    a = ''
    for arg in args:
        if type(arg) != str:
            if arg.dtype.type is not np.str_:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    arg = latexify(arg, newline=False, arraytype="bmatrix", 
                                   column = column, style_fraction = style_fraction,
                                   frmt = frmt)
            else:
                arg = string_matrix(arg)
        # else:
        #     arg = r'\text{' + arg + '} '
        a += arg
    display(Math(a))

print("input example : ")
print(r">>> arr_T = np.array([[r'\vec{v}_1', r'\vec{v}_2']]).T")
print(r">>> print_tex(arr_T,'=', np.arange(1,5).reshape(2,-1)/4, r'; symbols: \otimes, \cdot,\times')")
print("output: ")
arr_T = np.array([[r'\vec{v}_1', r'\vec{v}_2']]).T
print_tex(arr_T,'=', np.arange(1,5).reshape(2,-1)/4, r'; symbols: \otimes, \cdot,\times')