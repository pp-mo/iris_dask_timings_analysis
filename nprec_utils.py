import numpy as np
import numpy.lib.recfunctions as nprec

#
# ===================== STRUCTURED ARRAY UTILS =========================
#

def nprec_from_name_column_pairs(*name_column_pairs):
    names_cols = [(name, np.array(col)) for name, col in name_column_pairs]
    names = [name for name, col in names_cols]
    cols = [col for name, col in names_cols]
    dtypes = [col.dtype for col in cols]
    data = [tuple(els) for els in zip(*cols)]
    array_dtype = [(name, dtype) for name, dtype in zip(names, dtypes)]
    result = np.array(data, array_dtype)
    return result


def nprec_print(array, names_widths_formats=None, line_format='{:s}'):
    """
    Print columns of a recarray as a formatted table.

    Args:

    * array (numpy recarray):
        data to print
    * names_widths_formats (iterable of 3-tuples):
        Each element specifies one column to print,
        It is either a name, or a tuple :
            (name [, width [, format]])
        where:
            name (string) is the name of the column in 'data'
            width (int) is the width of the printed output column
            format (string) is a type format specifier, e.g. 'i' or '+07.5f'
        If None, the defaults is to print all columns.
        Default column width is 20.
        Default column format is 'f', 'd' or 's' depending on column dtype.
    * line_format (string):
        A format for every output line (as a single string arg).
    * no_print (bool):
        If set, do not print the results but just return them.

    Returns:

        A list of strings, one per output line.

    """
    output_lines = []
    header_names = []
    row_fmt = ''
    col_formats = []
    # Pre-process specifications for defaults...
    names_widths_formats = names_widths_formats
    if names_widths_formats is None:
        names_widths_formats = [(name,) for name in array.dtype.names]
    else:
        names_widths_formats = [(el,) if isinstance(el, basestring) else el
                                for el in names_widths_formats]
    nwfs = []
    for nwf in names_widths_formats:
        name = nwf[0]
        if len(nwf) >= 2:
            width = nwf[1]
        else:
            # No width: use default of 20 (for now)
            width = 20
        if len(nwf) > 2:
            format = nwf[2]
        else:
            # Use default for type.
            col_dtype = array[name].dtype
            col_kind = col_dtype.kind
            if col_kind == 'f':
                format = 'f'
            elif col_kind in 'iub':
                format = 'd'
            elif col_kind == 'S':
                format = 's'
            else:
                msg = ('Default format not defined for Column {!r}, '
                       'which has dtype {!r}, kind {!r}')
                raise ValueError(msg.format(name, col_dtype, col_kind))
        nwfs.append((name, width, format))
        header_names.append(name.rjust(width))
        row_fmt += '{{:{}s}}'.format(width)

    def add_line(*el_strings):
        el_strings = [el.rjust(width)
                      for el, (name, width, format)
                      in zip(el_strings, nwfs)]
        print line_format.format(row_fmt.format(*el_strings))

    add_line(*header_names)
    dashes = [('-' * min(4, width -1)).rjust(width)
              for name, width, format in nwfs]
    add_line(*dashes)

    def full_format(type_format):
        return '{{:{}}}'.format(type_format)

    for row in array:
        cols_strs = [full_format(format).format(row[name])
                     for name, width, format in nwfs]
        add_line(*cols_strs)

