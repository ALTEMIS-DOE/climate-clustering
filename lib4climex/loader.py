import netCDF4

def netcdf_reader(filename:str, varname:str):
    """ functino to read netcdf filename. Assume variables are under root.
        :param filename:  string netcdf file 
        :param varname:   string variable name
        :return: numpy.ndarray
    """
    nc = netCDF4.Dataset(filename, 'r')
    var = nc.variables[varname][:]
    nc.close()
    return var