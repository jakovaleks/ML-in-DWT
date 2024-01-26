# This file is for converting .csv files into one .npy file.

import pandas as pd
import numpy as np
import datetime as dt
import os, fnmatch

# File locations
f_root = "/home/user1/Documents/ML_DWT/"
f_data = "Data/"
f_csv = f"{f_root}{f_data}stn_csv/"

# The data are stored as .csv files from each month;
# For example, stn_1809.csv is the data for September 2018.
# The first column is a string representing the date and time as DD-MM-YYYY H:MM.
# The remaining columns are the variables measured at that timestamp
# (e.g. influent temperature, alum dose, effluent turbidity, etc.).
list_csv = sorted(fnmatch.filter(os.listdir(f_csv), "stn_*.csv"))
start_date = list_csv[0][list_csv[0].find('_')+1:list_csv[0].find('.')]
end_date = list_csv[-1][list_csv[-1].find('_')+1:list_csv[-1].find('.')]
# Data ranges from July 2017 to November 2021
file_npy = f"{f_root}{f_data}stn_{start_date}-{end_date}.npy"
# Append "_decimal" for the modified file with decimal timestamps
file_npy_dec = file_npy[:file_npy.rfind('.')] + '_decimal' + file_npy[file_npy.rfind('.'):]


# Function to convert all the individual .csv files into one numpy array
def func_csv_to_npy(f_csv, list_csv, file_npy):
    print('Copying csv files to npy file...')
    array_pd = pd.DataFrame()
    # Iterate through all the files in list_csv
    for date_file in list_csv:
        data_pd = pd.read_csv(f_csv + date_file, encoding = 'unicode_escape', engine ='python')
        # Create a pandas array of all the data
        if date_file == list_csv[0]:
            array_pd = data_pd
        else:
            array_pd = pd.concat((array_pd, data_pd))
        print(f'Finished copying data for {date_file}.')
    # Convery pandas array to numpy array
    array_np = array_pd.to_numpy()
    print('Done.\n')
    
    # Additionally change date format to YYYY-MM-DD HH:MM
    # (since OpenOffice Calc created them as DD-MM-YYYY H:MM)
    if array_np[0, 0][2] == '-':
        print('Converting dates in npy file to YMD...')
        # Manually extract dates/times from strings
        for i in range(len(array_np)):
            # For times ranging from 0:00 to 9:59
            if len(array_np[i, 0]) == 15:
                array_np[i, 0] = array_np[i, 0][6:10] + '-' + array_np[i, 0][3:6] + \
                                 array_np[i, 0][:2] + ' 0' + array_np[i, 0][11:]
            # For times ranging from 10:00 to 23:59
            elif len(array_np[i, 0]) == 16:
                array_np[i, 0] = array_np[i, 0][6:10] + '-' + array_np[i, 0][3:6] + \
                                 array_np[i, 0][:2] + array_np[i, 0][10:]
    np.save(file_npy, array_np, allow_pickle=True)
    print(f'Saved to {file_npy}.\n')


# Function to take the array from above and turn it into a float array with
# decimal years in the first column.
def func_npy_to_npy_decimal(file_npy, file_npy_dec):
    print('Converting date column in array from string to decimal dates...')
    array_np = np.load(file_npy, allow_pickle=True)
    # Create a new array and copy all variables except timestamps
    array_np_decimal = np.empty(shape=array_np.shape, dtype=np.float64)
    array_np_decimal[:, 1:] = array_np[:, 1:]
    # Calculate the decimal time for each timestamp and add it to the new array
    for i in range(len(array_np)):
        array_np_decimal[i, 0] = func_year_frac(array_np[i, 0])
    np.save(file_npy_dec, array_np_decimal)
    print(f'Saved to {file_npy_dec}.\n')


# Naive implementation of converting datetime object to decimal year.
# Note that the SCADA system fails to record 60 values in the fall
# due to being unable to store two of the same timestamp.
def func_year_frac(date):
    y, mo, d, h, mi = int(date[:4]), int(date[5:7]), int(date[8:10]), \
                      int(date[11:13]), int(date[14:])
    dt1 = dt.datetime(y, 1, 1)
    dt2 = dt.datetime(y, mo, d, h, mi)
    dtd = dt2 - dt1
    # For leap years
    if (y % 4 == 0) & (y % 100 > 0) | (y % 400 == 0):
        y_frac = y + (dtd.days / 366) + (dtd.seconds / (86400*366))
    # For non-leap years
    else:
        y_frac = y + (dtd.days / 365) + (dtd.seconds / (86400*365))
    return y_frac


#%% MAIN

if __name__ == '__main__':
    func_csv_to_npy(f_csv, list_csv, file_npy)
    func_npy_to_npy_decimal(file_npy, file_npy_dec)
    
# ================================================================ #
# stn_1707_2111 indices ========================================== #
# 0. decimal_time; 1. temperature_in; 2. turbidity_in; =========== #
# 3. conductivity; 4. ph_in; 5. scm; 6. coagulation_ph_1; ======== #
# 7. coagulation_ph_2; 8. actiflo_1_turbidty; 9. actiflo_1_tss; == #
# 10. actiflo_2_turbidity; 11. actiflo_2_tss; ==================== #
# 12. actiflo_3_turbidity; 13. actiflo_3_tss; ==================== #
# 14. alum; 15. pax; 16. actiflo_1_polymer; ====================== #
# 17. actiflo_2_polymer; 18. actiflo_3_polymer; 19. silica; ====== #
# 20. inflow, 21. actiflo_1_flow; 22. actiflo_2_flow; ============ #
# 23. actiflo_3_flow; ============================================ #
# ================================================================ #