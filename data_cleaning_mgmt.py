# This file is for cleaning up the data based on various heuristics that have
# been deemed appropriate for better model training. Note that some data is
# also removed or modified manually based on visual inspection (e.g. filling
# brief instances of empty values where the value is the same before and after
# the interruption, removing obvious sensor errors that are not sudden enough
# to be captured by the definition of a sudden spike, etc).

# It should be highlighted that this part of the process will be very specific
# to each individual data set: based on the quality of the data itself, the
# behaviour of the plant and the source water, etc. As a result this file has
# selected values that only apply to this data set. The operations performed
# below apply ONLY to this data set.

import numpy as np

# File locations
f_data = "Data/"
file_npy_dec = f_data + "stn_1707-2111_decimal.npy"
# Append "_mod" for modified file after cleaning
file_npy_mod = file_npy_dec[:file_npy_dec.rfind('.')] + '_mod' + file_npy_dec[file_npy_dec.rfind('.'):]

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

# This function just creates a copy of the decimal year array of the same
# name with '_mod' appended at the end to indicate it is now modified data
def func_create_copy(file_npy_dec, file_npy_mod):
    array_np = np.load(file_npy_dec)
    np.save(file_npy_mod, array_np)    


# Function to perform the data cleanup (see the class data_modifier below)
def func_clean_data(file_npy_dec, file_npy_mod):
    # Create a copy of the data and create an instance of data_modifier
    func_create_copy(file_npy_dec, file_npy_mod)
    Mod_1 = data_modifier(file_npy_mod)
    # Perform all the cleaning operations
    Mod_1.func_mod_af_flow()    # col 22 (cols 21 and 23 are fine)
    Mod_1.func_mod_temp()       # col 1
    Mod_1.func_mod_turb()       # col 2
    Mod_1.func_mod_cond()       # col 3
    Mod_1.func_mod_ph()         # cols 4, 6, 7
    Mod_1.func_mod_af_turb()    # cols 8, 10, 12
    Mod_1.func_mod_pax()        # col 15
    Mod_1.func_mod_inflow()     # col 20
    # Alum and TSS are fine and require no modification
    # Save the modified file
    Mod_1.f_save()


# Class for modifying data with different functions for different columns
class data_modifier:
    def __init__(self, file_npy_mod):
        # Load file with data
        self.file = file_npy_mod
        self.data = np.load(self.file)
        # Create variables to store indices of rows that need to be modified
        self.af_turb_idx = np.zeros((len(self.data), 3), dtype=int)
        self.pax_idx = np.zeros(len(self.data), dtype=int)
        self.af_flow_idx = np.zeros(len(self.data), dtype=int)
        # Variable to store indices of rows that need to be deleted
        self.del_idx = np.zeros(len(self.data), dtype=int)

# Modify the data in this class instance and save to the _mod file when finished
    def f_save(self):
        # Actiflo 2 flow is modified previously in func_mod_af_flow
        # Set marked turbidity values to 0
        self.data[:, (8, 10, 12)] *= (1 - self.af_turb_idx)
        # Trim pax readings
        self.data[self.pax_idx, 15] = self.max_pax_reading
        # Delete rows marked for deletion
        self.data = self.data[np.logical_not(self.del_idx)]
        # Save modified data
        np.save(self.file, self.data)
        print(f'\nFile modified and saved to {f_data + file_npy_mod}')

# Adjust the temperature data to remove anomalous spikes and sensor errors
    def func_mod_temp(self, delta_threshold=0.1):
        # Make a vector called delta which stores the difference between all
        # adjacent temperature values in the data set
        delta = self.data[1:, 1] - self.data[:-1, 1]
        delta = np.insert(delta, 0, 1e-4) # padding to make array same length as original
        delta = np.abs(delta)
        # Mark rows for deletion if either:
        #   a) the difference between the temperature at this time step and the previous
        #      time step exceeds a given threshold (i.e. 0.1 degrees C), OR
        #   b) the temperature does not change at all from the previous time step
        #      (given the number of decimal places recorded, this likely indicates
        #      a sensor error)
        del_new = ((delta > delta_threshold) | (delta == 0))
        self.del_idx += del_new
        print(f'Temperature evaluated; {np.sum(self.del_idx > 0)} total rows marked for deletion.')

# Adjust the influent turbidity data to remove spikes and sensor errors
    def func_mod_turb(self, delta_threshold=2, detection_limit=100):
        # Make a vector called delta which stores the difference between all
        # adjacent turbidity values in the data set
        delta = self.data[1:, 2] - self.data[:-1, 2]
        delta = np.insert(delta, 0, 1e-4) # padding to make array same length as original
        delta = np.abs(delta)
        # Mark rows for deletion if either:
        #   a) the difference between the turbidity at this time step and the previous
        #      time step exceeds a given threshold (i.e. 2 NTU), OR
        #   b) the turbidity does not change at all from the previous time step AND
        #      it is NOT at the detection limit of 100 NTU (given the number of decimal 
        #      places recorded, this likely indicates a sensor error)
        del_new = ((delta > delta_threshold) | ((delta == 0) & (self.data[:, 2] < detection_limit)))
        self.del_idx += del_new
        print(f'Turbidity evaluated; {np.sum(self.del_idx > 0)} total rows marked for deletion.')

# Adjust the conductivity data to remove anomalous drops and spikes
    def func_mod_cond(self, delta_threshold=5, detection_limit=200):
        # Make a vector called delta which stores the difference between all
        # adjacent conductivity values in the data set
        delta = self.data[1:, 3] - self.data[:-1, 3]
        delta = np.insert(delta, 0, 1e-4) # padding to make array same length as original
        delta = np.abs(delta)
        # Mark rows for deletion if either:
        #   a) the difference between the conductivity at this time step and the previous
        #      time step exceeds a given threshold (i.e. 5 uS/cm), OR
        #   b) the conductivity does not change at all from the previous time step AND
        #      it is NOT at the detection limit of 200 uS/cm (given the number of decimal 
        #      places recorded, this likely indicates a sensor error)
        del_new = ((delta > delta_threshold) | ((delta == 0) & (self.data[:, 3] < detection_limit)))
        self.del_idx += del_new
        print(f'Conductivity evaluated; {np.sum(self.del_idx > 0)} total rows marked for deletion.')
        
# Adjust the pH data to remove anomalous drops and spikes
    def func_mod_ph(self, delta_thresholds=(0.1, 0.3, 0.3), min_zero_delta_steps=3):
        # Make a vector called delta which stores the difference between all
        # adjacent pH values in the data set
        delta = np.empty((min_zero_delta_steps), dtype=object)
        insert_tuple = ()
        # Column 4 is the influent pH, columns 6 and 7 are two pH sensors
        # in parellel in the the coagulation basin
        for i in range(0, min_zero_delta_steps):
            insert_tuple += (0,)
            delta[i] = self.data[i+1:, (4, 6, 7)] - self.data[:-(i+1), (4, 6, 7)]
            delta[i] = np.insert(delta[i], insert_tuple, 1e-4, axis=0)
            delta[i] = np.abs(delta[i])
        # Mark rows for deletion if either:
        #   a) the difference between the pH at this time step and the previous
        #      time step exceeds a given threshold (i.e. 0.1 in the influent
        #      and 0.3 in the coagulation basin), OR
        #   b) the pH does not change at all for some number of consecutive time
        #      steps (min_zero_delta_steps, i.e. 3)
        del_new = ((delta[0] > delta_thresholds).any(axis=1) | (np.sum(delta[:]) == 0).any(axis=1))
        self.del_idx += del_new
        print(f'pH evaluated; {np.sum(self.del_idx > 0)} total rows marked for deletion.')
        
# Adjust the streaming current data to remove anomalous drops and spikes
    def func_mod_scm(self, delta_threshold=40):
        # Make a vector called delta which stores the difference between all
        # adjacent SCM values in the data set
        delta = self.data[1:, 5] - self.data[:-1, 5]
        delta = np.insert(delta, 0, 1e-4) # padding to make array same length as original
        delta = np.abs(delta)
        # Mark rows for deletion if either:
        #   a) the difference between the streaming current at this time step and the
        #      previous time step exceeds a given threshold (i.e. 40), OR
        #   b) the streaming current does not change at all from the previous time
        #      step (given the number of decimal places recorded, this likely
        #      indicates a sensor failure)
        del_new = ((delta > delta_threshold) | (delta == 0))
        self.del_idx += del_new
        print(f'Streaming current evaluated; {np.sum(self.del_idx > 0)} total rows marked for deletion.')

# Adjust the Actiflo turbidity data to remove negative values and any other
# anomalies, as well as to set turbidity to zero when flow in the basin is zero
    def func_mod_af_turb(self, detection_limit=100, min_zero_delta_steps=3):
        # Make a vector called delta which stores the difference between all
        # adjacent turbidity values in the data set
        delta = np.empty((min_zero_delta_steps), dtype=object)
        insert_tuple = ()
        for i in range(0, min_zero_delta_steps):
            insert_tuple += (0,)
            delta[i] = self.data[i+1:, (8, 10, 12)] - self.data[:-(i+1), (8, 10, 12)]
            delta[i] = np.insert(delta[i], insert_tuple, 1e-4, axis=0)
            delta[i] = np.abs(delta[i])
        # Mark rows for deletion if either:
        #   a) the recorded turbidity is above the detection limit (100 NTU), OR
        #   b) the recorded turbidity is negative, OR
        #   c) the turbidity does not change at all for some number of consecutive
        #      time steps (min_zero_delta_steps, i.e. 3) AND the turbidity is 
        #      nonzero (given the number of decimal places recorded, this likely 
        #      indicates a sensor failure)
        del_new = ((self.data[:, (8, 10, 12)] > detection_limit).any(axis=1) |
                   (self.data[:, (8, 10, 12)] < 0).any(axis=1) |
                   (((np.sum(delta[:]) == 0) & ((self.data[:, -3:] != 0))).any(axis=1)))
        self.del_idx += del_new
        # Mark rows to be set to zero if the flow in the basin is zero at the
        # given time step.
        self.af_turb_idx = (self.data[:, -3:] == 0)
        print(f'Actiflo turbidity evaluated; {np.sum(self.del_idx > 0)} total rows marked for deletion.')
        print(f'Additionally, {np.sum(self.af_turb_idx)} turbidity data points will be set to zero.')

# No alum modifications needed
    # def func_mod_alum(self):
        # pass

# Adjust the PAX data to trim excessive values
    def func_mod_pax(self, max_pax_reading=20):
        self.max_pax_reading = max_pax_reading
        # Mark rows to be set to max_pax_reading (i.e. 20 mg/L) if they exceed this value.
        self.pax_idx = (self.data[:, 15] > self.max_pax_reading)
        print(f'PAX evaluated; {np.sum(self.pax_idx)} PAX data points will be trimmed to {max_pax_reading}.')

# Adjust the inflow data to remove data at plant shutdown
    def func_mod_inflow(self, min_threshold=60):
        # Mark rows for deletion if the inflow is lower than a given threshold
        # (i.e. 60 dam3/d) that suggests plant shutdown
        del_new = (self.data[:, 20] < min_threshold)
        self.del_idx += del_new
        print(f'Inflow evaluated; {np.sum(self.del_idx > 0)} total rows marked for deletion.')

# Adjust the Actiflo basin flow data to replace erroneous low flow rates
# with zeros (i.e. where neighbouring time steps both report zero flow)
    def func_mod_af_flow(self):
        for basin in [3, 2, 1]:
            data_plus1 = np.insert(self.data[:, -basin], 0, 0)
            data_plus1 = np.delete(data_plus1, -1)
            data_minus1 = np.insert(self.data[:, -basin], -1, 0)
            data_minus1 = np.delete(data_minus1, 0)
            self.af_flow_idx = ((self.data[:, -basin] > 0) & ((data_plus1 == 0) & (data_minus1 == 0)))
            self.data[self.af_flow_idx, -2] = 0
            print(f'Actiflo basin {4 - basin} flow evaluated; {np.sum(self.af_flow_idx)} flow data points will be set to zero.')