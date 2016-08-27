import os.path
import exceptions
import zipfile
import pandas as pd


# =========================================
# Timing Statistics
# =========================================
# read_hdf = 5.16s
# read_csv = 44.61s
# read_csv and write_hdf = 47.20 s

def try_get_df_from_csv(file_name_without_ext, data_file_dtypes, sub_folder = 'data'):

    #TODO: assert if ext on file found

    if sub_folder is not None:
        relative_file_path = os.path.join(sub_folder, file_name_without_ext)
    else:
        relative_file_path = file_name_without_ext

    # we have an optimized version of the data that will load faster, use it
    if os.path.isfile(relative_file_path + '.csv.hdf'):

        print("found *.csv.hdf! reading file: %s" % relative_file_path + '.csv.hdf')

        df = pd.read_hdf(relative_file_path + '.csv.hdf', file_name_without_ext, dtype=data_file_dtypes)

        return df


    # we have a compressed version of the data that needs to be extracted before we use it
    if os.path.isfile(relative_file_path + '.csv.zip') and not os.path.isfile(relative_file_path + '.csv'):

        print ("found *.csv.zip! writing out *.csv: %s" % relative_file_path + '.csv.zip')

        with zipfile.ZipFile(relative_file_path + '.csv.zip','r') as zip_file:
            zip_file.extractall()


    # we have the data in csv form, write out optimized form for next run
    if os.path.isfile(relative_file_path + '.csv'):

        df = pd.read_csv(relative_file_path + '.csv', dtype=data_file_dtypes)

        print ("found *.csv! writing out *.csv.hdf: %s" % relative_file_path + '.csv')

        df.to_hdf(relative_file_path + '.csv.hdf', file_name_without_ext, mode='w',complib='blosc')


        return df

    # we could not find the data, raise error
    raise exceptions.IOError('*.csv file or any of its derivatives not found. Have you downloaded the necessary csv files?')


