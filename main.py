import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

import time
import os
import psutil

from preprocesser import *
from reader import *
from eda import *

def get_memory_usage(prefix = ""):
    process = psutil.Process(os.getpid())
    print("%s %.2f GB" % (prefix, (process.memory_info().rss / (1024.0 ** 3))))

def print_function(x):
    print(x)

def calculate_grouped_log_demand(df, group_order, suffix):
    grouped_df = df.groupby(group_order, as_index=False).agg({'log_demand' : np.mean})
    grouped_df = grouped_df.rename(columns={'log_demand' : 'log_demand_' + suffix })
    return grouped_df

def calculate_and_merge_grouped_log_demand(df, dft, group_order, suffix):

    grouped_df = calculate_grouped_log_demand(df, group_order, suffix)

    merged_df = pd.merge(grouped_df,
                         df,
                         on=group_order,
                         how='inner',
                         copy=False)

    return merged_df


def calculate_and_merge_multiple_grouped_log_demand(dfs, df_train_name):

    get_memory_usage("before adding log_demand")

    dfs[df_train_name]['log_demand'] = np.float16(np.log1p(dfs[df_train_name]['Demanda_uni_equil']))

    get_memory_usage("before adding p_log_demand")

    dfs[df_train_name] = calculate_and_merge_grouped_log_demand(dfs[df_train_name], dfs['test'], ['Producto_ID'], 'p')

    get_memory_usage("before adding pr_log_demand")
    dfs[df_train_name] = calculate_and_merge_grouped_log_demand(dfs[df_train_name], dfs['test'], ['Producto_ID', 'Ruta_SAK'], 'pr')

    get_memory_usage("before adding pca_log_demand")
    dfs[df_train_name] = calculate_and_merge_grouped_log_demand(dfs[df_train_name], dfs['test'], ['Producto_ID', 'Cliente_ID', 'Agencia_ID'], 'pca')

    get_memory_usage("done merging")


def script(data_frames):

    #       Demanda_uni_equil
    #count       7.418046e+07
    #mean        7.224564e+00
    #std         2.177119e+01
    #min         0.000000e+00
    #25%         2.000000e+00
    #50%         3.000000e+00
    #75%         6.000000e+00
    #max         5.000000e+03
    #Demanda_uni_equil    0
    #dtype: int64

    mean_log_demand =  np.mean(np.log1p(data_frames['train']['Demanda_uni_equil']))

    get_memory_usage("before dropping unnecessary columns")
    data_frames['train'].drop(
        ['Semana',
         'Canal_ID',
         'Venta_uni_hoy',
         'Venta_hoy',
         'Dev_uni_proxima',
         'Dev_proxima',
         'log_demand'],
        axis=1,
        inplace=True)
    #data_frames['train'] = data_frames['train'][['Producto_ID', 'Cliente_ID', 'Ruta_SAK', 'Agencia_ID',  'log_demand_pca', 'log_demand_pr', 'log_demand_p']]

    #data_frames['test'].drop(['Semana', 'Canal_ID'], axis=1, inplace=True)

    get_memory_usage("before merging log_demand_p")

    # add log_demand_p column
    data_frames['test'] = pd.merge(data_frames['test'],
                                   data_frames['train'][['Producto_ID', 'log_demand_p']].drop_duplicates(subset=['Producto_ID']),
                                   on=['Producto_ID'],
                                   how='left',
                                   copy=False)

    get_memory_usage("before merging log_demand_pr")

    #add log_demand_pr column
    data_frames['test'] = pd.merge(data_frames['test'],
                                   data_frames['train'][['Producto_ID', 'Ruta_SAK', 'log_demand_pr']].drop_duplicates(subset=['Producto_ID', 'Ruta_SAK'], keep='last'),
                                   on=['Producto_ID', 'Ruta_SAK'],
                                   how='left',
                                   copy=False)

    get_memory_usage("before merging log_demand_pca")

    #add log_demand_pca column
    data_frames['test'] = pd.merge(data_frames['test'],
                                   data_frames['train'][['Producto_ID', 'Cliente_ID', 'Agencia_ID',  'log_demand_pca']].drop_duplicates(subset=['Producto_ID', 'Cliente_ID', 'Agencia_ID'], keep='last'),
                                   on=['Producto_ID', 'Cliente_ID', 'Agencia_ID'],
                                   how='left',
                                   copy=False)

    #TODO: test gc.collect()

    # create dummy column
    data_frames['test']['Demanda_uni_equil'] = np.NaN

    # fill in demand using optimized pca + pr
    optimized_vars = [0.72284372, 0.19369224, 0.09257224]
    null_demand_indicies = data_frames['test']['Demanda_uni_equil'].isnull()
    non_null_pca_indicies = data_frames['test']['log_demand_pca'].notnull()
    non_null_pr_indicies = data_frames['test']['log_demand_pr'].notnull()
    intersected_indicies = null_demand_indicies & non_null_pca_indicies & non_null_pr_indicies
    data_frames['test'].loc[intersected_indicies, 'Demanda_uni_equil'] =\
        optimized_vars[0] * data_frames['test'].loc[intersected_indicies, 'log_demand_pca'].apply(np.expm1) +\
        optimized_vars[1] * data_frames['test'].loc[intersected_indicies, 'log_demand_pr'].apply(np.expm1) +\
        optimized_vars[2]

    print(data_frames['test'].iloc[180])


    # fill in demand using pca
    optimized_vars = [0.82172756, 0.45792182]
    null_demand_indicies = data_frames['test']['Demanda_uni_equil'].isnull()
    non_null_pca_indicies = data_frames['test']['log_demand_pca'].notnull()
    intersected_indicies = null_demand_indicies & non_null_pca_indicies
    data_frames['test'].loc[intersected_indicies, 'Demanda_uni_equil'] = optimized_vars[0] * data_frames['test'].loc[intersected_indicies, 'log_demand_pca'].apply(np.expm1) + optimized_vars[1]

    print(data_frames['test'].iloc[180])

    # fill in demand using pr
    optimized_vars = [0.96057857, 0.612459958111]
    null_demand_indicies = data_frames['test']['Demanda_uni_equil'].isnull()
    non_null_pr_indicies = data_frames['test']['log_demand_pr'].notnull()
    intersected_indicies = null_demand_indicies & non_null_pr_indicies
    data_frames['test'].loc[intersected_indicies, 'Demanda_uni_equil'] = optimized_vars[0] * data_frames['test'].loc[intersected_indicies, 'log_demand_pr'].apply(np.expm1) + optimized_vars[1]

    print(data_frames['test'].iloc[180])

    # fill in demand using p
    optimized_vars = [0.97724085,  0.682583674679]
    null_demand_indicies = data_frames['test']['Demanda_uni_equil'].isnull()
    non_null_p_indicies = data_frames['test']['log_demand_p'].notnull()
    intersected_indicies = null_demand_indicies & non_null_p_indicies
    data_frames['test'].loc[null_demand_indicies, 'Demanda_uni_equil'] = optimized_vars[0] * data_frames['test'].loc[intersected_indicies, 'log_demand_p'].apply(np.expm1) + optimized_vars[1]

    print(data_frames['test'].iloc[180])

    null_demand_indicies = data_frames['test']['Demanda_uni_equil'].isnull()

    optimized_vars = [0.87969229, 0.43052957]
    data_frames['test'].loc[null_demand_indicies, 'Demanda_uni_equil'] = optimized_vars[0] * np.expm1(mean_log_demand) + optimized_vars[1]

    print(data_frames['test'].iloc[180])

    print(data_frames['test'].loc[:, ['Demanda_uni_equil']].isnull().values.any())

    data_frames['test'].loc[:, ['id', 'Demanda_uni_equil']].to_csv(os.path.join('data','submission.csv'), index=False)

    '''
    # TODO: see if iloc is faster than loc using these! V V V
    test_df_p_index = data_frames['test'].columns.get_loc('log_demand_p')
    test_df_pr_index = data_frames['test'].columns.get_loc('log_demand_pr')
    test_df_pca_index = data_frames['test'].columns.get_loc('log_demand_pca')

    def log_demand_func(row):

        log_demand_p_value = row[test_df_p_index]
        log_demand_pr_value = row[test_df_pr_index]
        log_demand_pca_value = row[test_df_pca_index]

        if not np.isnan(log_demand_pca_value):
            return log_demand_pca_value
        elif not np.isnan(log_demand_pr_value):
            return log_demand_pr_value
        elif not np.isnan(log_demand_p_value):
            return log_demand_p_value
        else:
            print("no log_demand value")
            return 1
    col =  data_frames['test'].apply(log_demand_func, axis=1)
    print(col)
    '''

    ##submission = np.zeros(len(df_test))


def check_submission_file():
    df = pd.read_csv(os.path.join('data','submission.csv'))
    print(df.shape)
    #6,999,251
    quit()


if __name__ == '__main__':

    #check_submission_file()

    '''
    PCA+PR =  0.72284372*x1 + 0.19369224*x2 + 0.09257224
    PCA = 0.82172756 * x + 0.45792182
    PR = 0.96057857  * x + 0.612459958111
    P = 0.97724085 * x + 0.682583674679
    NoneType = 0.87969229 *x +  0.43052957
    '''


    data_files = {}
    data_files['train'] = {'Agencia_ID': np.int16,  'Producto_ID': np.uint16, 'Ruta_SAK' : np.uint16, 'Cliente_ID' : np.uint16, 'Demanda_uni_equil': np.int16}
    # 'Semana': np.int8, 'Canal_ID': np.int8, 'Venta_uni_hoy': np.uint16, 'Dev_uni_proxima': np.int32,

    data_files['test'] = {'id': np.int32, 'Agencia_ID': np.int16, 'Producto_ID': np.uint16, 'Ruta_SAK' : np.uint16, 'Cliente_ID' : np.uint16}
    # 'Semana': np.int8, 'Canal_ID': np.int8,

    #'cliente_tabla',
    #'producto_tabla',
    #'town_state',

    data_frames = {}

    t_start = time.time()
    for data_file_name, data_file_dtypes in data_files.items():
        data_frames[data_file_name] =  try_get_df_from_csv(data_file_name, data_file_dtypes)

    t_end = time.time()

    print("read data files: %.2f sec" % (t_end - t_start))


    # MEMORY LEAK
    #for  data_file_name, data_frame in data_frames.items():
    #    data_frame.info(memory_usage='deep')


    t_start = time.time()
    calculate_and_merge_multiple_grouped_log_demand(data_frames, 'train')
    t_end = time.time()

    print("add grouped log_demand: %.2f sec" % (t_end - t_start))


    script(data_frames)

    '''
    data_frames['producto_tabla'] = preprocess_producto_tabla(data_frames['producto_tabla'])

    with pd.option_context('display.max_rows', 1000, 'display.width', 1000):
        #print(data_frames['producto_tabla']['product_name'].value_counts(dropna=False))
        print data_frames['train']

    sns.distplot(data_frames['producto_tabla']['weight'].dropna())
    plt.show()
    '''


get_memory_usage()

#TODO: eda
