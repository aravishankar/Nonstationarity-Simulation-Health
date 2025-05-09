import pandas as pd
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt
import tsfel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import metrics
#from tslearn import svm
from sklearn.metrics import mean_squared_error
import math
import pickle
import tensorflow as tf
from tensorflow import keras
from matplotlib.pyplot import figure
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM, RNN
from tensorflow.keras.utils import to_categorical
from time import time
import warnings
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from statistics import mean
from sklearn.utils import class_weight
from roufcp import roufCP
from scipy import stats
from scipy.stats import mannwhitneyu
from datetime import datetime
import string
from scipy.optimize import curve_fit, minimize
from scipy.stats import norm, uniform, expon
#from numba import jit
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')


#for detecting changepoints and duration
def identify_df_trends(df, column, frequency=None, window_size=20, identify='both'):
        """
        This function receives as input a pandas.DataFrame from which data is going to be analysed in order to
        detect/identify trends over a certain date range. A trend is considered so based on the window_size, which
        specifies the number of consecutive days which lead the algorithm to identify the market behaviour as a trend. So
        on, this function will identify both up and down trends and will remove the ones that overlap, keeping just the
        longer trend and discarding the nested trend.
        Args:
            df (:obj:`pandas.DataFrame`): dataframe containing the data to be analysed.
            column (:obj:`str`): name of the column from where trends are going to be identified.
            window_size (:obj:`int`, optional): number of samples from where market behaviour is considered a trend.
            identify (:obj:`str`, optional):
                which trends does the user wants to be identified, it can either be 'both', 'up' or 'down'.
        Returns:
            :obj:`pandas.DataFrame`:
                The function returns a :obj:`pandas.DataFrame` which contains the retrieved historical data from Investing
                using `investpy`, with a new column which identifies every trend found on the market between two dates
                identifying when did the trend started and when did it end. So the additional column contains labeled date
                ranges, representing both bullish (up) and bearish (down) trends.
        Raises:
            ValueError: raised if any of the introduced arguments errored.
        """
        print(column)
        print(window_size)
        if df is None:
            raise ValueError("df argument is mandatory and needs to be a `pandas.DataFrame`.")

        if not isinstance(df, pd.DataFrame):
            raise ValueError("df argument is mandatory and needs to be a `pandas.DataFrame`.")

        if column is None:
            raise ValueError("column parameter is mandatory and must be a valid column name.")

        if column and not isinstance(column, str):
            raise ValueError("column argument needs to be a `str`.")

        if isinstance(df, pd.DataFrame):
            if column not in df.columns:
                raise ValueError("introduced column does not match any column from the specified `pandas.DataFrame`.")
            else:
                if df[column].dtype not in ['int64', 'float64']:
                    raise ValueError("supported values are just `int` or `float`, and the specified column of the "
                                    "introduced `pandas.DataFrame` is " + str(df[column].dtype))

        if not isinstance(window_size, int):
            raise ValueError('window_size must be an `int`')

        if isinstance(window_size, int) and window_size < 3:
            raise ValueError('window_size must be an `int` equal or higher than 3!')

        if not isinstance(identify, str):
            raise ValueError('identify should be a `str` contained in [both, up, down]!')

        if isinstance(identify, str) and identify not in ['both', 'up', 'down']:
            raise ValueError('identify should be a `str` contained in [both, up, down]!')

        objs = list()

        up_trend = {
            'name': 'Up Trend',
            'element': np.negative(df[column])
        }

        down_trend = {
            'name': 'Down Trend',
            'element': df[column]
        }

        if identify == 'both':
            objs.append(up_trend)
            objs.append(down_trend)
        elif identify == 'up':
            objs.append(up_trend)
        elif identify == 'down':
            objs.append(down_trend)
            

        results = dict()

        for obj in objs:
            limit = None
            values = list()

            trends = list()

            for index, value in enumerate(obj['element'], 0):
                if limit and limit > value:
                    values.append(value)
                    limit = mean(values)
                elif limit and limit < value:
                    if len(values) > window_size:
                        min_value = min(values)

                        for counter, item in enumerate(values, 0):
                            if item == min_value:
                                break

                        to_trend = from_trend + counter
                        to_trend = min(to_trend, len(df.index.tolist()) - 1) #to avoid index out of bounds error 
                        #print(f"To_trend: {to_trend}")

                        trend = {
                            'from': df.index.tolist()[from_trend],
                            'to': df.index.tolist()[to_trend],
                        }

                        trends.append(trend)

                    limit = None
                    values = list()
                else:
                    from_trend = index

                    values.append(value)
                    limit = mean(values)

            results[obj['name']] = trends

        if identify == 'both':
            up_trends = list()

            for up in results['Up Trend']:
                flag = True

                for down in results['Down Trend']:
                    if down['from'] < up['from'] < down['to'] or down['from'] < up['to'] < down['to']:
                        if (up['to'] - up['from']).total_seconds() > (down['to'] - down['from']).total_seconds():
                            flag = True
                        else:
                            flag = False
                    else:
                        flag = True

                if flag is True:
                    up_trends.append(up)
            ### fix this 
            print(f"Uptrend is: {len(up_trends)}")
            labels = [letter for letter in string.ascii_uppercase[:len(up_trends)]]
            #print(f"Labels: {labels}")
            labels += [str(i).zfill(3) for i in range(1, len(up_trends) - len(labels) + 1)] #add more character for the labels
            print(f"Labels len: {len(labels)}")
            #print(len(labels))
            
            
            for up_trend, label in zip(up_trends, labels):
                for index, row in df[up_trend['from']:up_trend['to']].iterrows():
                    df.loc[index, 'Up Trend'] = label

            down_trends = list()

            for down in results['Down Trend']:
                flag = True

                for up in results['Up Trend']:
                    if up['from'] < down['from'] < up['to'] or up['from'] < down['to'] < up['to']:
                        if (up['to'] - up['from']).total_seconds() > (down['to'] - down['from']).total_seconds():
                            flag = True
                        else:
                            flag = False
                    else:
                        flag = True

                if flag is True:
                    down_trends.append(down)
            
            print(f"Downtrend is: {len(down_trends)}")
            labels = [letter for letter in string.ascii_uppercase[:len(down_trends)]]
            labels += [str(i).zfill(3) for i in range(1, len(down_trends) - len(labels) + 1)] #add more character for the labels
            print(f"Labels len: {len(labels)}")

            for down_trend, label in zip(down_trends, labels):
                for index, row in df[down_trend['from']:down_trend['to']].iterrows():
                    df.loc[index, 'Down Trend'] = label

            print("Return")
            return df
        elif identify == 'up':
            up_trends = results['Up Trend']

            up_labels = [letter for letter in string.ascii_uppercase[:len(up_trends)]]

            for up_trend, up_label in zip(up_trends, up_labels):
                for index, row in df[up_trend['from']:up_trend['to']].iterrows():
                    df.loc[index, 'Up Trend'] = up_label

            return df
        elif identify == 'down':
            down_trends = results['Down Trend']

            down_labels = [letter for letter in string.ascii_uppercase[:len(down_trends)]]

            for down_trend, down_label in zip(down_trends, down_labels):
                for index, row in df[down_trend['from']:down_trend['to']].iterrows():
                    df.loc[index, 'Down Trend'] = down_label

            return df
        
#populate the table with cpd and duration
def annonate_cp_duration(df):
    df['dates'] = pd.to_datetime(df['dates'])
    df['duration'] = np.nan
    df['cpd'] = 0
    freq = pd.Timedelta(seconds=0.25)  # 4Hz = 0.25 seconds per sample
    labels = df['Up Trend'].dropna().unique().tolist()
    for label in labels:
        dt_index = df[df['Up Trend'] == label].index[0]
        dt = df.loc[dt_index, 'dates']
        dt2 = None
        i = dt_index + freq
        while ((i > dt_index) & (i < df.index[-1])):
            dt2_row = df.loc[i]
            if ((dt2_row['Up Trend'] == label)):
                dt2 = dt2_row['dates']
            else:
                break
            i += freq
        if dt2 is not None:
            df.loc[df.dates == dt, 'cpd'] = int(1)
            duration = int((dt2 - dt).seconds/60.0)
            df.loc[df.dates == dt, 'duration'] = duration
    labels = df['Down Trend'].dropna().unique().tolist()
    for label in labels:
        dt_index = df[df['Down Trend'] == label].index[0]
        dt = df.loc[dt_index, 'dates']
        dt2 = None
        i = dt_index + freq
        while ((i > dt_index) & (i < df.index[-1])):
            dt2_row = df.loc[i]
            if ((pd.isna(dt2_row['Up Trend'])) & (dt2_row['Down Trend'] == label)):
                dt2 = dt2_row['dates']
            else:
                break
            i += freq
        if dt2 is not None:
            df.loc[df.dates == dt, 'cpd'] = int(1)
            duration = int((dt2 - dt).seconds/60.0)
            df.loc[df.dates == dt, 'duration'] = duration
    return df


##contains function for lstm classification modelling.
def find_changepoint(signal, type_of_cpd, min_size, jump):
    # PELT change point detection
    n = len(signal) #number of samples
    model = "l2"  # "l2", "rbf"
    if (type_of_cpd == "Pelt"):
        algo = rpt.Pelt(model=model, min_size=min_size, jump=jump).fit(signal) #min_size=12, jump=5
        predicted_cp = algo.predict(pen=3) #what is pen all about??
    elif (type_of_cpd == "Binary"):
        algo = rpt.Binseg(model=model, jump=1).fit(signal) #jump=1
        predicted_cp = algo.predict(pen=3) #what is pen all about??
    elif(type_of_cpd == "Window"):
        algo = rpt.Window(model=model, width=6).fit(signal) #width=6
        predicted_cp = algo.predict(pen=3) #what is pen all about??
    elif(type_of_cpd == "BottomUp"):
        algo = rpt.BottomUp(model=model).fit(signal)
        predicted_cp = algo.predict(pen=3) #what is pen all about??
    elif(type_of_cpd == "roufCP"): #gradual changepoint
        cpd_output = roufCP(delta = 3, w = 3).fit(signal, moving_window = 6, k = 2)
        predicted_cp = cpd_output['cp']
    #predicted_cp =  algo.predict(epsilon=3 * n * sigma**2)
    predicted_cp = predicted_cp[ : -1]
    actual_cp = predicted_cp
    print(f"Number of changepoint:{len(predicted_cp)}. First ten:{predicted_cp[0:10]}")
    return predicted_cp, len(predicted_cp)

def find_cp_tuning(signal, type_of_cpd,tune_var):
    model ='l2'
    if(tune_var=='min_size'):
        for i in range (0, 5):
            algo = rpt.Pelt(model=model, min_size=i, jump=5).fit(signal)
            predicted_cp = algo.predict(pen=3) #what is pen all about??
            print(f"Number of changepoint:{len(predicted_cp)}. First ten:{predicted_cp[0:10]}")

    return predicted_cp





def annotate_data_with_cp(df, cpd_type, cpd_colname, min_size, jump):
    pt_list = df['PtID'].unique()
    all_pt = []
    for i in (pt_list):
        df_pt = df[df['PtID'] == i].reset_index()
        #print(df_pt.head)
        signal = df_pt['EDA'].values
        predicted_cp = find_changepoint(signal, cpd_type, min_size, jump)
        # put it in the table
        for i in predicted_cp:
            #print("About to annotate")
            df_pt[cpd_colname][i] = 1 #df_pt['cpd_pelt'][i] = 1
        #add it back to the combined dataframe
        all_pt.append(df_pt)
    all_pt = pd.concat(all_pt)
    return all_pt



def get_total_cp(df, cpd_type, cpd_colname, min_size, jump):
    '''
    Total cp across all the patients 
    '''
    pt_list = df['PtID'].unique()
    total_cp = 0 #sum of the cp across the patients
    for i in (pt_list):
        df_pt = df[df['PtID'] == i].reset_index()
        df_pt = df_pt.dropna()
        #print(df_pt.head)
        signal = df_pt['glucose_level'].values
        predicted_cp, no_of_cp = find_changepoint(signal, cpd_type, min_size, jump)
        # sum it all up
        total_cp += no_of_cp
    return total_cp

##@jit(nopython=True)   
#delete later 
def plot_changepoints(data, start_date, end_date, ptID, col_name):
    figure(figsize=(12, 4), dpi=80)
    data['dates'] = pd.to_datetime(data['dates'])
    data_small = data[(data['dates'] >= start_date) & (data['dates'] <= end_date)]
    plt.plot(data_small['dates'], data_small['glucose_level'])

    #get the breakpoints index from the data 
    breakpoints = data_small[data_small[col_name] == 1].index.tolist()
    print(breakpoints)
    breakpoints = data_small[data_small[col_name] == 1]['dates'].tolist()
    #for i in range (len(breakpoints)-1):
    for i in (breakpoints):
        #changepoint = data_small.iloc[breakpoints[i]]['dates']
        print(i)
        #changepoint = data_small.iloc[i]['dates']
        plt.axvline(x = i, color="red", linestyle="--", label='changepoint')
    plt.title(f"Patient {ptID} from {start_date} to {end_date}.")
    plt.ylabel('CGM (mg/dl)', fontsize = 12)
    plt.xlabel('Timestamp',fontsize = 12)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    plt.legend(handles, labels, loc='best')
    plt.show()

def plot_cpd(data, start_date, end_date, ptID, col_name, show_ckp=False):
    '''
    Plot the cpd in the data. Set new_glucose to True if you want to plot the transformed glucose as well
    col_name: the col name of the cpd indicator...most times cpd_pelt
    '''
    figure(figsize=(12, 4), dpi=80)
    data['dates'] = pd.to_datetime(data['dates'])
    data_small = data[(data['dates'] >= start_date) & (data['dates'] <= end_date)]
    plt.plot(data_small['dates'], data_small['glucose_level'], label='sim_glucose')
    plt.plot(data_small['dates'], data_small['glucose_level_new'], label='sim_glucose_new')

    #get the breakpoints index from the data 
    if(show_ckp == True):
        breakpoints = data_small[data_small[col_name] == 1].index.tolist()
        print(breakpoints)
        #for i in range (len(breakpoints)-1):
        for i in (breakpoints):
            #changepoint = data_small.iloc[breakpoints[i]]['dates']
            changepoint = data_small['dates'][i]
            plt.axvline(x = changepoint, color="red", linestyle="-", label='changepoint')
    plt.title(f"Patient {ptID} from {start_date} to {end_date}.")
    plt.ylabel('CGM (mg/dl)', fontsize = 12)
    plt.xlabel('Timestamp',fontsize = 12)

    handles, labels = plt.gca().get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    plt.legend(handles, labels, loc='best')
    plt.show()

def group_into_daily(patient):
    '''
    Function to group a patient's readings into days

    patient: the patient's csv file
    '''
    #extract it into df per day
    patient.dates = pd.to_datetime(patient.dates)
    patient_daily = []
    for i in patient.groupby(patient.dates.dt.floor('d')):
        patient_daily.append(i[1])
    print(f"There are {len(patient_daily)} days data in this file")
    return patient_daily

def get_features(pt_sample, dataset, frequency=4, window_size=20):
    '''
    Feature extraction for classification task
    Args:
        pt_sample: DataFrame containing the EDA data
        dataset: Name of the dataset (e.g., 'WESAD')
        frequency: Sampling frequency in Hz (default 4Hz)
        window_size: Number of samples in the window (default 20 samples = 5 seconds at 4Hz)
    '''
    ### Extract Features
    print("In get features method")
    X_train_total = pd.DataFrame()
    y_train_total = []
    
    # Get feature configuration for statistical and temporal features
    cfg_file = tsfel.get_features_by_domain("statistical")
    cfg_file.update(tsfel.get_features_by_domain("temporal"))
    
    if dataset == "oaps":
        step = 50
    else:
        step = 1
        
    # daily_list = group_into_daily(pt_sample)
    for i in range(window_size, len(pt_sample), step):
        X_train = pd.Series([])
        patient_id = pt_sample['PtID'][i] if 'PtID' in pt_sample.columns else 'unknown'
        
        # Get exactly window_size samples before current point
        window = pt_sample['eda_signal'].values[i-window_size:i]
        # print(f"Window shape: {window.shape}")
        
        try:
            X_train = tsfel.time_series_features_extractor(cfg_file, window, n_jobs = 3, verbose = 0)
            # print(f"X_train_total shape: {X_train_total.shape}")
            # print(f"X_train shape: {X_train.shape}")
            # print(f"X_train df shape: {pd.DataFrame([X_train]).shape}")
            y_train = pt_sample['cpd'][i]
            # Concatenate along rows (axis=0)
            X_train_total = pd.concat([X_train_total, X_train], axis=0, ignore_index=True)
            y_train_total.append(y_train)
        except Exception as e:
            print(f"Issues with PtID: {patient_id} || index: {i}")
            print(f"Error: {str(e)}")
            
    return X_train_total, y_train_total, pt_sample

# def get_features_per_day(pt_sample, dataset):
#     '''
#     Feature extraction for classification task
#     '''
#     ### Extract Features
#     print("In get features method")
#     X_train_total = pd.DataFrame()
#     window_size = 120 #in minutes #60
#     index_list = []
#     cfg_file = tsfel.load_json("features_simple.json")
#     #cfg_file = tsfel.get_features_by_domain("statistical")
#     #if (dataset=='oaps'):
#         #pt_sample_step = pt_sample[::100].reset_index()
#         #my_list = list(range(0,len(pt_sample),100))
#     daily_list = group_into_daily(pt_sample)
#     ptID = pt_sample['PtID'][0]
#     if dataset == "oaps":
#         step = 100
#     else:
#         step = 1
#     for i in range(len(daily_list)):
#         per_day = daily_list[i]
#         #print(len(per_day))
#         for j in range(12, len(per_day), step):
#             t1 = pd.Timestamp(per_day['dates'].iloc[j]) 
#             X_train = pd.Series([])
#             t2 = t1 - pd.to_timedelta(window_size, unit='m')
#             #print(f"Processing Index {j} of Day {i} data in file: {ptID}. T1: {t1} T2: {t2}")
#             window = per_day[(per_day['dates'] >= t2) & (per_day['dates'] < t1)]['glucose_level'].values
#             #print(f"Window: {window}")
#             try:
#                 X_train = tsfel.time_series_features_extractor(cfg_file, window, n_jobs = 3, verbose = 0)
#                 X_train_total = X_train_total.append(X_train, True)
#             except:
#                 print("Issues with PtID: {} || index: {}".format(ptID, i))      
#     print(f"X train total: {len(X_train_total)}")
#     return X_train_total, pt_sample

    
def new_value2(mu_1, mu_2, sigma_1, sigma_2, x):
    y = mu_2 + (((x - mu_1)/sigma_1)*sigma_2)
    return y



def transform_data(mu,sigma,data):
    """
    transform the data to have a new mean and variance
    full_data: the full data to get the history window from
    """
    print("Processing ", len(data), " values")
    print(f"New Properties: {mu}, {sigma}")
    transformed_data = list()
    mu_1 = np.round(data['glucose_level'].describe()['mean'], 2)
    sigma_1 = np.round(data['glucose_level'].describe()['std'], 2)
    mu_2 = mu
    sigma_2 = sigma
    #data_indices = data.index #indices of the data
    #print(data_indices)
    
    print(f"Previous properties: mu={mu_1}, sigma={sigma_1}")
    for i in range(len(data)): #for each timestamp in the data
        row_index = data['glucose_level'].index[i]
        y = new_value2(mu_1, mu_2, sigma_1, sigma_2, data['glucose_level'][row_index])
        if (y < 0 or (math.isinf (float(y)))):
            print(row_index)
            y = data['glucose_level'][row_index] #mu_2 
            
        #print("Processing: Value=",i,"|| Row=",row_index,"|| Glucose_value=", data['glucose_level'][row_index], 
              #"|| New Glucose=", y)
        #data.at[row_index, 'new_glucose'] = y
        transformed_data.append(y)
    #display(data.head(10))
    return transformed_data

def modify_property(post_ckp, mu2, sigma2, type_of_change):
    
    mu1 = np.mean(post_ckp)
    sigma1 = np.std(post_ckp)
    
    print(f"Current mu1 and sigma1: {mu1}:{sigma1}")
    
    #to avoid division by zero, include epison
    epsilon = 0.0000001
    std_diff = sigma2/(sigma1+epsilon)
    print(f'std dif:{std_diff}')
    post_ckp_new = [mu2 + (i - mu1) * std_diff for i in post_ckp]
    
    '''
    if(type_of_change == 'mean+std'):
        post_ckp_new = [mu2 + (i - mu1) * std_diff for i in post_ckp]
    elif(type_of_change == 'mean'):
        post_ckp_new = [mu2 + (i - mu1) for i in post_ckp]       
    elif(type_of_change == 'std'):
        post_ckp_new = [mu1 + (i - mu1) * std_diff for i in post_ckp]  
    '''
    return post_ckp_new



def insert_values(df, index_list, values_list,col_name):
#print(values_list)
    for i in range(len(index_list)):
        df.at[index_list[i], col_name] = values_list[i] 
    return df

def check_sig_diff(x, y):
    sig_diff = False
    if (len(x) > 0 and len(y) > 0):
        #print("YESS diff dey")
        stat, p_value = mannwhitneyu(x, y)
        #print(p_value)
        alpha = 0.05
        if(p_value < alpha):
            sig_diff = True #reject H0: x and y are the same
    return sig_diff 

def draw_sample(mu, sigma):
    s = np.random.normal(mu, sigma, 1)
    #print(f"Sample:{s}")
    return s

def save_to_pkl(filepath, file_to_save):
    with open(filepath, 'wb') as fp:
        pickle.dump(file_to_save, fp)

def read_from_pkl(filepath):
    with open(filepath, 'rb') as fp:
        mdl = pickle.load(fp)
    return mdl

def large_missing_intervals(df, max_interval):
    #ensure changes spanning large missing intervals are not included
    df.reset_index(inplace=True)
    df['dates'] = pd.to_datetime(df['dates'])
    time_diff = df['dates'].diff()
    threshold = pd.Timedelta(minutes=max_interval)
    df.loc[time_diff > threshold, 'Up Trend'] = np.nan
    df.loc[time_diff > threshold, 'Down Trend'] = np.nan
    result = df[time_diff > threshold]
    return result, df

def get_dist(data, save_file=None, data_name='', xlabel='Value', plot_graph=False):
    # Define a list of candidate distributions to fit
    candidate_distributions = [stats.uniform, stats.norm, stats.expon]
    if (data_name == 'mean_change'):#only one that looks like bimodal
        candidate_distributions.append('bimodal_pdf')
    best_fit = None
    best_fit_params = {}
    best_fit_loglikelihood = -np.inf
    plt.figure(figsize=(8, 5))
    # Iterate through candidate distributions and fit to data using MLE
    for distribution in candidate_distributions:
        # Fit the distribution to the data using MLE
        if distribution == 'bimodal_pdf':
            try:
                y, x = np.histogram(data, bins=50)
                # Calculate the bin centers from the bin edges
                x = (x[1:] + x[:-1]) / 2
               
                # Provide initial guesses for the parameters of the bimodal distribution
                initial_params = (np.mean(data), np.std(data), 50, np.mean(data), np.std(data), 50)
                params, _ = curve_fit(bimodal, x, y, initial_params)
                distribution_name = 'bimodal'
                bimodal_params = params
            except RuntimeError:
                print("Bimodal fitting failed, skipping...")
                continue
        else:
            params = distribution.fit(data, loc=np.mean(data), scale=np.std(data))
            distribution_name = distribution.name 
        loglikelihood = log_likelihood(data, distribution_name, params, distribution)

        print(f"Distribution: {distribution_name} || Log-Likelihood: {loglikelihood} || Params: {params}")
        
        # Check if this distribution provides a better fit
        if loglikelihood > best_fit_loglikelihood:
            best_fit = distribution
            best_fit_params = params
            best_fit_loglikelihood = loglikelihood
            best_fit_name = distribution_name

    if plot_graph:
        # Plot the histogram of the data
        y, x = np.histogram(data, bins=100, density=True)
        x_mid = (x[1:] + x[:-1]) / 2
        plt.bar(x_mid, y, width=(x[1] - x[0]), color='skyblue', alpha=0.5)

        # Define a colormap for the distributions
        colormap = {
            stats.norm: 'orange',
            stats.uniform: 'green',
            stats.expon: 'red',
            'bimodal_pdf': 'blue'
        }

        # Plot the best-fitting distribution
        x = np.linspace(np.min(data), np.max(data), 100)
        if best_fit == 'bimodal_pdf':
            bimodal_curve = bimodal(x, *bimodal_params)
            plt.plot(x, bimodal_curve * y.max() / bimodal_curve.max(), 'r-', color=colormap[best_fit], label=f'Best fit: {distribution_name}')
        else:
            plt.plot(x, best_fit.pdf(x, *best_fit_params), 'r-', color=colormap[best_fit], label=f'Best fit: {best_fit.name}')

        # Plot the other candidate distributions
        for distribution in candidate_distributions:
            if distribution == best_fit:
                continue
            if (distribution == 'bimodal_pdf' and best_fit != 'bimodal_pdf'):
                try:
                    bimodal_curve = bimodal(x, *bimodal_params)
                    plt.plot(x, bimodal_curve * y.max() / bimodal_curve.max(), 'r-', color=colormap[distribution], label=f'{distribution_name}')
                except:
                    continue
            elif (distribution != best_fit):
                params = distribution.fit(data, loc=np.mean(data), scale=np.std(data))
                plt.plot(x, distribution.pdf(x, *params), color=colormap[distribution], label=f"{distribution.name}")
                
        # Set plot labels and legend
        plt.xlabel(xlabel, fontsize=22)
        plt.ylabel('Probability Density', fontsize=22)
        plt.xticks(fontsize=22)             # Adjust the font size of the X ticks
        plt.yticks(fontsize=22) 
        plt.legend(fontsize=18)
        plt.show()

    # Print the best-fitting distribution and its parameters
    print(f"Best fit: {best_fit_name}")
    print(f"Parameters: {best_fit_params}")
    return best_fit_name, best_fit_params

# Calculate the log-likelihood of the data given the bimodal distribution
def neg_log_likelihood_bimodal(params, data):
    mu1, sigma1, A1, mu2, sigma2, A2 = params
    likelihood = bimodal(data, mu1, sigma1, A1, mu2, sigma2, A2)
    neg_log_likelihood = -np.sum(np.log(likelihood))
    return neg_log_likelihood

# Define the bimodal function
def bimodal(x, mu1, sigma1, A1, mu2, sigma2, A2):
    return A1 * np.exp(-(x - mu1) ** 2 / (2 * sigma1 ** 2)) + A2 * np.exp(-(x - mu2) ** 2 / (2 * sigma2 ** 2))

# Calculate the log-likelihood of the data given a distribution
def log_likelihood(data, dist_name, params,distribution):
    if dist_name == 'bimodal':
        return -np.sum(np.log(bimodal(data, *params)))
    else:
        return np.sum(distribution.logpdf(data, *params))