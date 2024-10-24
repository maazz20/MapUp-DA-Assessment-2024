import pandas as pd
import numpy as np

#1
def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    unique_ids = pd.Index(np.union1d(df['id_start'], df['id_end']))

    dist_matrix = pd.DataFrame(np.inf, index=unique_ids, columns=unique_ids)

    np.fill_diagonal(dist_matrix.values, 0)

    for _, row in df.iterrows():
        id_start, id_end, distance = row['id_start'], row['id_end'], row['distance']
        dist_matrix.at[id_start, id_end] = distance
        dist_matrix.at[id_end, id_start] = distance  

    df = dist_matrix

    return df

#2
def unroll_distance_matrix(matrix_df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    unrolled_data = []

    for id_start in matrix_df.index:
        for id_end in matrix_df.columns:
            if id_start != id_end:  
                distance = matrix_df.at[id_start, id_end]
                unrolled_data.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})

    result_df = pd.DataFrame(unrolled_data)
    return result_df

#3
def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    reference_data = df[df['id_start'] == reference_id]
    
    reference_avg_distance = reference_data['distance'].mean()
    
    lower_bound = reference_avg_distance * 0.9 
    upper_bound = reference_avg_distance * 1.1  
    
    valid_ids = []
    for id_start in df['id_start'].unique():
        if id_start == reference_id:
            continue

        avg_distance = df[df['id_start'] == id_start]['distance'].mean()

        if lower_bound <= avg_distance <= upper_bound:
            valid_ids.append(id_start)

    df= sorted(valid_ids)
    return df

#4
def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    df['moto'] = df['distance'] * 0.8
    df['car'] = df['distance'] * 1.2
    df['rv'] = df['distance'] * 1.5
    df['bus'] = df['distance'] * 2.2
    df['truck'] = df['distance'] * 3.6
    return df

#5
def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    from datetime import datetime
 
    discount_factors = {
        ("Monday", "10:00:00", "18:00:00"): 1.2,
        ("Tuesday", "10:00:00", "18:00:00"): 1.2,
        ("Wednesday", "10:00:00", "18:00:00"): 1.2,
        ("Thursday", "10:00:00", "18:00:00"): 1.2,
        ("Friday", "10:00:00", "18:00:00"): 1.2,
        ("Saturday", "00:00:00", "23:59:59"): 0.7,
        ("Sunday", "00:00:00", "23:59:59"): 0.7
    }

    def calculate_toll(row):
        day, start_time = row["start_day"], row["start_time"]
        start_time = datetime.strptime(start_time, "%H:%M:%S").time()

        for (d, time_start, time_end), factor in discount_factors.items():
            if day == d and datetime.strptime(time_start, "%H:%M:%S").time() <= start_time <= datetime.strptime(time_end, "%H:%M:%S").time():
                return factor

        return 0.8

    df["discount_factor"] = df.apply(calculate_toll, axis=1)
    vehicle_columns = ["moto", "car", "rv", "bus", "truck"]
    
    for col in vehicle_columns:
        df[col] = df[col] * df["discount_factor"]
    
    return df

