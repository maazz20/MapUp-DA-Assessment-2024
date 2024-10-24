from typing import Dict, List

import pandas as pd

#1
def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    for i in range(0, len(lst), n):

        left = i
        right = min(i + n - 1, len(lst) - 1)  
        
        while left < right:
            lst[left], lst[right] = lst[right], lst[left]
            left += 1
            right -= 1
    return lst 

#2
def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    length_dict = {}
    
    for string in lst: 
        length = len(string)
        
        if length not in length_dict:
            length_dict[length] = [string]
        else:
            length_dict[length].append(string)
    
    dict = sorted(length_dict.items())  
    return dict

#3
def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    result = {}
    
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            for subkey, subvalue in flatten_dict(value, sep).items():
                result[f"{key}{sep}{subkey}"] = subvalue
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    for subkey, subvalue in flatten_dict(item, sep).items():
                        result[f"{key}{sep}{index}{sep}{subkey}"] = subvalue
                else:
                    result[f"{key}{sep}{index}"] = item
        else:
            result[key] = value
    
    return result

#4
def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    def backtrack(start: int):
        if start == len(nums):
            result.append(nums[:]) 
            return

        seen = set() 
        for i in range(start, len(nums)):
            if nums[i] in seen:  
                continue
            seen.add(nums[i]) 
            nums[start], nums[i] = nums[i], nums[start]  
            backtrack(start + 1)  
            nums[start], nums[i] = nums[i], nums[start]  

    result = []
    nums.sort()  
    backtrack(0)  
    return result
    pass

#5
def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    date_pattern = r'\b(\d{2}-\d{2}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}\.\d{2}\.\d{2})\b'
    
    import re
    dates = re.findall(date_pattern, text)
    
    return dates
    pass

#6
def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    import numpy as np
    import polyline
    def haversine(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        # Haversine formula
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a)) 
        r = 6371000  
        return c * r
    coordinates = polyline.decode(polyline_str)

    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])

    distances = [0]  
    for i in range(1, len(df)):
        dist = haversine(df.iloc[i-1]['latitude'], df.iloc[i-1]['longitude'],
                         df.iloc[i]['latitude'], df.iloc[i]['longitude'])
        distances.append(dist)

    df['distance'] = distances
    df = pd.DataFrame

#7
def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    
    def rotate_matrix(matrix: List[List[int]]) -> List[List[int]]:
   
        n = len(matrix)
        rotated = [[0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                rotated[j][n - 1 - i] = matrix[i][j]
        
        return rotated

    def transform_matrix(matrix: List[List[int]]) -> List[List[int]]:
    
        n = len(matrix)
        result = [[0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                row_sum = sum(matrix[i])
                col_sum = sum(matrix[k][j] for k in range(n))
                result[i][j] = row_sum + col_sum - matrix[i][j]  # Exclude the element itself
                
        return result

    rotated = rotate_matrix(matrix)
    transformed = transform_matrix(rotated)
    return transformed

#8
def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    df['start_datetime'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end_datetime'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])
    
    def check_group(group):
        unique_days = group['start_datetime'].dt.date.unique()
        if len(unique_days) < 7:
            return False
        
        for day in unique_days:
            day_start = pd.Timestamp(day) + pd.Timedelta(hours=0)  # 12:00 AM
            day_end = pd.Timestamp(day) + pd.Timedelta(hours=23, minutes=59, seconds=59)  # 11:59:59 PM
            
            if not (group['start_datetime'] <= day_end).any() or not (group['end_datetime'] >= day_start).any():
                return False
        
        return True

    result = df.groupby(['id', 'id_2']).apply(check_group)
    result = pd.series()

    return pd.Series()
