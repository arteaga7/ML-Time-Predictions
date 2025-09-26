
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
from typing import Tuple    # For python3.8

path_parquet = './data/raw/Lecturas_Eneero_2025'
path_tables = './data/raw/'
path_processed = './data/processed'


def get_data(path: 'str') -> pd.DataFrame:
    """Function to obtain data as DataFrame."""
    content = os.listdir(path)
    files = []
    no_empty = []
    # Create DataFrame
    df = pd.DataFrame(
        {
            'ReadId': [],
            'TimeSpan': [],
            'SensorId': [],
            'Value': [],
            'LocalTimeSpan': [],
        })

    print("Reading parquet files, this may take a few minuts...")
    for folder in content:
        if folder.endswith(''):
            path_files = os.path.join(path, folder)
            for file in os.listdir(path_files):
                if file.endswith('parquet'):
                    files.append(file)
                    try:
                        df_1 = pd.read_parquet(os.path.join(path_files, file))
                        df = pd.concat([df, df_1])
                        no_empty.append(file)
                    except OSError:
                        pass

    df = df.reset_index(drop=True)
    print(
        f"Total files (*.parquet) found: {len(files)}, Total non-empty files: {len(no_empty)}")
    return df


def read_tables(path: 'str') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Function to read tables.
    Tuple[] for python3.8 and tuple[] for python3.9"""
    df_cities = pd.read_csv(os.path.join(
        path, 'dimCities.csv'), sep=',', header=0)
    df_cities = df_cities.rename(columns={'Name': 'CityName'})

    df_devices = pd.read_csv(os.path.join(
        path, 'dimDevices.csv'), sep=',', header=0)
    df_devices = df_devices.rename(
        columns={'Name': 'DeviceName', 'Active': 'DeviceActive'})

    df_locations = pd.read_csv(os.path.join(
        path, 'dimLocations.csv'), sep=',', header=0)
    df_locations = df_locations.rename(
        columns={'Name': 'LocationName', 'Active': 'LocationActive'})

    df_sensors = pd.read_csv(os.path.join(
        path, 'dimSensors.csv'), sep=',', header=0)
    df_sensors = df_sensors.rename(
        columns={'Name': 'SensorName', 'Active': 'SensorActive'})

    df_sublocations = pd.read_csv(os.path.join(
        path, 'dimSublocations.csv'), sep=',', header=0)
    df_sublocations = df_sublocations.rename(
        columns={'Name': 'SubLocationName', 'Active': 'SubLocationActive'})

    df_unidades = pd.read_csv(os.path.join(
        path, 'dimUnidades.csv'), sep=',', header=0)
    df_unidades = df_unidades.rename(
        columns={'Name': 'MeasureName', 'Active': 'UnitActive'})

    return df_cities, df_devices, df_locations, df_sensors, df_sublocations, df_unidades


def merge_DataFrames(path_processed, df, df_cities, df_devices, df_locations,
                     df_sensors, df_sublocations, df_unidades) -> None:
    """Function to merge DataFrames."""
    df_sensors = df_sensors[['SensorId', 'SensorTyId',
                             'DeviceId', 'SensorName', 'SensorActive']]
    df_unidades = df_unidades[['SensorTyId',
                               'MeasureName', 'UnitName', 'UnitAbbreviation']]
    df_devices = df_devices[['DeviceId', 'SubLocationId', 'DeviceName']]

    df2 = df_sensors.merge(df_unidades, on='SensorTyId', how='left')
    sensor_device_unit = df2.merge(df_devices, on='DeviceId', how='left')

    df3 = df_locations.merge(df_cities, on='CityId', how='left')
    location_subloc_city = df3.merge(
        df_sublocations, on='LocationId', how='left')

    print("Saving tables...")
    # Save DataFrames
    file_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    df_name = f'factLecturas_{file_date}.csv'

    df.to_csv(os.path.join(path_processed, df_name), index=False)

    sensor_device_unit.to_csv(os.path.join(
        path_processed, 'sensor_device_unit.csv'), index=False)

    location_subloc_city.to_csv(os.path.join(
        path_processed, 'location_subloc_city.csv'), index=False)


def main():
    df = get_data(path_parquet)
    df_cities, df_devices, df_locations, df_sensors, df_sublocations, df_unidades = read_tables(
        path_tables)
    merge_DataFrames(path_processed, df, df_cities, df_devices, df_locations,
                     df_sensors, df_sublocations, df_unidades)


# Run main function
if __name__ == "__main__":
    main()
