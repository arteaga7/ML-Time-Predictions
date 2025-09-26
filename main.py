
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from typing import Tuple    # For python3.8
from prophet import Prophet
from sklearn.metrics import mean_absolute_error


path_parquet = './data/raw/Lecturas_Eneero_2025'
path_tables = './data/raw/'
path_processed = './data/processed'
FUTURE_DAYS = 6
RESAMPLE_FREQ = 'H'
output_dir = './data/processed/prophet_forecasts'


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


def merge_DataFrames(path_processed: 'str', df, df_cities, df_devices, df_locations,
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


def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    '''Function to preprocess the data'''
    df['TimeSpan'] = pd.to_datetime(df['TimeSpan'])
    df['LocalTimeSpan'] = pd.to_datetime(df['TimeSpan'])
    df = df.drop_duplicates(subset=['ReadId'])
    df = df[['TimeSpan', 'SensorId', 'Value']]
    return df


def prophet_model(output_dir: 'str', df: pd.DataFrame) -> dict:
    '''Function to create and train the prophet model for forecasting'''
    # Convert to compatible format for Prophet (ds, y)
    df = df.rename(columns={'TimeSpan': 'ds', 'Value': 'y'})
    df = df.sort_values(by='ds', ascending=True)

    # Resample by hour and obtain the mean of data
    df_resampled = df.set_index('ds').groupby(
        'SensorId')['y'].resample(RESAMPLE_FREQ).mean().reset_index()

    # Create directory to save the results
    os.makedirs(output_dir, exist_ok=True)

    # Obtain all sensors as a list
    sensor_ids = df_resampled['SensorId'].unique()

    # Dictionary to save the resulting DataFrames
    results = {}

    for sensor_id in sensor_ids:
        df_sensor = df_resampled[df_resampled['SensorId'] == sensor_id].copy()
        df_sensor = df_sensor.dropna(subset=['y'])

        if len(df_sensor) < 2:
            print(f" ❌ Insufficient data for sensor {sensor_id}. Skipping.")
            continue

        # Create model. Weekly and daily seasonality is automatically activated
        # if sufficient data is detected
        model = Prophet(
            yearly_seasonality='auto',
            weekly_seasonality='auto',
            daily_seasonality='auto',
            # changepoint_prior_scale=0.03
        )

        # Fit the model
        try:
            model.fit(df_sensor)
        except Exception as e:
            print(
                f" ❌ Error training the model for {sensor_id}: {e}. Skipping.")
            continue

        # Make forecast
        future = model.make_future_dataframe(
            periods=FUTURE_DAYS * 24, freq=RESAMPLE_FREQ)
        forecast = model.predict(future)

        # Add column 'y' to DataFrame forecast
        actual_values = df_sensor[['ds', 'y']]
        forecast = pd.merge(forecast, actual_values, on='ds', how='left')

        # Store the forecast
        results[sensor_id] = forecast

        # Visualization
        fig = model.plot(forecast)
        plt.title(f"Time Series forecast for sensor: {sensor_id}")
        plt.xlabel("Date")
        plt.ylabel("Value")

        # Save plot
        fig.savefig(os.path.join(output_dir, f'forecast_{sensor_id}.png'))
        plt.close(fig)

    print("\n✅ Process completed.")
    print(
        f"The forecast charts have been saved to the directory '{output_dir}'.")
    print("The results of the predictions are in the 'results' dictionary.")
    return results


def mae(results: dict) -> dict:
    '''Function to calculate MAE for every model'''
    mae_scores = {}

    for sensor_id, forecast_df in results.items():
        historical_comparison = forecast_df.dropna(subset=['y'])

        if len(historical_comparison) == 0:
            print(
                f"There are not enough points to calculate the MAE for {sensor_id}.")
            mae_scores[sensor_id] = np.nan
            continue

        mae = mean_absolute_error(
            historical_comparison['y'], historical_comparison['yhat'])
        mae_scores[sensor_id] = mae

    return mae_scores


def main():
    df = get_data(path_parquet)
    df_cities, df_devices, df_locations, df_sensors, df_sublocations, df_unidades = read_tables(
        path_tables)
    merge_DataFrames(path_processed, df, df_cities, df_devices, df_locations,
                     df_sensors, df_sublocations, df_unidades)
    df = preprocessing(df)
    results = prophet_model(output_dir, df)
    mae_scores = mae(results)


# Run main function
if __name__ == "__main__":
    main()
