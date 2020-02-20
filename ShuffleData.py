import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def shuffleData(source_directory, destination_directory, test_size=0.2, remove_useless_data=True, standardize=True,
                split_to_windows=False, window_size=50, overlap=5):

    files = [x for x in os.listdir(source_directory) if x.endswith('.xlsx')]
    data_frame = pd.read_excel(os.path.join(source_directory, files[0]))
    for file in files[1:]:
        tempDF = pd.read_excel(os.path.join(source_directory, file))

        # Remove rows if they don't have label, set correct data type, append to the main data frame
        tempDF = tempDF[np.isfinite(tempDF['label'])]
        tempDF['label'] = tempDF['label'].astype('int32')
        data_frame.append(tempDF, ignore_index=True)

    columns = list(data_frame.columns.values)

    # Convert all numeric data to float
    data_frame[' acc_x'] = data_frame[' acc_x'].astype('float64')
    data_frame[' acc_y'] = data_frame[' acc_y'].astype('float64')
    data_frame[' acc_z'] = data_frame[' acc_z'].astype('float64')
    data_frame[' gyro_x'] = data_frame[' gyro_x'].astype('float64')
    data_frame[' gyro_y'] = data_frame[' gyro_y'].astype('float64')
    data_frame[' gyro_z'] = data_frame[' gyro_z'].astype('float64')
    data_frame[' lacc_x'] = data_frame[' lacc_x'].astype('float64')
    data_frame[' lacc_y'] = data_frame[' lacc_y'].astype('float64')
    data_frame[' lacc_z'] = data_frame[' lacc_z'].astype('float64')
    data_frame[' eul_x'] = data_frame[' eul_x'].astype('float64')
    data_frame[' eul_y'] = data_frame[' eul_y'].astype('float64')
    data_frame[' eul_z'] = data_frame[' eul_z'].astype('float64')


    if remove_useless_data:
        data_frame = data_frame.drop(['timestamp', ' sensor'], axis=1).copy()

    # Separate X and Y
    ys = data_frame['label']
    xs = data_frame.drop(['label'], axis=1).copy()

    if standardize:
        scaler = StandardScaler()
        scaled_xs = scaler.fit_transform(xs)
        data_frame = pd.DataFrame(data=scaled_xs, columns=columns[2:-1]).copy()

    data_frame[columns[-1]] = ys

    if not os.path.exists(os.path.join(os.getcwd(), destination_directory)):
        os.makedirs(destination_directory)


    # Do not split to windows
    if not split_to_windows:
        # Shuffle rows
        data_frame = shuffle(data_frame)

        # Split train and test sets
        train, test = train_test_split(data_frame, test_size=test_size)

        train.to_csv(os.path.join(destination_directory, "train.csv"), index=False)
        test.to_csv(os.path.join(destination_directory, "test.csv"), index=False)

    # Split to windows (for CNN)
    else:
        # TODO: Which shape of data do you need for CNN?
        pass

    return 0

def shuffleData2(source_directory, destination_directory, test_size=0.2, remove_useless_data=True, standardize=True,
                 split_to_windows=False, window_size=50, overlap=5):
    files = [x for x in os.listdir(source_directory) if x.endswith('.csv')]
    tmp = []
    for file in files:
        df = pd.read_csv(os.path.join(source_directory, file), index_col=None, header=0, low_memory=False)
        tmp.append(df)
    data_frame = pd.concat(tmp, axis=0, ignore_index=True)

    if not os.path.exists(os.path.join(os.getcwd(), destination_directory)):
        os.makedirs(destination_directory)

    if not split_to_windows:
        data_frame = shuffle(data_frame)
        train, test = train_test_split(data_frame, test_size=test_size)

        train.to_csv(os.path.join(destination_directory, "train.csv"), index=False)
        test.to_csv(os.path.join(destination_directory, "test.csv"), index=False)

    # Split to windows (for CNN)
    else:
        # TODO: Which shape of data do you need for CNN?
        pass

    return 0


if __name__ == '__main__':
    shuffleData("data/", "shuffled_data/")
    shuffleData2("data2/", "shuffled_data2/")
    shuffleData2("data3/", "shuffled_data3/")
