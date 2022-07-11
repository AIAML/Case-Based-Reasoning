import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsClassifier

def main():
    # First Step We Get .csv file for iris input dataset using pandas
    iris, cases = pd.read_csv('input/iris.csv'), pd.read_csv('input/cases.csv')

    print('\n> Initial iris')
    print(f'\n{iris}')

    base = iris.iloc[:, range(iris.shape[1] - 1)]
    iris_classes = iris.iloc[:, iris.shape[1] - 1]

    # [2] Show our normalized data
    base = pd.get_dummies(base)
    problems = pd.get_dummies(cases)

    # [3] Calculate using covariance matrix
    print('\n> Calculating\n')

    # Check all cases
    for i in range(problems.shape[0]):

        # [3.1] Get inverse covariance matrix for the base cases
        covariance_matrix = base.cov()  # Covariance
        inverse_covariance_matrix = np.linalg.pinv(covariance_matrix)  # Inverse
        if i != problems.shape[0]:
            case_row = problems.loc[i:i+1, :]
        else:
            case_row = problems.loc[i-1:i, :]
        # Empty distances array to store mahalanobis distances obtained comparing each iris cases
        distances = np.zeros(base.shape[0])

        # [3.3] For each base cases rows
        for j in range(base.shape[0]):
            # Get base case row
            base_row = base.loc[j, :]

            # [3.4] Calculate mahalanobis distance between case row and base cases, and store it
            distances[j] = distance.mahalanobis(case_row, base_row, inverse_covariance_matrix)

        # [3.5] Returns the index (row) of the minimum value in distances calculated
        min_distance_row = np.argmin(distances)

        # [4] Get solution based on index of found minimum distance, and append it to main iris
        # From cases, append iris 'similar' solution
        case = np.append(cases.iloc[i, :], iris.iloc[min_distance_row, -1])

        # Print
        print(f'> For case/problem {i}: {cases.iloc[i, :].to_numpy()}, solution is {case[-1]}')

        # [5] Store
        # Get as operable pandas Series
        case = pd.Series(case, index=iris.columns)  # Case with Solution
        iris = iris.append(case, ignore_index=True)  # Append to iris

        # [6] Reuse
        base = pd.get_dummies(base)  # Get new one-hot encoded base

    # [7] Output
    print('\n> Output iris')
    print(f'\n{iris}')

    # Save 'iris' output as file
    iris.to_csv('output/outmain.csv', index=False)


# Call
if __name__ == '__main__':
    main()
