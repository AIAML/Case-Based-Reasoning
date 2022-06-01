# Import
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

def main():
    # First Step We Get .csv file for iris input dataset using pandas
    iris, cases = pd.read_csv('iris.csv'), pd.read_csv('cases.csv')

    print('\n> Initial iris')
    print(f'\n{iris}')
    neighbors = 7
    base = iris.iloc[:, range(iris.shape[1] - 1)]
    iris_classes = iris.iloc[:, iris.shape[1] - 1]

    # [2] Show our normalized data
    base = pd.get_dummies(base)
    problems = pd.get_dummies(cases)

    print('\n> Calculating\n')

    for i in range(problems.shape[0]):
        if i != problems.shape[0]:
            case_row = problems.loc[i:i+1, :]
        else:
            case_row = problems.loc[i-1:i, :]

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(base.to_numpy(), iris_classes)
        y_pred = knn.predict(case_row.to_numpy())
        case = np.append(cases.iloc[i, :], y_pred[0])
        print(f'> For case/problem {i}: {cases.iloc[i, :].to_numpy()}, solution is {case[-1]}')
        case = pd.Series(case, index=iris.columns)  # Case with Solution
        iris = iris.append(case, ignore_index=True)  # Append to iris
        base = pd.get_dummies(base)  # Get new one-hot encoded base

    print('\n> Output iris')
    print(f'\n{iris}')

    # Save 'iris' output as file
    iris.to_csv('outknn.csv', index=False)
# Call
if __name__ == '__main__':
    main()
