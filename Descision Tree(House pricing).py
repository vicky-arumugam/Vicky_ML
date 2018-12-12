# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 17:07:03 2018

@author: ssn
"""

#decision tree
    import pandas as pd
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor
    iowa_file_path = '../input/home-data-for-ml-course/train.csv'
    home_data = pd.read_csv(iowa_file_path)
    y = home_data.SalePrice
    features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
    X = home_data[features]
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
    iowa_model = DecisionTreeRegressor(random_state=1)
    iowa_model.fit(train_X, train_y)
    val_predictions = iowa_model.predict(val_X)
    val_mae = mean_absolute_error(val_predictions, val_y)
    print(val_mae)
    for max_leaf_nodes in [5, 50, 500, 1000]:
        my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
        print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" (max_leaf_nodes, my_mae))
    final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)
    final_model.fit(X,y)