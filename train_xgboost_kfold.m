function [r2_score, rmse_score] = train_xgboost_kfold(X_all, Y_all, Time_array)
% TRAIN_XGBOOST_KFOLD Performs a Leave-One-24h-Out cross-validation 
% using XGBoost via the MATLAB-Python bridge to prevent temporal data leakage.

    % 1. Extract 24h Cycles (Midi-Midi)
    ID_24h = floor(Time_array - 0.5);
    cycles_uniques = unique(ID_24h);
    num_folds = length(cycles_uniques);
    
    % 2. Initialize Python XGBoost Bridge
    np = py.importlib.import_module('numpy');
    xgb = py.importlib.import_module('xgboost');
    
    n_cols = size(X_all, 2);
    Y_pred_total = NaN(size(Y_all));
    
    % 3. Spatio-Temporal Cross-Validation Loop
    for k = 1:num_folds
        idx_test = (ID_24h == cycles_uniques(k)); 
        idx_train = ~idx_test;
        
        if sum(idx_train) < 10 || sum(idx_test) < 10
            continue; 
        end
        
        % Format data for Python NumPy
        X_tr_mat = X_all(idx_train, :)'; 
        X_py_tr = np.array(X_tr_mat(:)').reshape(int32(sum(idx_train)), int32(n_cols));
        Y_py_tr = np.array(Y_all(idx_train)');
        
        X_te_mat = X_all(idx_test, :)';
        X_py_te = np.array(X_te_mat(:)').reshape(int32(sum(idx_test)), int32(n_cols));
        
        % Initialize and Train XGBoost
        mdl_xgb = xgb.XGBRegressor(pyargs('n_estimators', int32(200), ...
            'learning_rate', 0.05, 'max_depth', int32(4), ...
            'subsample', 0.8, 'n_jobs', int32(-1)));
            
        mdl_xgb.fit(X_py_tr, Y_py_tr);
        
        % Predict and retrieve data to MATLAB
        preds_py = mdl_xgb.predict(X_py_te);
        Y_pred_total(idx_test) = double(preds_py)';
    end
    
    % 4. Compute Global Metrics
    valid_mask = ~isnan(Y_pred_total) & ~isnan(Y_all);
    res = Y_all(valid_mask) - Y_pred_total(valid_mask);
    
    SSTot = sum((Y_all(valid_mask) - mean(Y_all(valid_mask))).^2);
    r2_score = 1 - (sum(res.^2) / SSTot);
    rmse_score = sqrt(mean(res.^2));
end
