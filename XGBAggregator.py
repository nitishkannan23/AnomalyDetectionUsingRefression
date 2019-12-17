#### Creating Aggregate Files and Performing Regression  ####
#### The Code expects dependent variables to belong to the alphanumeric column name category ####

import pandas as pd
import numpy as np
import os
import configparser
import re
import glob
from sklearn.externals import joblib
from dateutil.parser import parse
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.utils import check_array
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV 

### Calling Configuration Files ###

config = configparser.ConfigParser()
config.read('regression_config.ini')

data_path           =  config['DIRECTORY PATH']['data_path'] 
directory_path      =  config['DIRECTORY PATH']['directory_path']
write_path          =  config['DIRECTORY PATH']['write_path_aggregate_regression']
utils_path          =  config['DIRECTORY PATH']['read_util_path']
machine_name        =  config['MACHINE NAME TYPE']['name']
train_machines      =  config['TRAIN MACHINE']['mac']
test_machines       =  config['TEST MACHINE']['mac']
dependent_variable  =  config['DEPENDENT VARIABLE']['dep']
ignore_columns      =  config['IGNORE']['COLS']
timestamp_variable  =  config['TIMESTAMP']['var']
device_variable     =  config['DEVICE']['var']
filter_var_1        =  config['filter']['var1']
limit_1             =  float(config['filter']['limit1'])
filter_var_2        =  config['filter']['var2']
limit_2             =  float(config['filter']['limit2'])
Col                 =  config['TIMESTAMP']['timestamp']
remove              =  config['TIMESTAMP']['remove']
upper_cor           =  float(config['CORRELATION']['upper'])
lower_cor           =  float(config['CORRELATION']['lower'])
window              =  int(config['MOVING_AVG']['WINDOW'])
threshold           =  float(config['CORRELATION']['threshold'])
upto                =  int(config['FEATURE NUMBERS']['upto'])

ignore_columns = ignore_columns.split(",")
train_machines = train_machines.split(",")
test_machines  = test_machines.split(",")
dependent_variable  = dependent_variable.split(",")
X1 = timestamp_variable
X2 = device_variable

os.chdir(directory_path)
write_path  = write_path+machine_name+"XGB"
if not os.path.exists(write_path):
    os.makedirs(write_path)

file_list = glob.glob(data_path+machine_name+"*.csv")

train_path =  []
test_path  =  []
for  i in train_machines:
    for j in file_list:
        if i in j:
            train_path.append(j)
for  i in test_machines:
    for j in file_list:
        if i in j:
            test_path.append(j)

train = pd.DataFrame()
test  = pd.DataFrame()
for i in range(0,len(train_path)):
    df = pd.read_csv(train_path[i])
    try:
        df.drop('Unnamed: 0', axis = 1,inplace = True)
    except ValueError:
        pass
    train = pd.concat([train,df], axis = 0, ignore_index = True)

for i in range(len(test_path)):
    df = pd.read_csv(test_path[i])
    try:
        df.drop('Unnamed: 0', axis = 1,inplace = True)
    except ValueError:
        pass
    test = pd.concat([test,df], axis = 0, ignore_index = True)


num_col_df = pd.read_csv(utils_path+"/"+"AlphaNumericColumn.csv")
try:
    num_col_df.drop('Unnamed: 0', axis = 1,inplace = True)
except ValueError:
    pass  

def remove_numbers(s):
    
    return ''.join([i for i in s if not i.isdigit()])
def Get_Week_Year_Number(df,Col):
        Week_Number = []
        Year=[]
        week_year=[]
        for i in list(df[Col]):
            temp = re.split(" ",i)[0]
            Year.append(parse(temp).year)
            Week_Number.append(parse(temp).isocalendar()[1])
            week_year.append(str(parse(temp).isocalendar()[1])+"-"+str(parse(temp).year))
        return Week_Number,Year,week_year

def mean_absolute_percentage_error(y_true, y_pred):
        y_true = check_array(y_true)
        y_pred = check_array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100 

def XGB_parameter(X_train,y_train):
        
    param_test1 = {'max_depth':list(range(3,10,1)),'min_child_weight':list(range(1,6,1))}
    gsearch1 = GridSearchCV(estimator = XGBRegressor(),param_grid = param_test1,cv=5)
    gsearch1.fit(X_train,y_train)
    gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
    
    depth=gsearch1.best_params_['max_depth']
    child_weight=gsearch1.best_params_['min_child_weight']
    
    param_test2 = {'gamma':[i/10.0 for i in range(0,5)]}
    gsearch2 = GridSearchCV(estimator = XGBRegressor(max_depth=depth,min_child_weight=child_weight),param_grid = param_test2,cv=5)
    gsearch2.fit(X_train,y_train)
    gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_

    g= gsearch2.best_params_['gamma']
    
    param_test3 = {'subsample':[i/10.0 for i in range(6,10)],'colsample_bytree':[i/10.0 for i in range(6,10)]}
    gsearch3 = GridSearchCV(estimator = XGBRegressor(max_depth=depth, min_child_weight=child_weight, gamma=g),param_grid = param_test3,cv=5)
    gsearch3.fit(X_train,y_train)
    gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
    
    subsample=gsearch3.best_params_['subsample']
    colsample=gsearch3.best_params_['colsample_bytree']
    
    param_test4 = {'reg_alpha':[1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 100]}
    gsearch4 = GridSearchCV(estimator = XGBRegressor(max_depth=depth, min_child_weight=child_weight, gamma=g,colsample_bytree=colsample,subsample=subsample),param_grid = param_test4,cv=5)
    gsearch4.fit(X_train,y_train)
    gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_

    al=gsearch4.best_params_['reg_alpha']      
                
    param_test5 = {'reg_alpha':[0.5*al,al,5*al]}
    gsearch5 = GridSearchCV(estimator = XGBRegressor(max_depth=depth, min_child_weight=child_weight, gamma=g,colsample_bytree=colsample,subsample=subsample),param_grid = param_test5,cv=5)
    gsearch5.fit(X_train,y_train)
    gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_
    
    alpha=gsearch5.best_params_['reg_alpha']
    
    param_test6 = {'learning_rate' : [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]}
    gsearch6 = GridSearchCV(estimator = XGBRegressor(max_depth=depth, min_child_weight=child_weight, gamma=g,colsample_bytree=colsample,subsample=subsample,reg_alpha=alpha),param_grid = param_test6,cv=5)
    gsearch6.fit(X_train,y_train)
    gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_       
    
    learn_rate=gsearch6.best_params_['learning_rate']
    
    return depth,child_weight,g,subsample,colsample,alpha,learn_rate

def XGB_train(X_train,y_train,columns_train,max_depth,min_child_weight,gamma,subsample,colsample_bytree,reg_alpha,learn_rate,device_name,dep):
    
    regr_xgb=xgb.XGBRegressor(min_child_weight=min_child_weight,max_depth=max_depth,gamma=gamma,colsample_bytree=colsample_bytree,subsample=subsample,reg_alpha=reg_alpha,learning_rate=learn_rate)
    regr_xgb.fit(X_train,y_train)
    print(regr_xgb.score(X_train,y_train),"train score")
    weight=regr_xgb.booster().get_score()
    f_names=list(weight.keys())
    XGBfeatures_dataframe=pd.DataFrame({'feature_name':f_names,'value':list(weight.values())}).sort_values('value',ascending=False).reset_index()
    XGBfeatures_dataframe.to_excel(write_path +"/"+device_name+'_'+dep+"_XGB_Train_Feature.xlsx")
    return regr_xgb, XGBfeatures_dataframe
    

def get_redundant_pairs(df):
        pairs_to_drop = set()
        cols = df.columns
        for i in range(0, df.shape[1]):
            for j in range(0, i+1):
                pairs_to_drop.add((cols[i], cols[j]))
        return pairs_to_drop

def remove_correlated_independent_variables(df,list_of_features,threshold):
            # correlation matrix for the important features
            au_corr = pd.DataFrame(df[i] for i in list_of_features).T.corr().abs().unstack()
         
            # drop redundant pairs
            labels_to_drop = get_redundant_pairs(pd.DataFrame(df[i] for i in list_of_features).T)
            au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
            au_corr =  au_corr.reset_index()
            keep = []
            others = []
            for i in list_of_features:
                if(i not in others):
                    keep.append(i)
                    others.extend(list(au_corr['level_1'][(au_corr['level_0'] == i) & (au_corr[0]>= threshold)]))
            return keep,others
            


dep_feature_nonum = []

for i  in dependent_variable:
    dep_feature_nonum.append(remove_numbers(i))

col_list_train = list(train.columns)
col_list_test  = list(test.columns)
col_list = list(set(col_list_train + col_list_test))
 
          
    
for j in range(len(dependent_variable)):
    
    
    train_df =train[col_list]
    test_df = test[col_list]    
    num_col_list = list(set(num_col_df['non_num']))
    drop_num_col = num_col_df['var'][num_col_df['non_num']==dep_feature_nonum[j]].tolist()
    drop_num_col.remove(dependent_variable[j])
    
    drop_col = []
    drop_col = drop_col + drop_num_col
    #drop_col = drop_col + ignore_columns
    for i in range(0,len(col_list)):
        if type(train_df[col_list[i]][0]) == str:
            drop_col.append(col_list[i])
    
    col1 = list(set(train_df.columns) - set(train_df._get_numeric_data().columns))
    drop_col = drop_col + col1
    drop_col = list(set(drop_col))
    drop_col = [i for i in drop_col if X1 not in i ]
    drop_col = [i for i in drop_col if X2 not in i]

    col_list_1 = [i for i in col_list if i not in drop_col]
    num_col_list.remove(dep_feature_nonum[j])
    
    train_df = train_df[col_list_1]
    test_df  =  test_df[col_list_1]
    drop_list = []
    for i in num_col_list:
        dc = num_col_df['var'][num_col_df['non_num']==i].tolist()
        if len(dc)>1:
            drop_list = drop_list + dc
            train_df[i+"_avg"] = train_df[dc].mean(axis = 1)
            test_df[i+"_avg"] = test_df[dc].mean(axis = 1)
    col_list_2 = list(train_df.columns)
    col_list_2 = [i for i in col_list_2 if i not in drop_list]
    train_df = train_df[col_list_2]
    test_df  =  test_df[col_list_2] 
    
    ### Regression ###
    ### implement filters ###
    if (len(filter_var_1)>0 & len(filter_var_2)>0):
        train_df=train_df[(train_df[filter_var_1]>=limit_1) & (train_df[filter_var_2]>=limit_2)]
        test_df=test_df[(test_df[filter_var_1]>=limit_1) & (test_df[filter_var_2]>=limit_2)]
    elif(len(filter_var_1)>0):
        train_df=train_df[train_df[filter_var_1]>=limit_1]
        test_df=test_df[test_df[filter_var_1]>=limit_1]
    else:
         train_df=train_df[train_df[filter_var_2]>=limit_2]
         test_df=test_df[test_df[filter_var_2]>=limit_2]
   
    try:
        train_df.drop(remove, axis = 1,inplace = True)
        test_df.drop(remove, axis =1, inplace = True)
    except ValueError:
        pass
    
    correlation_matrix = train_df.corr() 
    cor = correlation_matrix[dependent_variable[j]].abs()    
    high_corr = pd.DataFrame(cor[(cor >= lower_cor) & (cor< upper_cor)])    
    l4 = list(high_corr.T.columns)    
    null  = train_df.columns[train_df.isnull().any()].tolist()
    train_df.drop(l4+null, axis=1, inplace=True)
    test_df = test_df[train_df.columns]

     
    inp_df =pd.DataFrame()
    predict_df = pd.DataFrame()
    
    def moving_average(df,window):
        if X2 in df.columns: df.drop(X2,axis=1,inplace=True)
        df=df.rolling(window=window).mean().dropna(axis=0)
        return df
    
    
    
    if len(train_machines)>1:
        for i in range(len(train_machines)):
            df = train_df[train_df[X2]==train_machines[i]]
            df = moving_average(df,window)
            inp_df = pd.concat([inp_df,df], axis = 0, ignore_index = True)
    else:
         df = train_df[train_df[X2]==train_machines[0]]
         inp_df = moving_average(df,window)
    
    
    if len(test_machines)>1:
        for i in range(len(test_machines)):
            df = test_df[train_df[X2]==test_machines[i]]
            df = moving_average(df,window)
            predict_df = pd.concat([predict_df,df], axis = 0, ignore_index = True)
    else:
        df = test_df[test_df[X2]==test_machines[0]]
        predict_df = moving_average(df,window)
        
            
    (Week_train, Year_train, Week_Year_train) = Get_Week_Year_Number(inp_df,Col)
    (Week_test, Year_test, Week_Year_test) = Get_Week_Year_Number(predict_df,Col)       
        
    y_train = inp_df[dependent_variable[j]].values
    tr_l = list(inp_df.columns)
    if dependent_variable[j] in tr_l: tr_l.remove(dependent_variable[j])
    X = inp_df[tr_l]

    df_train=pd.DataFrame(X,columns=tr_l)
    temp = tr_l
    if 'Week_Number' in temp: temp.remove('Week_Number')
    if 'UTCDeviceTimeStamp' in temp: temp.remove('UTCDeviceTimeStamp')
    if 'Device' in temp: temp.remove('Device')
    
    temp = [i for i in temp if X1 not in i]
    columns_train=temp
    len(columns_train)
    X=df_train[columns_train]
    
    device_name = ''
    for i in range(len(train_machines)):
        device_name+=str(train_machines[i])
    
    para=[]    
    (max_depth,min_child_weight,gamma,subsample,colsample_bytree,reg_alpha,learn_rate)=XGB_parameter(X,y_train)
    para.append([dependent_variable[j],max_depth,min_child_weight,gamma,subsample,colsample_bytree,reg_alpha,learn_rate])   
    para_df=pd.DataFrame(para,columns=['Feature','max_depth','min_child_weight','gamma','subsample','colsample_bytree','reg_alpha','learn_rate'])
    para_df.to_excel(write_path+"/"+device_name+'_XGB_Parameters.xlsx')
    (regr_xgb,xgb_df) = XGB_train(X,y_train,columns_train,max_depth,min_child_weight,gamma,subsample,colsample_bytree,reg_alpha,learn_rate,device_name,dependent_variable[j])    
    list_of_features = list(xgb_df.feature_name)
    (keep, others)= remove_correlated_independent_variables(df_train,list_of_features,threshold) 
    keep = keep[0:upto]
    
    X_upd= df_train[keep]
    columns_upd = keep
    
    (regr_xgb_upd,xgb_df_upd) = XGB_train(X_upd,y_train,columns_upd,max_depth,min_child_weight,gamma,subsample,colsample_bytree,reg_alpha,learn_rate,device_name,dependent_variable[j])

    joblib.dump(regr_xgb_upd, write_path+"/"+dependent_variable[j]+'_xgb_train_model.pkl')
    
    train_score=[]
    device=[]
    dep_variable =[]
    
    train_score.append(regr_xgb_upd.score(X_upd,y_train))
    device.append(device_name)
    dep_variable.append(dependent_variable[j])
    
    
    
    
    predict_df['Week_Number']=Week_test
    predict_df['Year']=Year_test
    predict_df['Week_Year']=Week_Year_test
    wk = list(predict_df.groupby([predict_df['Week_Number'],predict_df['Year']]))
    
    test_scores = []
    rms_wk = []
    mape_wk = []
    mae_wk = []
    week_yr_no = []

    rsq=pd.DataFrame()
    mae=pd.DataFrame()
    rms=pd.DataFrame()
    mape=pd.DataFrame()
    regression_output = pd.DataFrame()
    actual_avg_output= []
    predicted_avg_output = []
    for i in range(0,len(wk)):
        wk_test_df = wk[i][1]
        wk_test_name = wk[i][0]
        week_yr_no.append(wk[i][0])
        y_test = wk_test_df[dependent_variable[j]].values
        actual_avg_output.append(y_test.mean())
        X_test = wk_test_df[columns_upd]
        columns_test = columns_upd
        xgb_mod = joblib.load(write_path+"/"+dependent_variable[j]+'_xgb_train_model.pkl')
        y_test_xgb = xgb_mod.predict(X_test)
        predicted_avg_output.append(y_test_xgb.mean())
        test_scores.append(regr_xgb_upd.score(X_test, y_test))
        df3 = pd.DataFrame()
        df3 = pd.DataFrame({'week_number':str(wk[i][0]), 'y_actual':y_test,'y_predicted': y_test_xgb})
        rms_wk.append(sqrt(mean_squared_error(list(df3.y_actual),list(df3.y_predicted))))
        mape_wk.append(mean_absolute_percentage_error(list(df3.y_actual.reshape(-1,1)),list(df3.y_predicted.reshape(-1,1)) ))
        mae_wk.append(mean_absolute_error(list(df3.y_actual),list(df3.y_predicted)))
    #pd.DataFra
    rsq[dependent_variable[j]]=test_scores
    mae[dependent_variable[j]]=mae_wk
    mape[dependent_variable[j]]=mape_wk
    rms[dependent_variable[j]]=rms_wk
    
    
    rsq['Week_Year']=week_yr_no
    mae['Week_Year']=week_yr_no
    mape['Week_Year']=week_yr_no
    rms['Week_Year']=week_yr_no
    
    regression_output['Week_Year']=week_yr_no
    regression_output['Actual']=actual_avg_output
    regression_output['Predicted']=predicted_avg_output
    
    
    file_write_path = write_path +"/"+ dependent_variable[j]
    if not os.path.exists(file_write_path):
        os.makedirs(file_write_path) 
    
    regression_output.to_excel(file_write_path + "/" + "RegressionOutput.xlsx")
    rsq.to_excel(file_write_path+"/"+"R_square.xlsx")
    mae.to_excel(file_write_path+"/"+"MAE.xlsx")
    mape.to_excel(file_write_path+"/"+"MAPE.xlsx")
    rms.to_excel(file_write_path+"/"+"RMS.xlsx")
    pd.DataFrame({'Dep_Variable':dep_variable,'Train_Score':train_score}).to_excel(file_write_path+"/"+'Training_Scores.xlsx')
        
    