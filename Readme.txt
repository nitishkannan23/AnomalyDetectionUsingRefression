Regression and Aggregation
**************************


The readme file serves the purpose of providing information about the Codes and associated files. 
There are 3 codes in total for this particular project and 1 config file for the same. The codes have 
been tested out with two datasets.


Code - 1
ExtractAlphaNumCol.py

This code takes in a sample machine file of a particular year and identifies all the alpha numeric column names 
and maps them to a root column name with no digits involved. The output here will act as an input file for the Regression
Aggregation code. This can also sort of act as a pointer to what dependent variables can be selected.

Code - 2
RegressionAggregator.py

This code takes in a list of dependent variables, the AlphaNumeric Columns Mapping file and creates aggregate data for each
iteration (each dependent variable). Then it drops the columns used for averaging. The model feature selection includes
RF and Elastic Net methods. Finally the training on the top features and testing on the test data is done and results are 
saved.

Code - 3
XGBAggregator.py

performs regression using the XG Boost Method and uses XGB Parameter tuning for selecting the best XG Boost parameters




Config File
regression_config.ini

The regression_config file can be used as a common config file for the above codes.

directory_path = base directory path 
data_path      = path where machine files are stored 
write_path_extractalpha = path where alphanumeric mapping file is stored
read_util_path = path from which the RegressionAggregator Code picks up some of the 
                 utility/input files
write_path_aggregate_regression = path to store results of the regression

name = identifies the machine type ex. JF or WT

dep = list of dependent variables

train machine
mac = names of training machines

test machine
mac = names of test machines

ignore
cols = list of columns that shouldnt be included in the regression models

timestamp

var = for performing operations on timestamp columns
timestamp = name of the timestamp column to be used for parsing 
remove = remove any other time stamp column

Device
var = for performing operations on the Device columns(this has been put in 
        a variable because different sets of machines might have disparate device
       columns)

Filter
var1 = filter variable 1
var2 = filter variable 2
limit1 =filter limit 1
limit2 = filter limit 2

Correlation
upper = upper limit of correlation of variables with dependent variable
lower = lower limit of correlation of variables with dependent variable
threshold = values of correlation above which all variables are removed


Moving_Avg
window = number of rows to be considered for averaging

feature_numbers
upto = number of features to keep 


Chronology of Codes to be run:
******************************

RawData Year Wise Extracted =====> Pre-Process data =====> ExtractAlphaNumCol.py =====> RegressionAggreagtor.py
