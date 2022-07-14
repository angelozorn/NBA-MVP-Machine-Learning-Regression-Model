#!/usr/bin/env python
# coding: utf-8

# In[1]:


headers  = {
    'Connection': 'keep-alive',
    'Accept': 'application/json, text/plain, */*',
    'x-nba-stats-token': 'true',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36',
    'x-nba-stats-origin': 'stats',
    'Sec-Fetch-Site': 'same-origin',
    'Sec-Fetch-Mode': 'cors',
    'Referer': 'https://stats.nba.com/',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-US,en;q=0.9',
}
column_names = ["PLAYER_ID",
"PLAYER_NAME",
"NICKNAME",
"TEAM_ID",
"TEAM_ABBREVIATION",
"AGE",
"GP",
"W",
"L",
"W_PCT",
"MIN",
"FGM",
"FGA",
"FG_PCT",
"FG3M",
"FG3A",
"FG3_PCT",
"FTM",
"FTA",
"FT_PCT",
"OREB",
"DREB",
"REB",
"AST",
"TOV",
"STL",
"BLK",
"BLKA",
"PF",
"PFD",
"PTS",
"PLUS_MINUS",
"NBA_FANTASY_PTS",
"DD2",
"TD3",
"WNBA_FANTASY_PTS",
"GP_RANK",
"W_RANK",
"L_RANK",
"W_PCT_RANK",
"MIN_RANK",
"FGM_RANK",
"FGA_RANK",
"FG_PCT_RANK",
"FG3M_RANK",
"FG3A_RANK",
"FG3_PCT_RANK",
"FTM_RANK",
"FTA_RANK",
"FT_PCT_RANK",
"OREB_RANK",
"DREB_RANK",
"REB_RANK",
"AST_RANK",
"TOV_RANK",
"STL_RANK",
"BLK_RANK",
"BLKA_RANK",
"PF_RANK",
"PFD_RANK",
"PTS_RANK",
"PLUS_MINUS_RANK",
"NBA_FANTASY_PTS_RANK",
"DD2_RANK",
"TD3_RANK",
"WNBA_FANTASY_PTS_RANK",
"CFID",
"CFPARAMS"]

cols = ["PLAYER_NAME","W_PCT", "MIN", "GP","FG_PCT_RANK","PTS_RANK", "NBA_FANTASY_PTS_RANK","W_PCT_RANK", "PLUS_MINUS_RANK","REB_RANK","AST_RANK","TOV_RANK","STL_RANK","BLK_RANK","FTA_RANK", "Year"]


# In[1]:


import requests
import pandas as pd
from bs4 import BeautifulSoup


# In[ ]:


d = [*range(96,100,1)]
e = [*range(0,10,1)]
f = [*range(10,23,1)]
for i in range(len(e)):
    e[i] = '%02d' % i
    
for i in range(len(d)):
    d[i] = str(d[i])
    
for i in range(len(f)):
    f[i] = str(f[i])
    
    

years = d + e + f

for i in range(len(years)-1):
    years[i] = str(years[i]+"-"+years[i+1])
    
    if years[i][0] == '8' or years[i][0] == '9':
        years[i] = str('19'+years[i])
        
for i in range(4,len(years)-1,1):
    years[i] = str('20'+years[i])
    

years.pop()
print(years)


# In[ ]:


urlyearslist = []
i = 0
while i < len(years):
    playerinfourl = 'https://stats.nba.com/stats/leaguedashplayerstats?College=&Conference=&Country=&DateFrom=&DateTo=&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=' + years[i] + '&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=&Weight='
    urlyearslist.append(str(playerinfourl))
    i = i+1


# In[9]:


years_list = [*range(1997,2023,1)]
print(len(years_list))


# In[ ]:


data_list = []


j = 0
while j < len(urlyearslist):
    tempurl = urlyearslist[j]
    
    response = requests.get(url=tempurl,headers=headers).json()
    player_info = response['resultSets'][0]['rowSet']
    
    year_data = pd.DataFrame(player_info, columns = column_names)
    #newdf = year_data[cols]
    year_data["Year"] = years_list[j]
    
    data_list.append(year_data)
    
    j = j+1
    
player_stats_all_years = pd.concat(data_list)


# In[ ]:


player_stats_all_years


# In[ ]:


player_stats_all_years.to_csv("full_player_stats.csv")


# In[ ]:


data_list = []

for year in years_list:
    with open("years_html/{}.html".format(year),encoding="utf-8") as f:
        page = f.read()
    
    soup = BeautifulSoup(page, "html.parser")
    soup.find('tr', class_="over_header").decompose()
    mvp_table = soup.find(id="mvp")
    mvp = pd.read_html(str(mvp_table))[0]
    mvp["Year"] = year
    data_list.append(mvp)
    


# In[ ]:



mvp_all_years = pd.concat(data_list)
mvp_all_years.to_csv("mvp_all_years.csv")


# In[ ]:


mvps_df = pd.read_csv("mvp_all_years.csv")


# In[ ]:


mvp_columns = ["Player","Year","Pts Won", "Pts Max", "Share"]
mvps_df = mvps_df[mvp_columns]
mvps_df.rename(columns = {'Player':'PLAYER_NAME'}, inplace = True)
mvps_df["PLAYER_NAME"] = mvps_df["PLAYER_NAME"].str.replace("ć","c", regex=False)
mvps_df["PLAYER_NAME"] = mvps_df["PLAYER_NAME"].str.replace("č","c", regex=False)
mvps_df["PLAYER_NAME"] = mvps_df["PLAYER_NAME"].str.replace("ó","o", regex=False)
mvps_df.head()


# In[ ]:


all_players_df = pd.read_csv("full_player_stats.csv")
#del all_players_df["Unnamed: 0"]

a = mvps_df.loc[(mvps_df['PLAYER_NAME'] == 'Steve Smith')]
a


# In[ ]:


mvps_df = mvps_df[mvps_df['PLAYER_NAME'] != 'Steve Smith']


# In[ ]:


players_and_mvps = all_players_df.merge(mvps_df, how="outer", on=["PLAYER_NAME", "Year"])


# In[ ]:


replace_nan = ["Pts Won", "Pts Max", "Share"]
players_and_mvps[replace_nan] = players_and_mvps[replace_nan].fillna(0)
players_and_mvps.head(10)


# In[ ]:


cols = ["PLAYER_NAME","W_PCT", "MIN", "GP","FG_PCT_RANK","PTS_RANK", "NBA_FANTASY_PTS_RANK","W_PCT_RANK", "PLUS_MINUS_RANK","REB_RANK","AST_RANK","TOV_RANK","STL_RANK","BLK_RANK","FTA_RANK", "Year", "Pts Won", "Pts Max", "Share"]


# In[ ]:


players_and_mvps = players_and_mvps[cols]


# In[ ]:


players_and_mvps


# In[ ]:


players_and_mvps = players_and_mvps[players_and_mvps['PLAYER_NAME'].notna()]
pd.isnull(players_and_mvps).sum()


# In[ ]:


players_and_mvps.to_csv("ml_players_mvp_stats.csv")


# In[ ]:


players_and_mvps.corr()["Share"].sort_values(ascending=False)


# In[2]:


input_data = pd.read_csv("ml_players_mvp_stats.csv")
input_data.columns
del input_data["Unnamed: 0"]


# In[3]:


ml_years = list(range(1997,2023))
all_data = []
for year in ml_years:
    test = input_data[input_data["Year"] == year]
    all_data.append(test)
print(all_data)


# In[4]:


m = 0
while m<len(ml_years):
    if m == 23:
        all_data[m] = all_data[m][all_data[m]['GP']>40]
        all_data[m] = all_data[m][all_data[m]['MIN']>25]
        m = m+1
        continue
        
    else:
        all_data[m] = all_data[m][all_data[m]['GP']>55]
        all_data[m] = all_data[m][all_data[m]['MIN']>25]
    
    m = m+1


# In[5]:


input_data = pd.concat(all_data)


# In[6]:


features = ['W_PCT', 'MIN', 'GP', 'FG_PCT_RANK', 'PTS_RANK',
       'NBA_FANTASY_PTS_RANK', 'W_PCT_RANK', 'PLUS_MINUS_RANK', 'REB_RANK',
       'AST_RANK', 'TOV_RANK', 'STL_RANK', 'BLK_RANK', 'FTA_RANK']


# In[23]:


one_year_train = input_data[input_data["Year"]< 2022]
one_year_test = input_data[input_data["Year"] == 2022]


# In[24]:


from sklearn.linear_model import Ridge
sample_reg = Ridge(alpha=.1)
sample_reg.fit(one_year_train[features], one_year_train["Share"])


# In[25]:


sample_predictions = sample_reg.predict(one_year_test[features])
sample_predictions = pd.DataFrame(sample_predictions, columns = ["predictions"], index=one_year_test.index)
sample_playerprediction = pd.concat([one_year_test[["PLAYER_NAME","Share"]],sample_predictions],axis=1)


# In[26]:


sample_playerprediction = sample_playerprediction.sort_values("Share", ascending=False)
sample_playerprediction["rank"] = list(range(1,sample_playerprediction.shape[0]+1))
sample_playerprediction.head(10)


# In[27]:


sample_playerprediction = sample_playerprediction.sort_values("predictions", ascending=False)
sample_playerprediction["predicted_rank"] = list(range(1,sample_playerprediction.shape[0]+1))
sample_playerprediction.head(10)
#ridge regression training on 25 years and testing on 2022 did not show the correct prediction


# In[4]:


ml_years = list(range(1997,2023))
ml_years


# In[28]:


all_predictions = []
for year in ml_years[5:]:
    train = input_data[input_data["Year"] < year]
    test = input_data[input_data["Year"] == year]
    sample_reg.fit(train[features], train["Share"])
    predictions = sample_reg.predict(test[features])
    predictions = pd.DataFrame(predictions, columns = ["predictions"], index=test.index)
    combination = pd.concat([test[["PLAYER_NAME","Share"]],predictions],axis=1)
    all_predictions.append(combination)


# In[8]:


def add_ranks(combination):
    combination = combination.sort_values("Share", ascending=False)
    combination["rank"] = list(range(1,combination.shape[0]+1))
    combination = combination.sort_values("predictions", ascending=False)
    combination["predicted_rank"] = list(range(1,combination.shape[0]+1))
    return combination


# In[9]:


def ml_test(stats,model,years,predictors):
    all_predictions = []
    for year in years:
       
        train = stats[stats["Year"] < year]
        test = stats[stats["Year"] == year]
        model.fit(train[predictors], train["Share"])
        predictions = model.predict(test[predictors])
        predictions = pd.DataFrame(predictions, columns = ["predictions"], index=test.index)
        combination = pd.concat([test[["PLAYER_NAME","Share"]],predictions],axis=1)
        combination = add_ranks(combination)
        all_predictions.append(combination)
    return all_predictions


# In[31]:


get_ipython().system(' pip install sklearn')


# In[10]:


def scores(predictions):
    correct = 0
    total = 0
    for year in predictions:
        row_1=year.iloc[0]
        if row_1['rank'] == 1 and row_1['predicted_rank'] == 1:
            correct += 1
        total += 1
    score = correct/total
    return score


# In[34]:


alpha_list = [x/10 for x in range(0, 101, 1)]
def ridge_alpha_tuning(stats,alpha_list,years,predictors):
    alpha_tuning_predictions = []
    alpha_scores = pd.DataFrame(columns = ['Alpha', 'Score'])
    for alpha in alpha_list:
        model = Ridge(alpha=alpha)
        for year in years:
            train = stats[stats["Year"] < year]
            test = stats[stats["Year"] == year]
            model.fit(train[predictors], train["Share"])
            predictions = model.predict(test[predictors])
            predictions = pd.DataFrame(predictions, columns = ["predictions"], index=test.index)
            combination = pd.concat([test[["PLAYER_NAME","Share"]],predictions],axis=1)
            combination = add_ranks(combination)
            alpha_tuning_predictions.append(combination)
        score = scores(alpha_tuning_predictions)
        
        alpha_scores = alpha_scores.append({'Alpha' : alpha, 'Score' : score}, ignore_index = True)
    return alpha_scores


ridge_alpha_tuning_test = ridge_alpha_tuning(input_data,alpha_list,ml_years[5:], features)
print(ridge_alpha_tuning_test)


# In[35]:


print(ridge_alpha_tuning_test.sort_values("Score", ascending=False))
#ridge regression training on 5 years and testing on 21 years produced a best accuracy of 0.52 with small alphas


# In[36]:


ridge_alpha_tuning_test_ten = ridge_alpha_tuning(input_data,alpha_list,ml_years[10:], features)
print(ridge_alpha_tuning_test_ten)
#ridge regression training on 10 years and testing on 16 years produced a best accuracy of 0.63 with small alphas


# In[33]:


from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg_predictions = ml_test(input_data, linear_reg,ml_years[5:], features)
linear_reg_score = scores(linear_reg_predictions)
print(linear_reg_score)
#Linear regression training on 5 years and testing on 21 years produced an accuracy of 0.52 same as ridge regression


# In[38]:



linear_reg_predictions_ten = ml_test(input_data, linear_reg,ml_years[10:], features)
linear_reg_score_ten = scores(linear_reg_predictions_ten)
print(linear_reg_score_ten)
#Linear regression training on 10 years and testing on 16 years produced an accuracy of 0.625 same as ridge regression


# In[39]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 50, random_state=1, min_samples_split=4)
rf_all_predictions = ml_test(input_data, rf,ml_years[5:], features)

rf_score = scores(rf_all_predictions)
print(rf_score)
print(rf_all_predictions)

# random forest regression with the above parameters training on 5 years and testing on 10 years produce an accuracy of 0.67


# In[46]:



rf_all_predictions_ten = ml_test(input_data, rf,ml_years[10:], features)

rf_score_ten = scores(rf_all_predictions_ten)
print(rf_score_ten)
# random forest regression with the above parameters training on 10 years and testing on 21 years produce an accuracy of 0.69


# In[40]:


from sklearn.model_selection import RandomizedSearchCV
import numpy as np

n_estimators = [int(x) for x in range(50,1050,50)]

max_features = ['auto', 'sqrt']

max_depth = [int(x) for x in range(10, 110, 10)]
max_depth.append(None)
print(max_depth)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# In[41]:


rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                              n_iter = 100, scoring='neg_mean_absolute_error', 
                              cv = 3, verbose=2, random_state=42, n_jobs=-1,
                              return_train_score=True)

rf_random.fit(train[features], train["Share"])
rf_random.best_params_


# In[43]:


rf_parameter_tuned = RandomForestRegressor(n_estimators = 100, min_samples_split = 2, min_samples_leaf = 2, max_features = 'auto', max_depth = 50, bootstrap = True)
rf_parameter_tuned_predictions = ml_test(input_data, rf_parameter_tuned,ml_years[5:], features)

rf_parameter_tuned_score = scores(rf_parameter_tuned_predictions)
print(rf_parameter_tuned_score)

#Hyper parameter tuned random forest regressor training for 5 years and testing for 21 makes accurary 67% - no improvement in accuracy


# In[44]:


from sklearn.ensemble import RandomForestRegressor
rf_parameter_tuned = RandomForestRegressor(n_estimators = 100, min_samples_split = 2, min_samples_leaf = 2, max_features = 'auto', max_depth = 50, bootstrap = True)
rf_parameter_tuned_predictions_ten = ml_test(input_data, rf_parameter_tuned, ml_years[10:], features)

rf_parameter_tuned_score_ten = scores(rf_parameter_tuned_predictions_ten)
print(rf_parameter_tuned_score_ten)

#Hyper parameter tuned random forest regressor training for 10 years and testing for 16 makes accurary 75%


# In[11]:


from sklearn.svm import SVR
svm_classifier = SVR()
svm_all_predictions = ml_test(input_data, svm_classifier,ml_years[5:], features)

svm_score = scores(svm_all_predictions)
print(svm_score)
print(svm_all_predictions)
# Support vector regression testing on 5 years and testing on 21 gives an accuracy of 0.52


# In[48]:


svm_all_predictions_ten = ml_test(input_data, svm_classifier,ml_years[10:], features)

svm_score_ten = scores(svm_all_predictions_ten)
print(svm_score_ten)

# Support vector regression testing on 10 years and testing on 16 gives an accuracy of 0.63


# In[ ]:


from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,1, 20, 50], 'epsilon': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
grid = GridSearchCV(svm_classifier,param_grid, return_train_score=True)

grid.fit(train[features], train["Share"])
grid.best_params_


# In[12]:



svm_tuned_regressor = SVR(C=50,epsilon=.001,kernel="rbf")
svm_tuned_all_predictions = ml_test(input_data, svm_tuned_regressor,ml_years[5:], features)

svm_tuned_score = scores(svm_tuned_all_predictions)
print(svm_tuned_score)
print(svm_tuned_all_predictions)
# Hyperparameter Tuned support vector regression testing on 5 years and testing on 21 gives an accuracy of 0.57


# In[13]:


svm_tuned_all_predictions_ten = ml_test(input_data, svm_tuned_regressor,ml_years[10:], features)

svm_tuned_score_ten = scores(svm_tuned_all_predictions_ten)
print(svm_tuned_score_ten)
# Hyperparameter Tuned support vector regression testing on 5 years and testing on 21 gives an accuracy of 0.56
# this is interesting


# In[14]:


from sklearn.tree import DecisionTreeRegressor

decision_tree = DecisionTreeRegressor()
decision_tree_predictions = ml_test(input_data, decision_tree,ml_years[5:], features)

decision_tree_score = scores(decision_tree_predictions)
print(decision_tree_score)
print(decision_tree_predictions)
# Decision Tree regression training on 5 years testing on 21 produces accuracy of 0.47


# In[16]:


decision_tree_predictions_ten = ml_test(input_data, decision_tree,ml_years[10:], features)

decision_tree_score_ten = scores(decision_tree_predictions_ten)
print(decision_tree_score_ten)
# Decision Tree regression training on 10 years testing on 16 produces accuracy of 0.44
# Lower then training on 5 years


# In[17]:


from sklearn.neighbors import KNeighborsRegressor
k_neighbors = KNeighborsRegressor()

k_neighbors_predictions = ml_test(input_data, k_neighbors,ml_years[5:], features)

k_neighbors_score = scores(k_neighbors_predictions)
print(k_neighbors_score)
print(k_neighbors_predictions)
# k_nearest neighbors regression training for 5 years and testing for 21 produces 0.38 accuracy


# In[18]:


k_neighbors_predictions_ten = ml_test(input_data, k_neighbors,ml_years[10:], features)

k_neighbors_score_ten = scores(k_neighbors_predictions_ten)
print(k_neighbors_score_ten)
# k_nearest neighbors regression training for 10 years and testing for 16 produces 0.44 accuracy


# In[ ]:


#Highest accuracy model is hyper parameter tuned random forest regressor training for 10 years and testing for 16 with accurary 75%

