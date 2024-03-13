import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
 
RANDOM = 8
 
df = pd.read_csv('../datafiles/bike.tsv',sep="\t")
temp= pd.read_json("../datafiles/temp.json")
temp=temp.T
# weather = pd.read_csv("../datafiles/weather.csv", encoding='cp932')
# df2=df.merge(weather,how="inner",on="weather_id")
 
df3 = df.merge(temp,how="left",on="dteday")
 
# 欠損値の線形補完
inter_col = ["atemp","hum","temp","windspeed"]
df3[inter_col] = df3[inter_col].astype(float)
df3[inter_col] = df3[inter_col].interpolate()
 
# データを学習とテストに分離
train,test = train_test_split(df3,test_size=0.2,random_state=RANDOM)
 
# 学習に使用する列を定義
columns = [
    "holiday",
    "weekday",
    "workingday",
    "weather_id",
    "atemp",
    "hum",
#     "temp",
    "windspeed"
    ]  
x = train.loc[:,columns ].astype('float')
# x2 多項式特徴量
x['atemp2'] = x['atemp']*x['atemp']
x['weather_id2'] = x['weather_id']*x['weather_id']
x['windspeed2'] = x['windspeed']*x['windspeed']
# x3 多項式特徴量
x['atemp3'] = x['atemp']*x['atemp']*x['atemp']
x['weather_id3'] = x['weather_id']*x['weather_id']*x['weather_id']
x['windspeed3'] = x['windspeed']*x['windspeed']*x['windspeed']
y = train[['cnt']]
 
# x_train, x_val, y_train, y_val = train_test_split(x, t,
#     test_size = 0.2, random_state = RANDOM)
 
model = LinearRegression()
model.fit(x,y) # 欠損値予測のためのモデルを予測
print('val:',model.score(x, y))
 
x_test = test.loc[:,columns].astype('float')
# 多項式特徴量
x_test['atemp2'] = x_test['atemp']*x_test['atemp']
x_test['weather_id2'] = x_test['weather_id']*x_test['weather_id']
x_test['windspeed2'] = x_test['windspeed']*x_test['windspeed']
# 多項式特徴量
x_test['atemp3'] = x_test['atemp']*x_test['atemp']*x_test['atemp']
x_test['weather_id3'] = x_test['weather_id']*x_test['weather_id']*x_test['weather_id']
x_test['windspeed3'] = x_test['windspeed']*x_test['windspeed']*x_test['windspeed']
y_test = test[['cnt']]
print('test: ',model.score(x_test, y_test))