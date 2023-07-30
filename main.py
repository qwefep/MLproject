import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


####ПОДГОТОВКА ДАТАФРЕЙМА

df = pd.read_csv('train_dataset_train.csv')
y1 = df['time_to_under']
y2 = df['label']
#df = df.head(500000)
########## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#column_station_nm = df['station_nm']
#column_station_id = df['station_id']
#column_entrance_id = df['entrance_id']
#column_entrance_nm = df['entrance_nm']

columns_to_drop = [
    ############## НЕНУЖНЫЕ
    'id',
    'ticket_id',

    'station_nm',
    'entrance_nm',
    'line_nm',




]



df.drop(columns_to_drop, axis=1, inplace=True)

#список колонок датасета
print(df.columns)

#описательные статистики для всего датасета
print(df.describe())


#Распечатаем в цикле по каждой колонке название колонки, количество уникальных значений, а затем список возможных значений вместе
# с их количеством появления в датасете.
cols = df.columns
for col in cols:
    print(f"Характеристика: {col}")
    print(f"Количество уникальных значений: {df[col].nunique()}")
    print(f"Список значений: {df[col]}")
    print(df[col])
    print('///////////////////////////////////////////////////')



###########Data Cleaning
#Найдем процент непустых столбцов

values = ((df.isna().sum() / len(df)) * 100).sort_values()
count = 0

for i in values:
    if i == 0:
        count += 1
print(f'Количество полностью заполненных объектов - {count}')
print(f'Их процент из всей выборки - {int(count / len (values) * 100)}%')

#Пропусков нет


######Преобразование переменных



###pass_dttm

#ОСТАВЛЯЕМ ТОЛЬКО ДНИ НЕДЕЛИ И ЧАСЫ

df.pass_dttm = pd.to_datetime(df.pass_dttm)
df['hour'] = df.pass_dttm.apply(lambda x: x.hour)
df['day_of_week'] = df.pass_dttm.dt.weekday
df = df.drop(['pass_dttm'], axis=1)


###Категориальные переменные

columns_to_drop1 = [
    'ticket_type_nm',
    'entrance_id',
    'station_id',
    'line_id'

]

ohe = OneHotEncoder(sparse=False)
ohe.fit(df[['ticket_type_nm','entrance_id','station_id','line_id']])
ohe_model = ohe.transform(df[['ticket_type_nm','entrance_id','station_id','line_id']])
df[ohe.get_feature_names_out()] = ohe_model
df.drop(columns_to_drop1, axis=1, inplace=True)


####Предсказания


X = df.drop(['time_to_under','label'], axis=1)

##Регрессия

X_train, X_test, y1_train, y1_test = train_test_split(X, y1,
                                                    test_size=0.2,
                                                    random_state=True)


reg = LinearRegression().fit(X_train, y1_train)
y1_pred = reg.predict(X_test)

#from sklearn.metrics import mean_squared_error as mse
#from sklearn.metrics import mean_absolute_percentage_error as mape
#print(mse(y1_pred, y1_test))
#print(mape(y1_pred, y1_test))


##Классификация
X_rf_train, X_rf_test, y2_train, y2_test = train_test_split(X, y2,
                                                    test_size=0.2,
                                                    random_state=True)

# Создаём модель леса из 10 деревьев
mrf = RandomForestClassifier(n_estimators=10)

# Обучаем на тренировочных данных
mrf.fit(X_rf_train, y2_train)
# Предсказания
y2_pred = mrf.predict(X_rf_test)

##Метрика

from sklearn.metrics import r2_score
from sklearn.metrics import recall_score

print(0.5 * (r2_score(y1_test, y1_pred) + recall_score(y2_test, y2_pred, average='weighted')))


