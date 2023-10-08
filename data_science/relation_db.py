from sqlalchemy import create_engine
import pandas as pd

data = pd.read_csv('address.csv')
print(data)

engine = create_engine('sqlite:///:memory:')

# Remove the 'schema' argument from the to_sql() method
data.to_sql('my_table', engine, if_exists='replace', index=False)

res1 = pd.read_sql_query('SELECT * FROM data_table', engine)
print('Result 1')
print(res1)
print('')
