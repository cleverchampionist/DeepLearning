from pymongo import MongoClient
from pprint import pprint

client = MongoClient()

db = client.test 

employee = db.employee
employee_details = {
    'Name' : 'Raj Kumar',
    "Address": 'Sears Streer, NZ',
    'Age': '42'
}

result = employee.insert_one(employee_details)

Queryresult = employee.find_one({'Age':'42'})

pprint(Queryresult)