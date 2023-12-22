from flask import Flask, jsonify
import pickle

app = Flask(__name__)

@app.route("/")
def hello():
  print('Hello')

@app.route("/predict/<float:full_sq>/<float:life_sq>/<float:kitch_sq>/<float:num_room>/\
<float:floor>/<float:max_floor>/<float:state>/<float:material>/<float:build_year>/<int:full_all>/<object:sub_area>/\
<float:salary>/<float:fixed_basket>/<float:rent_price_1room_eco>/<float:rent_price_2room_eco>/<float:rent_price_3room_eco>/<float:average_life_exp>/")
def predict(full_sq,life_sq,kitch_sq,num_room,floor,max_floor,state,material,build_year,full_all,sub_area,salary,fixed_basket,rent_price_1room_eco,rent_price_2room_eco,rent_price_3room_eco,average_life_exp):
  model = joblib.load('data/final/model.pkl')
  y_pred = model.predict([[full_sq,life_sq,kitch_sq,num_room,floor,max_floor,state,material,build_year,full_all,sub_area,salary,fixed_basket,rent_price_1room_eco,rent_price_2room_eco,rent_price_3room_eco,average_life_exp]])[0]
  return jsonify({'predict': y_pred})
