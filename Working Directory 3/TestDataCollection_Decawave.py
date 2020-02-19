import logging
import math
import os
import re
import threading
import time
import requests
import paho.mqtt.client as mqttClient
import ast
from influxdb import InfluxDBClient

broker_address= "192.168.0.198"      
port = 1883   

LOG_LEVEL = 25
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("decawave_reader")

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to broker")
        client.subscribe("#")
    else:
        print("Connection failed")

def on_message(client, userdata, message):
    global influxdb_client

    try:
        data = str(message.payload.decode('utf-8'))
    except:
        print('Data load failure')

    data_dict = ast.literal_eval(data)
    message_list = message.topic.split('/')
    if message_list[2] == '8d91':
        x = data_dict['position']['x']
        y = data_dict['position']['y']
        print('Location', x, y)
        influxdb_client.write_points([
            {
                "measurement": "TestSequenceLocation2",
                "fields": {
                    "X": x,
                    "Y": y
                }
            }
        ])

if __name__ == "__main__":                       
    client = mqttClient.Client("Python")          
    client.on_connect= on_connect    
    client.on_message= on_message        
    client.connect(broker_address, port=port)     
    client.loop_start()

    influxdb_client = InfluxDBClient(host='localhost', port=8086, database='Megallen2')

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("exiting")
        client.disconnect()
        client.loop_stop()
