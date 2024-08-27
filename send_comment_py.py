import re
from bs4 import BeautifulSoup
import numpy as np
from selenium import webdriver
from selenium.webdriver.edge.options import Options
import time
from selenium.webdriver.common.by import By
from json import dumps
from time import sleep
import time
from pandas import DataFrame as df
from kafka import KafkaProducer, KafkaClient
import json
import threading

from CrawlTravel import CrawlComment

def jsondump(x):
    return dumps(x,ensure_ascii=False).encode('utf-8')

topic = 'traveloka'
kafka_server = 'localhost:9092'
producer= KafkaProducer(bootstrap_servers=kafka_server,value_serializer = jsondump)

def func_text(x):
    return x.string
def comment_from_hotel(hotel_url_dict,hotel_name,city_name,producer,topic):
    ## init driver
    options = Options()
    options.headless = True 

    hotel_comment = []
    print(hotel_name)
    
    try:
        driver = webdriver.ChromiumEdge(options=options)
        driver.get(hotel_url_dict[hotel_name])
        time.sleep(10)
    except:
        print("cannot load!!!")

    # usually 7 pages contain user reviews
    page = 0
    for i in range(7):
        try:
            driver.find_element(By.XPATH,f"//div[@data-testid='undefined-{i+1}']").click()
            time.sleep(5)
        except:
            print("Out range comment pages")

        hotel_page_comment = BeautifulSoup(driver.page_source)

        findcm = hotel_page_comment.find_all("div",{'class':'css-901oao css-cens5h r-cwxd7f r-t1w4ow r-1b43r93 r-majxgm r-rjixqe r-fdjqy7'})

        # func_text = lambda x: x.string
        comments = list(map(func_text,list(findcm)))           

        for comment in comments:
            kafka_comment = dict(region = city_name,hotelname = hotel_name,comment = comment)
            producer.send(topic, value=kafka_comment)
            time.sleep(5)

        hotel_comment.extend(comments)

    return hotel_comment

if __name__ == '__main__':
    
    city_name = 'nha trang'
    crawl = CrawlComment()
    hotel_url_dict = crawl.hotel_and_url(city_name)

    hotels_name = list(hotel_url_dict.keys())[:5]
    print(hotels_name)
    threads = []
    for name in hotels_name:
        t = threading.Thread(target=comment_from_hotel, args=(hotel_url_dict,name,city_name,producer,topic))
        threads.append(t)
    
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
