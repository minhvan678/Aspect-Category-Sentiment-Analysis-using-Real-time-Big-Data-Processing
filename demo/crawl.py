from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.edge.options import Options,ChromiumOptions
from selenium.webdriver import EdgeOptions
import time
from selenium.webdriver.common.by import By
import json
from pymongo import MongoClient
import os
from kafka import KafkaProducer,KafkaConsumer,errors
import datetime
import threading
import multiprocessing 
from concurrent.futures import ProcessPoolExecutor

topic_name = 'traveloka'
kafka_server = 'localhost:9092'

# Classy Boutique Hotel, Milan Homestay - The Song Vung Tau, Seashore Hotel & Apartment, Classy Holiday Hotel & Spa, Khách sạn The Grace Dalat

def comment_from_hotel(args):
    hotel_url, hotel_id,seed = args

    import numpy as np
    np.random.seed(seed)
    mm = np.random.uniform(low=0.5, high=3, size=(1,))[0]
    time.sleep(mm)

    producer = KafkaProducer(bootstrap_servers='localhost:9092',value_serializer = lambda x: json.dumps(x).encode('utf-8'))

    ## init driver
    options = EdgeOptions()
    options.add_argument("--no-sandbox")
    # options.add_argument("--headless")
    options.add_argument("--start-maximized")    
        
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    # Adding argument to disable the AutomationControlled flag 
    options.add_argument("--disable-blink-features=AutomationControlled") 
    options.add_argument('--disable-dev-shm-usage')        

    list_rs = []

    try:
        driver = webdriver.ChromiumEdge(options=options)
        driver.get(hotel_url)
        mm = np.random.uniform(low=0.5, high=3, size=(1,))[0]
        time.sleep(mm)
    except:
        print("cannot load!!!")

    i = 1
    # elem = driver.find_element(By.XPATH,f"//div[@class='css-1dbjc4n r-13awgt0 r-1g40b8q']")
    # driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'nearest'})", elem)
    # elem.click()
    # time.sleep(2.5)
    
    # elem = driver.find_element(By.XPATH,f"//div[@data-testid='dropdown-menu-item'][2]")
    # # driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'nearest'});",elem)
    # elem.click()
    # time.sleep(2.5)

    for z in range(100):

        hotel_page_comment = BeautifulSoup(driver.page_source,'html.parser')
        # print(hotel_name)
        time.sleep(1)
        # print(hotel_page_comment)
        # findcm = hotel_page_comment.find_all("div",{'class':'css-901oao css-cens5h r-cwxd7f r-t1w4ow r-1b43r93 r-majxgm r-rjixqe r-fdjqy7'})
        all_comment_box = hotel_page_comment.find_all("div",{"class":"css-1dbjc4n r-14lw9ot r-h1746q r-kdyh1x r-d045u9 r-18u37iz r-1fdih9r r-1udh08x r-d23pfw"})
        # print(i)
        # print(all_comment_box)
        for j in range(len(all_comment_box)):
            try:
                cur = all_comment_box[j]
                user_name = cur.find_all("div",{"class":"css-901oao r-cwxd7f r-t1w4ow r-ubezar r-b88u0q r-135wba7 r-fdjqy7"})[0].string
                date = cur.find_all("div",{"class":"css-901oao r-1ud240a r-t1w4ow r-1b43r93 r-majxgm r-rjixqe r-fdjqy7"})[1].string
                rating = cur.find_all("div",{"class":"css-901oao r-1i6uqv8 r-t1w4ow r-1b43r93 r-majxgm r-rjixqe r-fdjqy7"})[0].string
                comment = cur.find_all("div",{'class':'css-901oao css-cens5h r-cwxd7f r-t1w4ow r-1b43r93 r-majxgm r-rjixqe r-fdjqy7'})[0].string
            except:
                continue
            try:
                trip_type = cur.find_all("div",{"class":"css-901oao r-cwxd7f r-t1w4ow r-1b43r93 r-majxgm r-rjixqe r-fdjqy7"})[0].string
            except:
                trip_type = "None"
            #date, rating, trip_type, comment, image, region, hotel_url
            rs = {
                "comment": comment,
                'hotel_id':hotel_id,
                'timestamp':datetime.datetime.ctime(datetime.datetime.now())
            }
            producer.send('traveloka', value= rs)
            time.sleep(0.1)
        try:
            driver.find_element(By.XPATH,f"//div[@data-testid='undefined-{i+1}']").click()
            mm = np.random.uniform(low=0.5, high=3, size=(1,))[0]
            time.sleep(mm)

            i += 1
        except Exception as e:
            print("Out range comment pages")
            print(e)
            break
    print("page:",z)

def crawl_comment(hotel_url,hotel_id,producer,topic):
    # user_name, password, db_name = 'nhq188','fiora123','traveloka'
    # a = MongoClient(f"mongodb+srv://{user_name}:{password}@cluster0.vcb5k8u.mongodb.net/?tls=true&tlsInsecure=true")
    # dbase = a[db_name]
    # list_hotel_obj = list(dbase['hotel'].find({}))
    
    # for i in range(5):
    #     hotel_id = list_hotel_name[i]
    #     hotel_url = list_hotel_url[i]
    #     print("At hotel:", hotel_url)

    comment_from_hotel(hotel_url,hotel_id,producer, topic)

if __name__ == "__main__":
    list_hotel_name = ['Classy Boutique Hotel', 'Milan Homestay - The Song Vung Tau', 'Seashore Hotel & Apartment', 'Classy Holiday Hotel & Spa','Khách sạn The Grace Dalat']
    list_hotel_url = [
        "https://www.traveloka.com/vi-vn/hotel/detail?spec=07-03-2024.09-04-2024.1.1.HOTEL.1000000399730",
        "https://www.traveloka.com/vi-vn/hotel/detail?spec=07-03-2024.09-04-2024.1.1.HOTEL.9000001071485",
        "https://www.traveloka.com/vi-vn/hotel/detail?spec=07-03-2024.09-04-2024.1.1.HOTEL.3000020011770",
        "https://www.traveloka.com/vi-vn/hotel/detail?spec=07-03-2024.09-04-2024.1.1.HOTEL.1000000430274",
        "https://www.traveloka.com/vi-vn/hotel/detail?spec=07-03-2024.09-04-2024.1.1.HOTEL.3000010042556"
    ]
    seed_list = [18,12,9,26,19]

    pool = multiprocessing.Pool(processes=5)
    hotel_data = zip(list_hotel_url, list_hotel_name,seed_list)

    pool.map(comment_from_hotel, hotel_data)
    pool.close()
    pool.join()