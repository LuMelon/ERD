#!/usr/bin/env python
# coding: utf-8

# In[1]:


from selenium import webdriver
import os
import sys
import time
import random

browser = webdriver.Chrome()

with open("/home/hadoop/ERD/rotten_imdb/obj.data") as fr:
    with open("/home/hadoop/ERD/rotten_imdb/obj_CN.data", "w") as fw:
        try:
            for line in fr:
                trans_url = "https://fanyi.baidu.com/#en/zh/%s"%line.replace(" ", "%20")
                browser.get(trans_url)
                browser.refresh()
                time.sleep(3)
                ele = browser.find_element_by_xpath("//*[@id='main-outer']/div/div/div[1]/div[2]/div[1]/div[2]/div/div/div[1]/p[2]/span") 
                fw.write(ele.text+"\n")
                sleep_time = random.gauss(20, 5)
                time.sleep(sleep_time)
        except:
            fw.close()
            raise

