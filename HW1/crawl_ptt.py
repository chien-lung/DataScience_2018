# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 16:48:38 2018

@author: Lung
"""
import re
import time
import requests
from bs4 import BeautifulSoup
 
def checkTitle(title):
    title = str(title)
    if title.find("[公告]") is 0:
        return None
    else:
        return title
    
def checkDateAndFormat(article_url):
    f_weird = open('weird_article.txt',"a",encoding='utf-8')
    article_re = requests.get(article_url)
    article_content = article_re.text
    article_soup = BeautifulSoup(article_content, 'html.parser')
    #check 發信站
    article_sendMail = article_soup.find_all("span",{"class":"f2"})
    if str(article_sendMail).find("發信站:") is -1:
        f_weird.write("找不到發信站,"+str(article_url)+"\n")
        f_weird.close()
        return None
    #check 日期
    article_metas = article_soup.find_all("",{'class','article-meta-value'})
    if len(article_metas) is 0:
        f_weird.write("找不到內文的標題,")
        article_main = article_soup.find("div",{"id":"main-content","class":"bbs-screen bbs-content"})
        article_titles = str(article_main.contents)
        pat = '[A-Z][a-z]{2} [A-Z][a-z]{2} [0-9]{1,2} [0-9]{2}:[0-9]{2}:[0-9]{2} 201\d'
        date_list = re.findall(pat,article_titles)
        if len(date_list) == 0:
            f_weird.write("找不到內文的時間,"+str(article_url)+'\n')
            f_weird.close()
            return None
        f_weird.write(str(article_url)+'\n')
        f_weird.close()
        article_date = str(date_list[0])
    elif len(article_metas) < 4:
        f_weird.write("內文時間格式不同,"+str(article_url)+'\n')
        f_weird.close()
        article_date = str(article_metas)
    else:
        f_weird.close()
        article_date = str(article_metas[3])
        
    if article_date.find("2017") is not -1:
        return article_date
    else: #Not 2017 articles
        return None

def crawl():
    start_url = '/bbs/Beauty/index1992.html' #index1992-2340 for 2017
    pptBase_url = 'https://www.ptt.cc'
    tStart = time.time()
    
    res = requests.get(pptBase_url+start_url)
    f_all = open('all_article.txt','w',encoding='utf-8')
    f_popular = open('all_popular.txt','w',encoding='utf-8')
    f_weird = open('weird_article.txt',"w",encoding='utf-8')
    f_weird.close()
    
    for page in range(1992,2341): #1992-2341
        content = res.text
        soup = BeautifulSoup(content,"html.parser")
        
        tag_lists = soup.find_all("div",{"class":"r-ent"})
        btn_list = soup.find_all("a",{"class":"btn wide"})
        nextPage_url = btn_list[2].get('href')
        
        for tag in tag_lists:
            title = link.contents[0]
            link = tag.find("div",{"class":'title'}).find('a')
            if link is not None: #detect article whether be deleted
                article_url = pptBase_url+link.get('href')
                date_str = str(tag.find("div",{"class":'date'}).string).replace("/","").replace(" ","")
                title = checkTitle(title) #check for Re: and 公告
                if title is not None and checkDateAndFormat(article_url) is not None:                
                    print(date_str, ",", title, ",", article_url)
                    f_all.write(date_str+ ","+ title+ ","+ str(article_url)+ "\n")
                    popular = tag.find("span",{"class":"hl f1"})
                    if popular is not None and popular.string == '爆':
                        f_popular.write(date_str+ ","+ title+ ","+ str(article_url)+ "\n")
                    
        time.sleep(0.5)
        res = requests.get(pptBase_url+nextPage_url)
        
    tEnd = time.time()
    m,s = divmod(tEnd-tStart, 60)
    print("Cost time: %d min %d sec"%(m,s))
    
    f_all.close()
    f_popular.close()


if __name__ == "__main__":
    crawl()