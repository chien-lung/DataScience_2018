# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 00:15:52 2018

@author: Lung
"""
import re
import sys
import time
import requests
from bs4 import BeautifulSoup
 
def checkTitle(title):
    title = str(title)
    if title.find("[公告]") is 0:
        return None
    else:
        return title
    
def checkDate(article_url):
    article_re = requests.get(article_url)
    article_content = article_re.text
    article_soup = BeautifulSoup(article_content, 'html.parser')
    #check 日期
    article_metas = article_soup.find_all("",{'class','article-meta-value'})
    if len(article_metas) is 0:
        article_main = article_soup.find("div",{"id":"main-content","class":"bbs-screen bbs-content"})
        article_titles = str(article_main.contents)
        pat = '[A-Z][a-z]{2} [A-Z][a-z]{2} [0-9]{1,2} [0-9]{2}:[0-9]{2}:[0-9]{2} 201\d'
        date_list = re.findall(pat,article_titles)
        if len(date_list) == 0:
            return None
        article_date = str(date_list[0])
    elif len(article_metas) < 4:
        article_date = str(article_metas)
    else:
        article_date = str(article_metas[3])
        
    if article_date.find("2017") is not -1:
        return article_date
    else: #Not 2017 articles
        return None

def crawl():
    start_url = '/bbs/Beauty/index1992.html' #index1992-2340 for 2017
    pptBase_url = 'https://www.ptt.cc'
    notFoundSendMail_list = ['https://www.ptt.cc/bbs/Beauty/M.1490936972.A.60D.html',
                             'https://www.ptt.cc/bbs/Beauty/M.1494776135.A.50A.html',
                             'https://www.ptt.cc/bbs/Beauty/M.1503194519.A.F4C.html',
                             'https://www.ptt.cc/bbs/Beauty/M.1504936945.A.313.html',
                             'https://www.ptt.cc/bbs/Beauty/M.1505973115.A.732.html',
                             'https://www.ptt.cc/bbs/Beauty/M.1507620395.A.27E.html',
                             'https://www.ptt.cc/bbs/Beauty/M.1510829546.A.D83.html',
                             'https://www.ptt.cc/bbs/Beauty/M.1512141143.A.D31.html']
    
    tStart = time.time()
    
    res = requests.get(pptBase_url+start_url)
    f_all = open('all_articles.txt','w',encoding='utf-8')
    f_popular = open('all_popular.txt','w',encoding='utf-8')
    
    for page in range(1992,2341): #1992-2341
        content = res.text
        soup = BeautifulSoup(content,"html.parser")
        
        tag_list = soup.find_all("div",{"class":"r-ent"})
        btn_list = soup.find_all("a",{"class":"btn wide"})
        nextPage_url = btn_list[2].get('href')
        
        for tag in tag_list:
            link = tag.find("div",{"class":'title'}).find('a')
            if link is not None: #detect article whether be deleted
                article_url = pptBase_url+link.get('href')
                if article_url in notFoundSendMail_list:
                    continue
                date = str(tag.find("div",{"class":'date'}).string).replace("/","").replace(" ","")
                title = link.contents[0]
                title = checkTitle(title) #check for [公告]
                if title is not None:
                    if (page == 1992 or page == 2340) :
                        if checkDate(article_url) is  None:
                            continue
                    #print(date, ",", title, ",", article_url)
                    f_all.write(date+ ","+ title+ ","+ str(article_url)+ "\n")
                    popular = tag.find("span",{"class":"hl f1"})
                    if popular is not None and popular.string == '爆':
                        f_popular.write(date+ ","+ title+ ","+ str(article_url)+ "\n")
                    
        time.sleep(0.5)
        res = requests.get(pptBase_url+nextPage_url)
        
    tEnd = time.time()
    m,s = divmod(tEnd-tStart, 60)
    print("Cost time: %d min %d sec"%(m,s))
    
    f_all.close()
    f_popular.close()
    
def getUrlList(filename, start_date, end_date):
    start_date = int(start_date)
    end_date = int(end_date)
    url_list = []
    with open(filename, 'r', encoding='utf-8') as f_all:
        for line in f_all:
            article_value_list = line.strip().split(',')
            if int(article_value_list[0]) < start_date or int(article_value_list[0]) > end_date:
                continue
            else:
                url_list.append(article_value_list[-1])
    
    return url_list
    
def push(start_date, end_date):
    all_like_times = 0
    all_boo_times = 0
    like_dict = {}
    boo_dict = {}
    
    tStart = time.time()
    
    url_list = getUrlList("all_articles.txt", start_date, end_date)
    
    for url in url_list:
        res = requests.get(url)
        content = res.text
        soup = BeautifulSoup(content,'html.parser')
        tag_list = soup.find_all('div',{"class":"push"})
        for tag in tag_list:
            article_push = tag.find("span",{"class":"hl push-tag"})
            if article_push is not None:
                user_id = tag.find("span",{"class":"f3 hl push-userid"}).contents[0]
                user_id = str(user_id)
                if user_id in like_dict:
                    like_dict[user_id] += 1
                else:
                    like_dict.update({user_id:1})
                all_like_times += 1
                continue
            
            article_boo = tag.find("span",{"class":"f1 hl push-tag"})
            if article_boo is not None:
                boo_str = str(article_boo.contents[0])
                if boo_str.find("噓") is -1:
                    continue
                user_id = tag.find("span",{"class":"f3 hl push-userid"}).contents[0]
                user_id = str(user_id)
                if user_id in boo_dict:
                    boo_dict[user_id] += 1
                else:
                    boo_dict.update({user_id:1})
                all_boo_times += 1
        time.sleep(0.2)
                
                
    tEnd = time.time()
    m,s = divmod(tEnd-tStart, 60)
    print("Cost time: %d min %d sec"%(m,s))

    sorted_like_list = sorted(like_dict.items() ,key= lambda x:x[1], reverse=True)[0:10]
    sorted_boo_list = sorted(boo_dict.items() ,key= lambda x:x[1], reverse=True)[0:10]

    with open("push[{}-{}].txt".format(start_date, end_date),'w',encoding='utf-8') as f_push:        
        f_push.write("all like: {}\n".format(all_like_times))
        f_push.write("all boo: {}\n".format(all_boo_times))
        for i in range(0,10):
            f_push.write("like #{}: {} {}\n".format(i+1, sorted_like_list[i][0], sorted_like_list[i][1]))
        for i in range(0,10):
            f_push.write("boo #{}: {} {}\n".format(i+1, sorted_boo_list[i][0], sorted_boo_list[i][1]))

def popular(start_date, end_date):
    pat = "https?:[/.\w\s-]*\.(?:jpg|png|jpeg|gif)"+"<"
    
    tStart = time.time()
    
    url_list = getUrlList('all_popular.txt', start_date, end_date)
    
    with open("popular[{}-{}].txt".format(start_date, end_date),'w',encoding='utf-8') as f_popular:
        f_popular.write('number of popular articles: {}\n'.format(len(url_list)))
        
    for url in url_list:
        res = requests.get(url)
        content = res.text
        soup = BeautifulSoup(content,'html.parser')
        main_content = soup.find('div',{'id':'main-content'})
        main_content = str(main_content)
        image_url_list = re.findall(pat, main_content)
        with open("popular[{}-{}].txt".format(start_date, end_date),'a',encoding='utf-8') as f_popular:
            for image_url in image_url_list:
                image_url = image_url[:-1]
                f_popular.write("{}\n".format(image_url))
        time.sleep(0.2)
        
    tEnd = time.time()
    m,s = divmod(tEnd-tStart, 60)
    print("Cost time: %d min %d sec"%(m,s))

def keyword(key, start_date, end_date):
    pat = "https?:[/.\w\s-]*\.(?:jpg|png|jpeg|gif)"+"<"
    
    f_all = open("keyword({})[{}-{}].txt".format(key, start_date, end_date),'a',encoding='utf-8')
    f_all.close()
    tStart = time.time()
    
    url_list = getUrlList("all_articles.txt", start_date, end_date)  

    for url in url_list:
        res = requests.get(url)
        content = res.text
        soup = BeautifulSoup(content,'html.parser')
        main_content = str(soup.find('div',{'id':'main-content'}))
        author_content = main_content[:main_content.find("\n--\n")]
        if author_content.find(key) < 0:
            continue
        image_url_list = re.findall(pat, main_content)
        with open("keyword({})[{}-{}].txt".format(key, start_date, end_date),'a',encoding='utf-8') as f_popular:
            for image_url in image_url_list:
                image_url = image_url[:-1]
                f_popular.write("{}\n".format(image_url))
        time.sleep(0.2)

    tEnd = time.time()
    m,s = divmod(tEnd-tStart, 60)
    print("Cost time: %d min %d sec"%(m,s))  

if __name__ == "__main__":
    if sys.argv[1] == 'crawl':
        crawl()
    elif sys.argv[1] == 'push':
        start_date = sys.argv[2]
        end_date = sys.argv[3]
        push(start_date, end_date)
    elif sys.argv[1] == 'popular':
        start_date = sys.argv[2]
        end_date = sys.argv[3]
        popular(start_date, end_date)
    elif sys.argv[1] == 'keyword':
        key = sys.argv[2]
        start_date = sys.argv[3]
        end_date = sys.argv[4]
        keyword(key, start_date, end_date)
    else:
        print("wrong input")