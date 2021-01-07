import sys
import datetime
import time
from selenium import webdriver
from bs4 import BeautifulSoup
import requests
import re
import argparse
from multiprocessing import Pool

def crawl(url):
    # open files
    all_article_file = open("all_articles.txt", "wb+")
    all_pupolar_file = open("all_popular.txt", "wb+")

    # mark start time
    start_time=datetime.datetime.now()
    # total pages in 2019
    num_page = 395

    while(num_page > 0):
        # solve over 18
        try:
            res = requests.get(url, cookies={"over18": "1"})
        except Exception as e: print(e)
        # pretend not like a hacker
        time.sleep(0.1)

        # start crawling
        soup = BeautifulSoup(res.text, "html.parser")
        posts = soup.find_all(class_="r-ent")

        for post in posts:
            # parse date
            date = post.find_all(class_="date")
            day_str = date[0].string.split('/')
            day = int(day_str[0] + day_str[1])
            # 12/31
            if url == "https://www.ptt.cc/bbs/Beauty/index3142.html" and int(day_str[0]) == 1:
                continue
            # 1/1
            if url == "https://www.ptt.cc/bbs/Beauty/index2748.html" and int(day_str[0]) == 12:
                continue
            # find url link
            link = post.find('a')
            if link:
                url_info = "https://www.ptt.cc" + link.get("href")
            else:
                continue
            # find title
            title = list(link.strings)
            # generate output line
            output_line = str(day) + ',' + "".join(title) + ',' + url_info + '\n'
            # print(output_line.rstrip())
            # check for "公告" -> remove
            if re.search("公告", output_line):
                continue
            all_article_file.write(output_line.encode("utf-8"))
            # check for "爆" -> add in popular file
            if re.search("爆", str(post.find_all("span"))):
                all_pupolar_file.write(output_line.encode("utf-8"))
        
        # go to the next page
        url = "https://www.ptt.cc" + soup.find_all(class_ = "btn wide")[2].get("href")
        num_page = num_page - 1
    
    # close files
    all_article_file.close()
    all_pupolar_file.close()

    # mark end time and calculate time consumption
    end_time = datetime.datetime.now()
    print("spent time: ", end_time-start_time)

def push(search_start_time, search_end_time):
    # mark start time
    start_time = datetime.datetime.now()
    
    # read data from .txt
    all_article_file = open("all_articles.txt", "r")
    posts = all_article_file.readlines()
    all_article_file.close()
    # open output file
    push_output_file = open("push[%d-%d].txt" %(search_start_time, search_end_time), "wb+")

    # initialize parameters
    like_count = 0
    boo_count = 0
    push_dict = {}

    for post in posts:
        select_post = post.split(',')
        # parse
        post_day = int(select_post[0])
        # post_title = str(select_post[1])
        post_url = str(select_post[-1]).rstrip()

        # check date range
        if post_day > search_end_time:
            break
        if post_day >= search_start_time:
            # solve over 18
            try:
                res = requests.get(post_url, cookies={"over18": "1"})
            except Exception as e: print(e)
            # pretend not like a hacker
            time.sleep(0.1)

            # start crawling
            soup = BeautifulSoup(res.text, "html.parser")    
            all_push = soup.find_all(class_="push")
            check_exist = soup.find(class_="bbs-screen bbs-content")
            if check_exist:
                content = check_exist.text
            else:
                content = "N/A"
            end = "※ 發信站"
            check = re.search(end, content)

            if check:
                for push in all_push:
                    push_info = push.find_all("span")
                    if len(push_info) != 0:
                        # tag = push_info[0].string
                        user_id = push_info[1].string
                    # find and count like
                    if re.search("推", str(push_info)):
                        like_count += 1
                        if user_id in push_dict:
                            push_dict[user_id]["like"] += 1
                        else:
                            push_dict[user_id] = {"like": 1, "boo": 0}
                    # find and count boo
                    if re.search("噓", str(push_info)):
                        boo_count += 1
                        if user_id in push_dict:
                            push_dict[user_id]["boo"] += 1
                        else:
                            push_dict[user_id] = {"like": 0, "boo": 1}
            else:
                print("there is no \"※ 發信站\" in ", post_url)
                continue

    # output total like/boo count
    output_line = []
    output_line.append("all like: %d\n" %like_count)
    output_line.append("all boo: %d\n" %boo_count)
    
    # sort like rank
    like_rank = sorted(push_dict, key=lambda x: (push_dict[x]["like"]*-1, x), reverse=True)
    for i, j in enumerate(reversed(like_rank[-10:])):
        temp = "like #%d: %s %d\n" %((i+1), j, push_dict[j]["like"])
        # print(temp.rstrip())
        output_line.append(temp)
    # sort boo rank
    boo_rank = sorted(push_dict, key=lambda x: (push_dict[x]["boo"]*-1, x), reverse=True)
    for i, j in enumerate(reversed(boo_rank[-10:])):
        temp = "boo #%d: %s %d\n" %((i+1), j, push_dict[j]["boo"])
        # print(temp.rstrip())
        output_line.append(temp)

    # output file
    output_line = "".join(output_line) + '\n'
    push_output_file.write((output_line).encode('utf-8'))
    push_output_file.close()

    # mark end time and calculate time consumption
    end_time = datetime.datetime.now()
    print("spent time: ", end_time-start_time)

def popular(search_start_time, search_end_time):
    # mark start time
    start_time = datetime.datetime.now()
    
    # read data from .txt
    all_popular_file = open("all_popular.txt", "r")
    popular_posts = all_popular_file.readlines()
    all_popular_file.close()
    # open output file
    popular_output_file = open("popular[%d-%d].txt" %(search_start_time, search_end_time), "wb+")

    # initialize parameters
    popular_number = 0
    output_line = []
    output_temp = []
    img_url = []

    for post in popular_posts:
        # parse
        select_post = post.split(',')
        post_day = int(select_post[0])
        # post_title = str(select_post[1])
        post_url = str(select_post[-1]).rstrip()

        # check date range
        if post_day > search_end_time:
            break
        if post_day >= search_start_time:
            popular_number += 1
            # solve over 18
            try:
                res = requests.get(post_url, cookies={"over18": "1"})
            except Exception as e: print(e)
            # pretend not like a hacker
            time.sleep(0.1)

            # start crawling
            soup = BeautifulSoup(res.text, "html.parser")    
            check_exist = soup.find(class_="bbs-screen bbs-content")
            if check_exist:
                content = check_exist.text
            else:
                content = "N/A"
            end = "※ 發信站"
            check = re.search(end, content)

            if check:
                img_url_pattern = 'href="(http|https)(.*)?(jpg|jpeg|png|gif)'
                img_url = re.findall(img_url_pattern, soup.prettify())
                for string in img_url:
                    output_temp.append("".join(string) + '\n')
            else:
                print("no \"發信站\" in \"%s\"" %(post_url))
                continue

    # output file
    output_line.append("number of popular articles: %d\n" %(popular_number))
    output_line.append("".join(output_temp) + '\n')
    output_line = "".join(output_line) + '\n'
    popular_output_file.write((output_line).encode('utf-8'))
    popular_output_file.close()

    # mark end time and calculate time consumption
    end_time = datetime.datetime.now()
    print("spent time: ", end_time-start_time)

def keyword(key_word, search_start_time, search_end_time):
    # mark start time
    start_time = datetime.datetime.now()
    
    # read data from .txt
    all_article_file = open("all_articles.txt", "r")
    posts = all_article_file.readlines()
    all_article_file.close()
    # open output file
    keyword_output_file = open("keyword(%s)[%d-%d].txt" %(key_word, search_start_time, search_end_time), "wb+")

    # initialize parameters
    output_line = []
    output_temp = []

    for post in posts:
        # parse
        select_post = post.split(',')
        post_day = int(select_post[0])
        # post_title = str(select_post[1])
        post_url = str(select_post[-1]).rstrip()

        # check date range
        if post_day > search_end_time:
            break
        if post_day >= search_start_time:
            # solve over 18
            try:
                res = requests.get(post_url, cookies={"over18": "1"})
            except Exception as e: print(e)
            # pretend not like a hacker
            time.sleep(0.1)

            # start crawling
            soup = BeautifulSoup(res.text, "html.parser")    
            check_exist = soup.find(class_="bbs-screen bbs-content")
            if check_exist:
                content = check_exist.text
            else:
                content = "N/A"
            end = "※ 發信站"
            check = re.search(end, content)

            if check:
                content_list = content.split('\n')

                for match in content_list:
                    if re.search(end, match):
                        break
                    else:
                        if re.search(key_word, match):
                            print("Find!", post_url)
                            img_url_pattern = 'href="(http|https)(.*)?(jpg|jpeg|png|gif)'
                            img_url = re.findall(img_url_pattern, soup.prettify())
                            for string in img_url:
                                output_temp.append("".join(string) + '\n')
            else:
                print("no \"發信站\" in \"%s\"" %(post_url))
                continue

    # output file
    output_line.append("".join(output_temp) + '\n')
    output_line = "".join(output_line) + '\n'
    keyword_output_file.write((output_line).encode('utf-8'))
    keyword_output_file.close()

    # mark end time and calculate time consumption
    end_time = datetime.datetime.now()
    print("spent time: ", end_time-start_time)

def craw_img(search_start_time, search_end_time):
    # mark start time
    start_time = datetime.datetime.now()
    
    # read data from .txt
    all_article_file = open("all_articles.txt", "r")
    posts = all_article_file.readlines()
    all_article_file.close()
    # open output file
    img_url_output_file = open("article_img_url[%d-%d].txt" %(search_start_time, search_end_time), "wb+")

    # initialize parameters
    article_number = 0
    output_line = []
    output_temp = []
    img_url = []

    for post in posts:
        # parse
        select_post = post.split(',')
        post_day = int(select_post[0])
        # post_title = str(select_post[1])
        post_url = str(select_post[-1]).rstrip()

        # check date range
        if post_day > search_end_time:
            break
        if post_day >= search_start_time:
            article_number += 1

            # solve over 18
            try:
                res = requests.get(post_url, cookies={"over18": "1"})
            except Exception as e: print(e)
            # pretend not like a hacker
            time.sleep(0.1)

            # start crawling
            soup = BeautifulSoup(res.text, "html.parser")    
            check_exist = soup.find(class_="bbs-screen bbs-content")
            if check_exist:
                content = check_exist.text
            else:
                content = "N/A"
            end = "※ 發信站"
            check = re.search(end, content)

            if check:
                img_url_pattern = 'href="(http|https)(.*)?(jpg|jpeg|png|gif)'
                img_url = re.findall(img_url_pattern, soup.prettify())
                for string in img_url:
                    output_temp.append("".join(string) + '\n')
            else:
                print("no \"發信站\" in \"%s\"" %(post_url))
                continue

    # output file
    output_line.append("number of articles: %d\n" %(article_number))
    output_line.append("".join(output_temp) + '\n')
    output_line = "".join(output_line) + '\n'
    img_url_output_file.write((output_line).encode('utf-8'))
    img_url_output_file.close()

    # mark end time and calculate time consumption
    end_time = datetime.datetime.now()
    print("spent time: ", end_time-start_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mod", type=str, nargs="+")
    args = parser.parse_args()

    if args.mod[0] == "crawl":
        print("2019 Beauty crawler") 
        url = "https://www.ptt.cc/bbs/Beauty/index2748.html"
        crawl(url)

    elif args.mod[0] == "push":
        print("crawl push and boo from", int(args.mod[1]), "to", int(args.mod[2]))
        push(int(args.mod[1]), int(args.mod[2]))
    
    elif args.mod[0] == "popular": 
        print("crawl popular from", int(args.mod[1]), "to", int(args.mod[2]))
        popular(int(args.mod[1]), int(args.mod[2]))
    
    elif args.mod[0] == "keyword":
        print("crawl and search", args.mod[1], "from", int(args.mod[2]), "to", int(args.mod[3]))
        keyword(str(args.mod[1]),int(args.mod[2]), int(args.mod[3]))

    elif arg.mode[0] == "craw_img_url":
        print("crawl image url from", int(args.mod[1]), "to", int(args.mod[2]))
        keyword(int(args.mod[1]), int(args.mod[2]))