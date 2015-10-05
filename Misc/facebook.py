#facebook.py
# to run: cd C:\Python27 > python facebook.py fb_api startdate enddate

import urllib
import facebook
import requests
import json
import datetime
import time
import pyodbc
import urllib2
import json
import csv
import sys
from urllib2 import HTTPError
import textblob
from textblob import TextBlob

		

def fb_api(startdate, enddate, page):
	#Access Facebook API
	end_date='%s/03/2015' %enddate
	start_date='%s/03/2015' %startdate
	t1=int(time.mktime(datetime.datetime.strptime(start_date, "%d/%m/%Y").timetuple())) 
	t2=int(time.mktime(datetime.datetime.strptime(end_date, "%d/%m/%Y").timetuple())) 
	outpath=r'C:\Python27'
	fileout='\message%s_%s.txt' %(startdate, enddate)
	csv_file='facebook%s_%s.csv' %(startdate, enddate)


	access_token='access token gets placed here'

	base_url = 'https://graph.facebook.com/%s' %page
	fields='posts.since(%s).until(%s).limit(100){shares,picture,properties,message,link,caption,name,likes.limit(0).summary(true),created_time, comments.summary(true){like_count, message}}' %(t1, t2)
	url = '%s?fields=%s&access_token=%s' %(base_url, fields, access_token)


	messages=[]
	j=requests.get(url).json()

	a=len(j.get('posts',{}).get('data',[[]]))
	for i in range(a):
			s=j['posts']['data'][i].get('shares',{'count':0}).get('count',0)
			q=j['posts']['data'][i].get('link')
			if q is None:
				u='NA'
			else: 
				try:
					req = urllib2.Request(q)
					req.add_unredirected_header('User-agent', 'Mozilla/5.0 (Windows NT 6.3; rv:36.0) Gecko/20100101 Firefox/36.0')
					url_open = urllib2.urlopen(req)
					u=url_open.url.encode('utf-8')
				except urllib2.HTTPError as e:
					print q
					print e.code
					u=e.code
				#print 'u0=', u
			m=j['posts']['data'][i].get('message','').encode('utf-8')
			blob=textblob.TextBlob(m.decode('utf-8'))
			message_length=len(blob.words)
			t=j['posts']['data'][i].get('created_time')
			c=j['posts']['data'][i].get('comments').get('summary', {'total_count':0}).get('total_count',0)
			l=j['posts']['data'][i].get('likes').get('summary', {'total_count':0}).get('total_count',0)
			p=j['posts']['data'][i].get('picture', '').encode('utf-8')
			v=j['posts']['data'][i].get('properties',[{'text':'NA'}])[0]['text']
			n=j['posts']['data'][i].get('name', '').encode('utf-8')
			messages.append([m, u, t, s, c, l, p, v, n, message_length])
	url=j.get('posts',{}).get('paging',{}).get('next', 'false')
	x=0


	while url!='false':
		j=requests.get(url).json()
		d=json.dumps(j, indent=1)
		g=open('facebook%s.txt'%(x), 'w')
		g.write(d)
		g.write(url)
		g.close()
		a=len(j.get('data',[[]]))
		print 'a',x,'=', a
		for i in range(a):
			s=j['data'][i].get('shares',{'count':0}).get('count',0)
			q=j['data'][i].get('link')
			#print x,i
			if q is None:
				u='NA'
			else: 
				try:
					req = urllib2.Request(q)
					req.add_unredirected_header('User-agent', 'Mozilla/5.0 (Windows NT 6.3; rv:36.0) Gecko/20100101 Firefox/36.0')
					url_open = urllib2.urlopen(req)
					u=url_open.url.encode('utf-8')
					#print 'try worked'
				except urllib2.HTTPError as e:
					print q
					print e.code
					u=e.code
			m=j['data'][i].get('message','').encode('utf-8')
			blob=textblob.TextBlob(m.decode('utf-8'))
			message_length=len(blob.words)
			t=j['data'][i].get('created_time')
			c=j['data'][i].get('comments').get('summary', {'total_count':0}).get('total_count',0)
			l=j['data'][i].get('likes').get('summary', {'total_count':0}).get('total_count',0)
			p=j['data'][i].get('picture', '').encode('utf-8')
			v=j['data'][i].get('properties',[{'text':'NA'}])[0]['text']
			n=j['data'][i].get('name', '').encode('utf-8')
			messages.append([m, u, t, s, c, l, p, v, n, message_length])
		url=j.get('paging',{}).get('next', 'false')
		x=x+1



	headers=['Post Message', 'Link', 'Created_at', 'Shares', 'Comments', 'Likes', 'Picture', 'Video Length', 'Link Title', 'Message Length']
	with open(csv_file, 'wb') as f: 
		f_csv=csv.writer(f)
		f_csv.writerow(headers)
		for row in messages:
			temprow=row
			f_csv.writerows([row])

if __name__ == '__main__':
	fb_api(*sys.argv[2:])

