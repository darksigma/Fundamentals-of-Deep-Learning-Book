#!/usr/bin/python

import sys
import urllib
import re
import json

import socket
socket.setdefaulttimeout(10)

from bs4 import BeautifulSoup

cache = {}

for line in open(sys.argv[1]):
	fields = line.rstrip('\n').split('\t')
	sid = fields[0]
	uid = fields[1]

	#url = 'http://twitter.com/%s/status/%s' % (uid, sid)
	#print url

        tweet = None
	text = "Not Available"
	if cache.has_key(sid):
		text = cache[sid]
	else:
                try:
                        f = urllib.urlopen("http://twitter.com/%s/status/%s" % (uid, sid))
                        #Thanks to Arturo!
                        html = f.read().replace("</html>", "") + "</html>"
                        soup = BeautifulSoup(html)

			jstt   = soup.find_all("p", "js-tweet-text")
			tweets = list(set([x.get_text() for x in jstt]))
			#print len(tweets)
			#print tweets
			if(len(tweets)) > 1:
				continue

			text = tweets[0]
			cache[sid] = tweets[0]

                        for j in soup.find_all("input", "json-data", id="init-data"):
                                js = json.loads(j['value'])
                                if(js.has_key("embedData")):
                                        tweet = js["embedData"]["status"]
                                        text  = js["embedData"]["status"]["text"]
                                        cache[sid] = text
                                        break
                except Exception:
                        continue

        if(tweet != None and tweet["id_str"] != sid):
                text = "Not Available"
                cache[sid] = "Not Available"
        text = text.replace('\n', ' ',)
        text = re.sub(r'\s+', ' ', text)
        #print json.dumps(tweet, indent=2)
        print "\t".join(fields + [text]).encode('utf-8')
