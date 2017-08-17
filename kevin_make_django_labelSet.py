# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 13:40:35 2017

@author: kevin
"""

import sqlite3 as lite
import json, os, glob
import numpy as np
import datetime
from lxml import html
#import urllib 
import urllib2
import cookielib
import pandas as pd

TIMESTAMP_LIST = ['2017-3-20','2017-3-27','2017-4-10']
plank_prorocentrum_root = '/data4/plankton_wi17/plankton/plankton_binary_classifiers/plankton_phytoplankton'

date = datetime.datetime.utcnow()
date = date.strftime('%s')
date1 = str(int(date)*1000)
date2 = str((int(date)*1000)-500)
# notes
# is machine
proro = {"label":"Prorocentrum","tag":"","images":[],"machine_name":"binary_proro_net01","started": date2, "submitted": date1}

for TIMESTAMP in TIMESTAMP_LIST:
    # Read in list of machine labels
    df = pd.read_csv(plank_prorocentrum_root + '/code/1_target_' + TIMESTAMP + '_image_path_labels.csv')

    # Get image id for each predicted prorocentrum (1)
    for index, row in df.iterrows():
        if row['predictions'] == 1:
            proro['images'].append(row['img_id'])
            #proro['confidence_list'].append(row['confidence_level']
    print TIMESTAMP + ' Total oithona labels: ' + str(len(proro['images']))

# outroot = '/media/storage/learning_files/oithona_project/'
#
# with open(os.path.join(outroot,'django_insertOith_classifier1.json'),'w') as f:
#     json.dump(oith, f, indent=4)
#
# with open(os.path.join(outroot,'django_insertEgg_classifier1.json'),'w') as f:
#     json.dump(egg, f, indent=4)
#
# with open(os.path.join(outroot,'django_insertPara_classifier1.json'),'w') as f:
#     json.dump(para, f, indent=4)

with open(os.path.join(plank_prorocentrum_root,'download_test','django_insertProro_classifier1.json'),'w') as f:
     json.dump(proro, f, indent=4)

proro_json = json.dumps(proro)
print 'done making json docs'

# Login to server
cj = cookielib.CookieJar()

opener = urllib2.build_opener(
    urllib2.HTTPCookieProcessor(cj),
    urllib2.HTTPHandler(debuglevel=1)
)

login_url = 'http://spc.ucsd.edu/data/admin/?next=/data/admin'
login_form = opener.open(login_url).read()

csrf_token = html.fromstring(login_form).xpath(
    '//input[@name="csrfmiddlewaretoken"]/@value'
)[0]

# make values dict
values = {
    'username': 'eric',
    'password': 'ceratium',
    #'csrfmiddlewaretoken': csrf_token,
}

params = json.dumps(values)

req = urllib2.Request('http://spc.ucsd.edu/data/rois/login_user', params, headers={'X-CSRFToken': str(csrf_token),
                                                                                   'X-Requested-With': 'XMLHttpRequest',
                                                                                   'User-agent':'Mozilla/5.0',
                                                                                   'Content-type': 'application/json'})
resp = opener.open(req)
print 'login ' + resp.read()

# write the labels
req1 = urllib2.Request('http://spc.ucsd.edu/data/rois/label_images', proro_json, headers={'X-CSRFToken': str(csrf_token),
                                                                                   'X-Requested-With': 'XMLHttpRequest',
                                                                                   'User-agent':'Mozilla/5.0',
                                                                                   'Content-type': 'application/json'})
resp1 = opener.open(req1)
print 'Proro labs: ' + str(len(resp1.read()))

# make these all json docs:
"""
oith_json = json.dumps(oith)
para_json = json.dumps(para)
egg_json = json.dumps(egg)

print 'done making json docs'
# Login to server
cj = cookielib.CookieJar()

opener = urllib2.build_opener(
    urllib2.HTTPCookieProcessor(cj), 
    urllib2.HTTPHandler(debuglevel=1)
)

login_url = 'http://spc.ucsd.edu/data/admin/?next=/data/admin'
login_form = opener.open(login_url).read()

csrf_token = html.fromstring(login_form).xpath(
    '//input[@name="csrfmiddlewaretoken"]/@value'
)[0]


# make values dict
values = {
    'username': 'eric',
    'password': 'ceratium',
    #'csrfmiddlewaretoken': csrf_token,
}

params = json.dumps(values)

req = urllib2.Request('http://spc.ucsd.edu/data/rois/login_user', params, headers={'X-CSRFToken': str(csrf_token),
                                                                                   'X-Requested-With': 'XMLHttpRequest',
                                                                                   'User-agent':'Mozilla/5.0',
                                                                                   'Content-type': 'application/json'})
resp = opener.open(req)
print 'login ' + resp.read()
# write the labels
req1 = urllib2.Request('http://spc.ucsd.edu/data/rois/label_images', oith_json, headers={'X-CSRFToken': str(csrf_token),
                                                                                   'X-Requested-With': 'XMLHttpRequest',
                                                                                   'User-agent':'Mozilla/5.0',
                                                                                   'Content-type': 'application/json'})
resp1 = opener.open(req1)
print 'Oithona labs: ' + str(len(resp1.read()))

req2 = urllib2.Request('http://spc.ucsd.edu/data/rois/label_images', para_json, headers={'X-CSRFToken': str(csrf_token),
                                                                                   'X-Requested-With': 'XMLHttpRequest',
                                                                                   'User-agent':'Mozilla/5.0',
                                                                                   'Content-type': 'application/json'})
resp2 = opener.open(req2)
print 'Oithona labs: ' + str(len(resp2.read()))

req3 = urllib2.Request('http://spc.ucsd.edu/data/rois/label_images', egg_json, headers={'X-CSRFToken': str(csrf_token),
                                                                                   'X-Requested-With': 'XMLHttpRequest',
                                                                                   'User-agent':'Mozilla/5.0',
                                                                                   'Content-type': 'application/json'})
resp3 = opener.open(req3)
print 'Oithona labs: ' + str(len(resp3.read()))                                                                     
"""