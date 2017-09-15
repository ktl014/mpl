"""
Downloads unlabeled images from spc.ucsd.edu

Run as:

$ python /path/to/pull_unlabeled_spcdata2.py /path/to/time_period.txt /path/to/desired/output/dir

5/24/17 - ECO
"""

import urllib
import urllib2
import os
import datetime
import pytz
import calendar
import json
import csv
from sys import argv
import numpy as np

# convert datetime
def convertDate(startTime, endTime):
    # convert date time string in Pacific time to unix time stamp.startTime and endTime are strings in 'YYYY-MM-DD HH:MM:SS'
    utcStart = calendar.timegm(pytz.timezone('America/Los_Angeles').localize(datetime.datetime.strptime(startTime, '%Y-%m-%d %H:%M:%S')).utctimetuple())
    utcEnd = calendar.timegm(pytz.timezone('America/Los_Angeles').localize(datetime.datetime.strptime(endTime, '%Y-%m-%d %H:%M:%S')).utctimetuple())
    
    # Add 3600 to for daylight savings time. 
    utcStart = utcStart+3600
    utcEnd = utcEnd+3600
     
    # don't forget to multiply by 1000 to make the utc format into microseconds for approriate http plug in
    utcStart = utcStart*1000
    utcEnd = utcEnd*1000
    return utcStart, utcEnd
    
# the url builder
def urlBuilder(entry):

    startTime = entry.split()[0]+' '+entry.split()[1]
    endTime = entry.split()[2]+' '+entry.split()[3]
    minLen = entry.split()[4]
    maxLen = entry.split()[5]
    cam = entry.split()[6]
    
    camRes = [7.38/1000, 0.738/1000]
    
    # convert minLen and maxLen to pixels from mm
    if cam == 'SPC2':
        minL = np.floor(float(minLen)/camRes[0])
        maxL = np.ceil(float(maxLen)/camRes[0])
    elif cam == 'SPCP2':
        minL = np.floor(float(minLen)/camRes[1])
        maxL = np.ceil(float(maxLen)/camRes[1])

    # convert datetime
    uS, uE = convertDate(startTime, endTime)

    # 0.3 ~ 1 Aspect ratio
    # ..ordered/skip/Any.. Any --> Class of choice
    pattern = "http://spc.ucsd.edu/data/rois/images/SPC2/{!s}/{!s}/0/24/300/{!s}/{!s}/0.3/1/noexclude/ordered/skip/Any/anytype/any/"
    out = pattern.format(uS, uE, int(minL), int(maxL))
    return out

def getImages(urlList, outdir, timestampList):
    # iterate through the list of urls to get the images
    nextPage = 'empty'
    inurl = 'http://spc.ucsd.edu{!s}'
    imgurl = 'http://spc.ucsd.edu{!s}.jpg'
    outdir = os.path.join(outdir, '{!s}.jpg')  #TODO: change by Will
    flag = 1
    out_list = []
    img_countr = 0
    page_countr = 0
    #image_countr = countImages(urlList, inurl, imgurl)
    #print(str(image_countr)+ " total images found")

    # Open file csv file to write image id,url, etc. - Kevin 20170718
    with open("target_image_path_labels.csv", "w") as label_csv:
        img_param = ['timestamp','img_path','img','img_id','img_label']
        labelwriter = csv.DictWriter(label_csv, fieldnames=img_param)
        labelwriter.writeheader()

        for i,item in enumerate(urlList):
            print 'Starting download ' + item
            urlToOpen = item

            while nextPage:
                print urlToOpen
                jsonDoc = json.load(urllib2.urlopen(urlToOpen))
                nextPage = jsonDoc['image_data']['next']

                # update the URL for next iteration (use index 21 to replace the internal server address with url)
                if nextPage:
                    urlToOpen = inurl.format(nextPage[21::])
                else:
                    pass
                img_dicts = jsonDoc['image_data']['results']
                for ii in range(len(img_dicts)):

                    # read in image url
                    img_url = img_dicts[ii]['image_url']
                    img = img_url.split('/')[6]+".jpg"
                    img_id = img_dicts[ii]['id']
                    img_label = img_dicts[ii]['user_labels']

                    # verify image count, url, and id - Kevin 20170718
                    #print("Image " + str(img_countr))
                    #print("     img: " + str(img))
                    #print("     img_id: " + str(img_id))
                    #print("     img_label: " + str (img_label))

                    # write to csv file img url and id - Kevin 20170718
                    labelwriter.writerow({'timestamp':timestampList[i],'img_path': str(img_url),'img':str(img),'img_id':str(img_id),'img_label':str(img_label)})

                    # make the url to download the image
                    inpath = imgurl.format(img_url)

                    # make the outpath
                    outpath = outdir.format(os.path.basename(img_url))

                    # download the image
                    urllib.urlretrieve(inpath,outpath)

                    img_countr += 1
                page_countr += 1
                print(str (img_countr) + " images after page " + str (page_countr))
            nextPage = 'empty'
            print "Done with " + str(flag) + ' of ' + str(len(urlList))
            flag += 1

            print(str(label_csv) + " successfully saved to " + str(os.getcwd()))
    label_csv.close()


# Count images - Kevin 20170718
def countImages(urlList, inurl, imgurl):
    nextPage = 'empty'
    page_countr = 0
    img_countr = 0
    for item in urlList:
        urlToOpen = item
        while nextPage:
            print urlToOpen
            jsonDoc = json.load (urllib2.urlopen (urlToOpen))
            nextPage = jsonDoc['image_data']['next']
            # update the URL for next iteration (use index 21 to replace the internal server address with url)
            if nextPage:
                urlToOpen = inurl.format (nextPage[21::])
            else:
                pass
            img_dicts = jsonDoc['image_data']['results']
            img_countr += len(img_dicts)
            page_countr += 1
            print(str (img_countr) + " images after page " + str (page_countr))
        nextPage = 'empty'
    return img_countr

# read in text file with date and time ranges. Expects text file with each line formatted as:
# startTime endTime minLen maxLen

if __name__ == '__main__':
    
    print argv[1]
    print argv[2]
    with open(argv[1],'rb') as f:
        dtStrings = list(f)

    outpath = argv[2]

    # iterate through and generate list of urls
    timestampList = []
    urlList = []
    for line in dtStrings:
        timestampList.append(line.split()[0]) # Keep timestamp as label for each image
        urlList.append(urlBuilder(line))
    print urlList

    # retrive images in given date ranges
    getImages(urlList, outpath, timestampList)
    

                    
    


