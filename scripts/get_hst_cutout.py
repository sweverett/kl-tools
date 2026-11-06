import sys
import os
import time
import re
import json

import requests
from urllib.parse import quote as urlencode

from astropy.table import Table
import numpy as np

import pprint
pp = pprint.PrettyPrinter(indent=4)


def mast_query(request):
    """Perform a MAST query.
    
        Parameters
        ----------
        request (dictionary): The MAST request json object
        
        Returns head,content where head is the response HTTP headers, and content is the returned data"""
    
    # Base API url
    request_url='https://mast.stsci.edu/api/v0/invoke'    
    
    # Grab Python Version 
    version = ".".join(map(str, sys.version_info[:3]))

    # Create Http Header Variables
    headers = {"Content-type": "application/x-www-form-urlencoded",
               "Accept": "text/plain",
               "User-agent":"python-requests/"+version}

    # Encoding the request as a json string
    req_string = json.dumps(request)
    req_string = urlencode(req_string)
    
    # Perform the HTTP request
    resp = requests.post(request_url, data="request="+req_string, headers=headers)
    
    # Pull out the headers and response content
    head = resp.headers
    content = resp.content.decode('utf-8')

    return head, content

def set_filters(parameters):
    return [{"paramName":p, "values":v} for p,v in parameters.items()]

def set_min_max(min, max):
    return [{'min': min, 'max': max}]

mast_request = {'service':'Mast.Caom.Cone',
                'params':{'ra':150.119166,
                          'dec':2.205833,
                          'radius':0.2},
                'filters'
                'format':'json',
                'pagesize':2000,
                'page':1,
                'removenullcolumns':True,
                'removecache':True}

headers, mast_data_str = mast_query(mast_request)

mast_data = json.loads(mast_data_str)

print(mast_data.keys())
print("Query status:",mast_data['status'])