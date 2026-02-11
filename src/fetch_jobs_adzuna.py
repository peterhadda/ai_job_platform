import os
import json
from typing import Any,Dict,List,Tuple
import requests
import hashlib
import re

def normalize_text(s):
    if not s:
        return ''
    s=s.replace("\r\n","\n").replace("\r","\n")
    s=s.strip()
    s=re.sub(r"\s+"," ",s)
    s=re.sub(r"\n{3,}","\n\n",s)
    return s

def make_job_id(title,company,location,url):
    seed=f"{title.lower().strip()}|{company.lower().strip()}|{location.lower().strip()}|{url.strip()}"
    return hashlib.md5(seed.encode("utf-8")).hexdigest()

