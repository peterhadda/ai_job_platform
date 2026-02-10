import csv

from config import *




def load_jobs_from_csv(csv_path):
    raw_jobs = []
    if not csv_path.exists():
        return raw_jobs
    with open(csv_path) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            raw_jobs.append(row)
    return raw_jobs

def clean_job_record(raw_job):
    clean_jobs=raw_job
    for jobs in clean_jobs:
        stripped_dict={k.strip(): v.strip() for k, v in jobs.items()}
        if "company" in stripped_dict:
            stripped_dict["company"] = stripped_dict["company"].lower()
        if "location" in stripped_dict:
                stripped_dict["location"] = stripped_dict["location"].lower()


        print(stripped_dict)



data_raw_jobs=load_jobs_from_csv(RAW_JOBS_PATH)
clean=clean_job_record(data_raw_jobs)



