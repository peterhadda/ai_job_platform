import pandas as pd
class CreateCSV:
    FILENAMES = ["title", "company", "location", "date_posted","description","url","source"]
    def __init__(self,csv,data):
        self.data = data
        self.csv=csv


    def create_csv(self):


