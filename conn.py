import gspread
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv
import os

def connect2Googlesheet():
    load_dotenv()
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path is None:
        raise ValueError("The environment variable GOOGLE_SHEET_CREDS is not set.")
    creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
    client = gspread.authorize(creds)
    sheet_url = 'https://docs.google.com/spreadsheets/d/1GRStLZdPvZTN-DDcxND3xaYaHGGl-wVCQxWdsFRhsqs/edit?gid=408470182#gid=408470182'
    # Open the Google Sheet (replace with your sheet name or URL)
    spreadsheet = client.open_by_url(sheet_url)  # Or use .open_by_url('URL')
    return spreadsheet

# Function to get ANNOTATED relevant documents for a specific question
def get_relevant_documents(df, question):
    relevant_docs = df[df[question] == 1]['Document'].tolist()
    return set(relevant_docs)