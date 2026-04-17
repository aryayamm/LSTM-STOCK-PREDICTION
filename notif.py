import requests
from config import TOKEN, MY_NUMBER

def send_whatsapp(message):
    url = "https://api.fonnte.com/send"
    headers = {"Authorization": TOKEN}
    payload = {"target": MY_NUMBER, "message": message, "countryCode": "62"}
    requests.post(url, headers=headers, data=payload)
    print("Message sent to WhatsApp!")