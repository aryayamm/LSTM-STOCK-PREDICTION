import requests
from config import TOKEN
from config import WHATSAPP_NUMBERS

def send_whatsapp(message):
    url     = "https://api.fonnte.com/send"
    headers = {"Authorization": TOKEN}
    payload = {
        "target"      : ",".join(WHATSAPP_NUMBERS),
        "message"     : message,
        "countryCode" : "62"
    }
    requests.post(url, headers=headers, data=payload)
    print(f"Message sent to {len(WHATSAPP_NUMBERS)} numbers!")