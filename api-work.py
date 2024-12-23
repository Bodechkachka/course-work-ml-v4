import requests
import json

url = "http://127.0.0.1:5000/api"

image_path = "D:/Machine-learning/course-work-ml/flask_app/static/uploads/ScreenShot140.png"

with open(image_path, "rb") as file:
    files = {"file": file}
    response = requests.post(url, files=files)

if response.status_code == 200:
    try:
        print(response.json())
    except json.decoder.JSONDecodeError:
        print("Ответ не является валидным JSON. Содержимое ответа:")
        print(response.text)
else:
    print(f"Ошибка: {response.status_code}")
    print(response.text)