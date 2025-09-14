import requests 
from tqdm import tqdm
import os

y_true = []
y_pred = []

clean_imgs = os.listdir('.test/clean')[:50]
for img in tqdm(clean_imgs):
    y_true.append(1)
    files = {'file': open(f'.test/clean/{img}', 'rb')}
    response = requests.post('http://localhost:8000/inference/', files=files)
    result = response.json()
    y_pred.append(result['status'])

dirty_damaged_imgs = os.listdir('.test/dirty-damaged')[:50]
for img in tqdm(dirty_damaged_imgs):
    y_true.append(0)
    files = {'file': open(f'.test/dirty-damaged/{img}', 'rb')}
    response = requests.post('http://localhost:8000/inference/', files=files)
    result = response.json()
    y_pred.append(result['status'])

from sklearn.metrics import classification_report, confusion_matrix

clf_report = classification_report(y_true, y_pred, target_names=['dirty/damaged', 'clean'], zero_division=0)

with open('test_report.txt', 'w') as f:
    f.write(clf_report)

conf_matrix = confusion_matrix(y_true, y_pred)

with open('confusion_matrix.txt', 'w') as f:
    f.write(str(conf_matrix))