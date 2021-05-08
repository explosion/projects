import requests

response = requests.get('http://localhost:5000/predict?sentence="I am very very happy to see you!"')
print('response: {}'.format(response))
print('response.text: {}'.format(response.text))