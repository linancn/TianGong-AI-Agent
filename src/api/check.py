
import requests


def checkApiLoginCode(userId, apiLoginCode):
    url = 'http://localhost:8081/api/user/checkApiLoginCode/'+str(userId)+'/'+apiLoginCode
    headers = {
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0'
    }
    params = {}
    res = requests.get(url=url, headers=headers, params=params)
    return(res.text)