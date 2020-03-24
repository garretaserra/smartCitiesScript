import requests
import urllib.request

for breed in ['terrier/yorkshire', 'beagle']:
    r = requests.get('https://dog.ceo/api/breed/' + breed + '/images')
    r = r.json()
    r = r['message']
    i = 0
    filename = ''
    if breed == 'terrier/yorkshire':
        filename = 'yorkshires'
    elif breed == 'beagle':
        filename = 'beagles'
    directory = './images/dogs/' + filename + '/image'
    for url in r:
        urllib.request.urlretrieve(url,  directory + str(i).zfill(4) + '.jpg')
        i += 1
        print(url, './images/dogs/yorkshires/image' + str(i).zfill(4) + '.jpg')
