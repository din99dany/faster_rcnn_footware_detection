import urllib.request
import os
import ssl
import base64
from PIL import Image
import io

def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

# Read the test image
file = open("<PATH TO IMAGE>", mode='rb')
data = file.read()

# Convert the raw bytes of the image to base64 encoding such that we can payload the image in the body of the request
body = base64.b64encode(data)

# Define api endpoint name and access credentials 
url = '<THE URL OF THE ENDPOINT GOES HERE>'
api_key = '<THE API KEY GOES HERE>'
if not api_key:
    raise Exception("A key should be provided to invoke the endpoint")

# Construct the request
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}
req = urllib.request.Request(url, body, headers)

try:
    # Send the request to the server
    response = urllib.request.urlopen(req)    
    result = response.read()

    # Parse the output of the request and show the image. 
    # The result of the request will be encoded in base64 so make sure to decode that 
    decoded_data = base64.b64decode(result)
    pil_image = Image.open(io.BytesIO(decoded_data))
    pil_image.show()

except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))
    print(error.info())
    print(error.read().decode("utf8", 'ignore'))
