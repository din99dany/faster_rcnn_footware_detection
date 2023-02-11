# Content explained

In this directory are the files you need in order to create a front end for the api provided in `api_endpoint`. How it works is that the file `index.js` is the definition of an Azure Function that makes calls to the api endpoints. The result of the request are embedded into the `.html` file `index.html` and then returned to the user. `index.html` is your landing page.

# How to use
In order to have a front end, please deploy the 2 files to azure functions. Also, please make sure to use your actual endpoint url and credentials from when you deployed the model api. Search for `<API KEY GOES HERE>` and `<ENDPOINT URL GOES HERE>` and update the values