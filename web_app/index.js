const http = require("http");
const fs = require("fs");
const axios = require("axios");

module.exports = async function (context, req) {
  context.log("JavaScript HTTP trigger function processed a request.");
  switch (req.method) {
    case "GET":
      return new Promise((resolve, reject) => {
        fs.readFile(__dirname + "/index.html", function (err, data) {
          if (err) {
            reject({
              status: 500,
              body: "An error occurred while reading the HTML file.",
            });
          } else {
            resolve({
              status: 200,
              headers: {
                "Content-Type": "text/html",
              },
              body: data.toString(),
            });
          }
        });
      })
        .then((result) => {
          context.res = result;
          context.done();
        })
        .catch((error) => {
          context.res = error;
          context.done();
        });
      break;
    case "POST":
      if (req.body) {
        const api_key = "83PSXgcDzoi6tqV4XISRawJijk9lz9cu";
        const base64Img = req.body.image.split(",")[1];
        const headers = {
          Authorization: `Bearer ${api_key}`,
          "Content-Type": "application/json",
        };

        try {
          const response = await axios({
            method: "post",
            url: "https://zxc.westeurope.inference.ml.azure.com/score",
            headers: headers,
            data: base64Img,
          });

          context.res = {
            status: 200,
            body: response.data,
          };
        } catch (error) {
          context.res = {
            status: 500,
            body: error,
          };
        }
      } else {
        context.res = {
          status: 400,
          body: "Please pass a string in the request body",
        };
      }

      break;
    default:
      context.res = {
        status: 405,
        body: "Method not allowed.",
      };
      context.done();
      break;
  }
};
