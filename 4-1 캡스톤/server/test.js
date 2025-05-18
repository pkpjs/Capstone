const http = require("http");
const express = require("express");
const path = require("path");

const app = express();

const port = 5000;

app.get("/ping", (req, res) => {
    res.send("pong");
});


http.createServer(app).listen(port, () => {
  console.log(`app listening at ${port}`);
});
