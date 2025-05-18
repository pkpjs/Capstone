const express = require('express')
const app = express()
const port = 4000 // <- 3000에서 다른 숫자로 변경

const cors = require('cors');
const bodyParser = require('body-parser');
const { exec } = require('child_process');

const fs = require('fs');
const path = require('path');

app.use(bodyParser.urlencoded({ extended: false }));
app.use(cors());
app.use(bodyParser.json());

app.get('/', (req, res) => {
  res.send('Hello World!')
})

app.post('/url', (req, res) => {
    //req
    const input_url = req.body.inText;
    console.log("Received URL : ",input_url);
    process.env.input_url = input_url; // 환경변수에 URL 값을 저장
    
    // Docker 컨테이너를 실행
    const command = `sudo docker run projectdiscovery/katana:latest -u "${input_url}"`;
    //const command = 'docker --version';
    //const command = `docker run -d -e URL=${process.env.input_url} your-image-name`;
    exec(command, (error, stdout, stderr) => {
        if (error) {
            console.error(`exec error: ${error}`);
            res.status(500).send({error: "Docker execution failed"});
            return;
        }
        //console.log(stderr);
        console.log(stdout);
        fs.writeFileSync(path.join(__dirname, './katana/url_list.txt'), stdout);
        //console.log(`stdout: ${stdout}`);
        //console.error(`stderr: ${stderr}`);

        // findXSS.js 스크립트 실행
        const findXSSPath = path.join(__dirname, './api/findXSS.js');
        exec(`node "${findXSSPath}"`, (xssError, xssStdout, xssStderr) => {
            if (xssError) {
                console.error(`XSS script error: ${xssError}`);
                return;
            }
            console.log(xssStdout);
        });

    });
    //res
    const sendText = {
        text: 'URL transport success!',
    };
    res.send(sendText);
});

app.listen(port, () => {
  console.log(`Example app listening at http://localhost:${port}`)
})
