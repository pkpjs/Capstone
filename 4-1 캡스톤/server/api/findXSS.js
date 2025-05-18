//const fs = require('fs');
//const { spawn } = require('child_process');
//const path = require('path');

// 파일에서 URL 목록 읽기
//const urlListPath = path.join(__dirname, '/home/ubuntu/findx/XXX/server/katana/url_list.txt');
//const urls = fs.readFileSync(urlListPath, 'utf8').split('\n').filter(line => line.trim() !== '');
// Dalfox 명령어를 spawn으로 실행
//const dalfox = spawn('dalfox', ['file', '/home/ubuntu/findx/XXX/server/katana/url_list.txt']);

//dalfox.stdout.on('data', (data) => {
//  console.log(`stdout: ${data}`);
//});

//dalfox.stderr.on('data', (data) => {
//  console.error(`stderr: ${data}`);
//});

//dalfox.on('close', (code) => {
//  console.log(`child process exited with code ${code}`);
//});

const fs = require('fs');
const path = require('path');
const { exec } = require('child_process');

// 파일에서 URL 목록 읽기
const urlListPath = path.join(__dirname, '../katana/url_list.txt');
const urls = fs.readFileSync(urlListPath, 'utf8').split('\n').filter(line => line.trim() !== '');

const command = 'dalfox file /home/ubuntu/findx/XXX/server/katana/url_list.txt'
exec(command, (error, stdout, stderr) => {
        console.log(`Results for : `)
        console.log(stdout);
        console.log(stderr);
});
