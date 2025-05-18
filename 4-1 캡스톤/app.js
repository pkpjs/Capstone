const express = require('express');
const path = require('path'); 
const morgan = require('morgan');
const cookiParser = require('cookie-parser');
const session = require('express-session');
const dotenv = require('dotenv');
//설치한 미들웨어 및 모듈  불러오기

dotenv.config();
const app = express();
app.set('port',process.env.PORT || 3000);
//app.set('port,포트) : 서버가 실행될 포트 설정

app.use(morgan('dev'));
app.use('/',express.static(path.join(__dirname, 'public')));
app.use(express.json());
app.use(express.urlencoded({ extended: false}));
app.use(cookiParser(process.env.COOKIE_SECRET));
app.use(session({
    resave: false,
    saveUninitialized: false,
    secret: process.env.COOKIE_SECRET,
    cookie: {
        httpOnly: true,
        secure: false,
    },
    name: 'session-cookie',
}));

app.use((req, res, next) => {
    console.log('모든 요청에 다 실행됩니다.');
});

app.get('/',(req,res)=>{
    res.send('Hello, Express');
});
/*app.get(주소, 라우터) : 주소에 대한 GET요청이 올 때 어떤 동작을 할지 적는 부분
ex) app.post, app.patch, app.put, app.delete, app.options
express에서는 http와 다르게 res.write, rew.end 대신 res.send 사용
**/

app.listen(app.get('port'),()=>{
    console.log(app.get('port'),'번 포트에서 대기 중');
});