import React, { useState } from 'react';

function UrlInput() {  // 컴포넌트 이름을 PascalCase로 변경
    const [state, setState] = useState({
        text: ''
    });

    const handleChange = (e) => {
        setState({
            text: e.target.value
        });
    };

    const onClick = () => {
        const textbox = { inText: state.text };

        fetch("http://3.39.143.22:4000/url", {
            method: "POST",  // 메소드 이름을 대문자로 통일
            headers: {
                "Content-Type": "application/json",  // 일반적으로 헤더 키도 대문자로 시작
            },
            body: JSON.stringify(textbox)
        })
        .then((res) => res.json())  // 이 부분을 한 줄로 간소화
        .then((json) => {
            console.log(json);
            setState({
                text: json.text,  // 응답으로 받은 데이터를 state에 반영
            });
        });
    };

    return (
        <div>
            <input type="text" onChange={handleChange} />
            <button onClick={onClick}>전송</button>
            <h3>{state.text}</h3>
        </div>
    );
}

export default UrlInput;
