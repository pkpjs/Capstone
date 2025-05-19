<?php
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $title = $_POST["title"];
    $content = $_POST["content"];

    // 데이터베이스 연결 설정
    $servername = "localhost";
    $username = "bae"; // 여기에 실제 데이터베이스 계정 정보를 입력하세요.
    $password = "bae1";
    $dbname = "member";

    $conn = new mysqli($servername, $username, $password, $dbname);

    if ($conn->connect_error) {
        die("Connection failed: " . $conn->connect_error);
    }

    // 데이터베이스에 글 저장
    $sql = "INSERT INTO notice (title, content) VALUES ('$title', '$content')";
    
    if ($conn->query($sql) === TRUE) {
        echo json_encode(["success" => true]);
    } else {
        echo json_encode(["success" => false, "message" => "Error: " . $sql . "<br>" . $conn->error]);
    }

    $conn->close();
} else {
    // POST로 요청이 아닌 경우 에러 메시지를 반환합니다.
    echo json_encode(["success" => false, "message" => "Invalid request"]);
}
?>