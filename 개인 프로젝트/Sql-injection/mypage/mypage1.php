<?php
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    // 폼에서 전송된 데이터 가져오기
    $name = $_POST["username"];
    $email = $_POST["email"];
    $newPassword = $_POST["password"];

    // 데이터베이스 연결 설정
    $servername = "localhost";
    $username = "bae"; // 실제 데이터베이스 계정 정보를 입력하세요.
    $password = "bae1";
    $dbname = "member";

    $conn = new mysqli($servername, $username, $password, $dbname);

    if ($conn->connect_error) {
        die("Connection failed: " . $conn->connect_error);
    }

    // 데이터베이스에서 해당하는 사용자 찾기
    $sql = "SELECT * FROM user WHERE username='$name' AND email='$email'";
    $result = $conn->query($sql);

    if ($result->num_rows > 0) {
        // 사용자가 존재하면 비밀번호 업데이트
        $updateSql = "UPDATE user SET password='$newPassword' WHERE username='$name' AND email='$email'";
        
        if ($conn->query($updateSql) === TRUE) {
            echo "<script>alert('정보가 성공적으로 수정되었습니다.');</script>";
        } else {
            echo "에러: " . $updateSql . "<br>" . $conn->error;
        }
    } else {
        echo "<script>alert('입력한 username 또는 email이 일치하지 않습니다.');</script>";
    }

    $conn->close();
}
?>