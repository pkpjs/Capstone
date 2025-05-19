<?php
session_start();

// 데이터베이스 연결 설정
$host = 'localhost';
$db   = 'member';
$user = 'root';
$pass = '1234';
$charset = 'utf8mb4';

$dsn = "mysql:host=$host;dbname=$db;charset=$charset";
$options = [
    PDO::ATTR_ERRMODE            => PDO::ERRMODE_EXCEPTION,
    PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
    PDO::ATTR_EMULATE_PREPARES   => false,
];

try {
    $pdo = new PDO($dsn, $user, $pass, $options);
} catch (\PDOException $e) {
    throw new \PDOException($e->getMessage(), (int)$e->getCode());
}

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $username = $_POST['username'];
    $password = $_POST['password'];

    // 데이터베이스에서 사용자 정보 조회
    $stmt = $pdo->prepare("SELECT id, username, password FROM user WHERE username = ?");
    $stmt->execute([$username]);
    $user = $stmt->fetch();

    if ($user && $password === $user['password']) {
        // 비밀번호가 일치하는 경우 세션 설정
        $_SESSION['loggedin'] = true;
        $_SESSION['username'] = $user['username'];
        $_SESSION['user_id'] = $user['id'];

        // 로그인 성공 후 메인 페이지로 리다이렉션
        header("Location: main.html");
        exit;
    } else {
        // 로그인 실패 시 오류 메시지
        $_SESSION['login_error'] = "잘못된 사용자 이름 또는 비밀번호입니다.";
        echo '<script>alert("잘못된 사용자 이름 또는 비밀번호입니다."); window.location.href = "login.html";</script>';
        exit;
    }
}
?>
