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
    $email = $_POST['email'];
    $password = $_POST['password'];

    if ($username == 'admin' || $username == 'root' || $password == 'admin' || $password == 'root')  {
        echo '<script>alert("특정한 아이디 또는 비밀번호는 사용할 수 없습니다."); window.location.href = "join.html";</script>';
        exit;
    }

    // 중복된 아이디를 허용하지 않기 위해 데이터베이스에서 확인
    $stmt = $pdo->prepare("SELECT id FROM user WHERE username = ?");
    $stmt->execute([$username]);
    $existingUser = $stmt->fetch();

    if ($existingUser) {
        echo '<script>alert("이미 존재하는 아이디입니다."); window.location.href = "join.html";</script>';
        exit;
    }

    // 중복된 이메일을 허용하지 않기 위해 데이터베이스에서 확인
    $stmt = $pdo->prepare("SELECT id FROM user WHERE email = ?");
    $stmt->execute([$email]);
    $existingEmail = $stmt->fetch();

    if ($existingEmail) {
        echo '<script>alert("이미 존재하는 이메일입니다."); window.location.href = "join.html";</script>';
        exit;
    }

    $sql = "INSERT INTO user (username, email, password) VALUES (?, ?, ?)";
    
    $stmt = $pdo->prepare($sql);
    $stmt->execute([$username, $email, $password]);

    // 세션 설정
    $_SESSION['loggedin'] = true;
    $_SESSION['username'] = $username;

    // 메인 페이지로 리다이렉션
    header("Location: main.html");
    exit;
}
?>
