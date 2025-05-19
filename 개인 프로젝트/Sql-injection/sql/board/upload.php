<?php
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
    die("DB 연결 실패: " . $e->getMessage());
}

// 업로드 폴더 지정
$uploadDir = "/var/www/html/test1/sql/board/uploads";


// 폴더가 없으면 생성
if (!is_dir($uploadDir)) {
    mkdir($uploadDir, 0755, true);
}

if ($_SERVER["REQUEST_METHOD"] == "POST" && isset($_FILES["file"])) {
    $title = $_POST["title"] ?? "";
    $content = $_POST["content"] ?? "";
    
    $file = $_FILES["file"];
    $originalFileName = $file["name"];
    $fileExt = pathinfo($originalFileName, PATHINFO_EXTENSION);
    $newFileName = time() . "_" . rand(1000, 9999) . "." . $fileExt;
    $targetPath = $uploadDir . $newFileName;

    // 파일 업로드 오류 코드 확인
    if ($file["error"] !== UPLOAD_ERR_OK) {
        die("파일 업로드 오류: " . $file["error"]);
    }

    // 업로드 성공 여부 확인
    if (move_uploaded_file($file["tmp_name"], $targetPath)) {
        // 데이터베이스에 글과 파일 정보 저장
        $sql = "INSERT INTO posts (title, content, file_path) VALUES (:title, :content, :file_path)";
        $stmt = $pdo->prepare($sql);
        $stmt->execute([
            ':title' => $title,
            ':content' => $content,
            ':file_path' => $targetPath
        ]);

        echo "<script>alert('게시글과 파일 업로드 성공!'); window.location.href = 'board.html';</script>";
    } else {
        die("파일 이동 실패: " . error_get_last()['message']);
    }
} else {
    die("잘못된 접근입니다.");
}
?>
