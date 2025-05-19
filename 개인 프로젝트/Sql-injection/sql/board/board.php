<?php
session_start();

// 어드민 여부를 체크하는 함수
function checkAdmin() {
    // 세션이 설정되어 있고, 어드민으로 마킹된 경우에만 어드민으로 간주
    return isset($_SESSION['username']) && $_SESSION['username'] === 'admin';
}

// 데이터베이스 연결 설정
$servername = "localhost";
$username = "root";
$password = "1234";
$dbname = "member";

$conn = new mysqli($servername, $username, $password, $dbname);

if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// GET 매개변수로부터 페이지 번호 가져오기
$page = isset($_GET['page']) ? $_GET['page'] : 1;
$postsPerPage = 10; // 페이지당 게시글 수

// 전체 게시글 수 가져오기
$totalPostsSql = "SELECT COUNT(*) as total FROM posts";
$totalPostsResult = $conn->query($totalPostsSql);
$totalPosts = $totalPostsResult->fetch_assoc()['total'];

// 전체 페이지 수 계산
$totalPages = ceil($totalPosts / $postsPerPage);

// 현재 페이지에서 가져올 게시글의 시작 인덱스 계산
$startIndex = ($page - 1) * $postsPerPage;

// 데이터베이스에서 페이지에 해당하는 게시글 목록 가져오기
$sql = "SELECT * FROM posts ORDER BY id ASC LIMIT $startIndex, $postsPerPage";
$result = $conn->query($sql);

$data = array();

if ($result->num_rows > 0) {
    while ($row = $result->fetch_assoc()) {
        $data[] = array(
            'id' => $row['id'],
            'title' => $row['title'],
            'content' => nl2br(htmlspecialchars($row['content'])), // 개행을 <br> 태그로 변환하고 HTML 이스케이프
            'file_path' => $row['file_path']
        );
    }
}

// 댓글 삭제 기능 추가
if (isset($_GET['delete']) && checkAdmin()) {
    $deleteId = $_GET['delete'];
    $deleteSql = "DELETE FROM posts WHERE id = $deleteId";
    if ($conn->query($deleteSql) === TRUE) {
        echo json_encode(['success' => true, 'message' => '댓글이 성공적으로 삭제되었습니다.']);
    } else {
        echo json_encode(['success' => false, 'message' => '댓글 삭제에 실패했습니다.']);
    }
} else {
    // JSON 형식으로 출력
    header('Content-Type: application/json');
    echo json_encode(['data' => $data, 'totalPages' => $totalPages, 'isAdmin' => checkAdmin()]);
}

$conn->close();
?>
