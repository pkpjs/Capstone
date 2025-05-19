<?php
session_start();

$response = array();

// 세션이 존재하고, 사용자의 이름이 세션에 저장되어 있는 경우
if (isset($_SESSION['username'])) {
    $response['userName'] = $_SESSION['username'];
} else {
    // 세션에 사용자의 이름이 없는 경우
    $response['userName'] = 'Guest'; // 또는 다른 기본값으로 설정
}

header('Content-Type: application/json');
echo json_encode($response);
?> 