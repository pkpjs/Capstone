<?php
session_start();

$response = array();

if (isset($_SESSION['user_id'])) {
    // 사용자가 로그인한 경우 세션 파기
    session_destroy();
    $response['success'] = true;
} else {
    // 사용자가 로그인하지 않은 경우
    $response['success'] = false;
}

// 로그인 상태를 확인하는 코드
if (isset($_SESSION['user_id'])) {
    // 사용자가 로그인한 경우
    $response['loggedIn'] = true;
} else {
    // 사용자가 로그인하지 않은 경우
    $response['loggedIn'] = false;
}

header('Content-Type: application/json');
echo json_encode($response);
?>
