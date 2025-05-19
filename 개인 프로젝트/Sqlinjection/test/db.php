<!-- <?php
$servername = "localhost";
$username = "root";
$password = "1234";
$database = "member";
$conn = new mysqli($servername, $username, $password, $database);

if ($conn->connect_error) {
    die("DB 연결 실패: " . $conn->connect_error);
}

$sql = "SHOW TABLES";
$result = $conn->query($sql);
$tables = [];

if ($result->num_rows > 0) {
    while ($row = $result->fetch_assoc()) {
        $tables[] = $row["Tables_in_member"];
    }
}

echo json_encode($tables);

$conn->close();
?> -->
