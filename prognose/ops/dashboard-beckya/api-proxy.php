<?php
declare(strict_types=1);

$requestUri = $_SERVER['REQUEST_URI'] ?? '/';
$targetBase = 'http://127.0.0.1:8000';

if (!str_starts_with($requestUri, '/api/') && $requestUri !== '/health') {
    http_response_code(404);
    header('Content-Type: application/json');
    echo json_encode(['detail' => 'Not found']);
    exit;
}

$targetUrl = $targetBase . $requestUri;
$method = $_SERVER['REQUEST_METHOD'] ?? 'GET';
$body = file_get_contents('php://input');

$headers = [];
foreach ($_SERVER as $key => $value) {
    if (!str_starts_with($key, 'HTTP_')) {
        continue;
    }
    $name = str_replace(' ', '-', ucwords(strtolower(str_replace('_', ' ', substr($key, 5)))));
    if (in_array(strtolower($name), ['host', 'connection', 'content-length'], true)) {
        continue;
    }
    $headers[] = $name . ': ' . $value;
}
if (isset($_SERVER['CONTENT_TYPE'])) {
    $headers[] = 'Content-Type: ' . $_SERVER['CONTENT_TYPE'];
}

$ch = curl_init($targetUrl);
curl_setopt_array($ch, [
    CURLOPT_CUSTOMREQUEST => $method,
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_HEADER => true,
    CURLOPT_FOLLOWLOCATION => false,
    CURLOPT_TIMEOUT => 180,
    CURLOPT_HTTPHEADER => $headers,
]);

if (!in_array($method, ['GET', 'HEAD'], true)) {
    curl_setopt($ch, CURLOPT_POSTFIELDS, $body === false ? '' : $body);
}

$response = curl_exec($ch);
if ($response === false) {
    http_response_code(502);
    header('Content-Type: application/json');
    echo json_encode(['detail' => 'API gateway proxy failed', 'error' => curl_error($ch)]);
    curl_close($ch);
    exit;
}

$status = curl_getinfo($ch, CURLINFO_RESPONSE_CODE) ?: 502;
$headerSize = curl_getinfo($ch, CURLINFO_HEADER_SIZE) ?: 0;
$rawHeaders = substr($response, 0, $headerSize);
$responseBody = substr($response, $headerSize);
curl_close($ch);

http_response_code((int) $status);
foreach (explode("\r\n", $rawHeaders) as $line) {
    if ($line === '' || str_starts_with(strtolower($line), 'http/')) {
        continue;
    }
    $name = strtolower(strtok($line, ':') ?: '');
    if (in_array($name, ['connection', 'transfer-encoding', 'content-length', 'server', 'date'], true)) {
        continue;
    }
    header($line, false);
}

echo $responseBody;
