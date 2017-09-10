<?php
// define variables and set to empty values
$domain = $text = "";

if ($_SERVER["REQUEST_METHOD"] == "POST") {
	$domain =$_POST["domain"];
	$text =$_POST["text"];
	$url = 'http://34.228.207.174:8085/textapi/';
	$data = array('domain' => $domain, 'text' => $text);

	// use key 'http' even if you send the request to https://...
	$options = array(
	    'http' => array(
	        // 'header'  => "Content-type: application/x-www-form-urlencoded\r\n",
	        'method'  => 'POST',
	        'content' => $data
	    )
	);
	$context  = stream_context_create($options);
	$result = file_get_contents($url, false, $context);
	if ($result === FALSE) { echo $result; }

	var_dump($result);
	echo json_encode($result);
	echo $result;
}
else {
echo "Else Part";
}