<html>
 <head>
 	<title>SQL injection</title>
 	<style>
 		body{
 		}
 		.user {
 			background-color: yellow;
 		}
 	</style>
 </head>
 
 <body>
 	<h1>PDO vulnerable a SQL injection</h1>
 
 	<?php
 		// sql injection possible:
 		// coses'); drop table test;'select 
		if( isset($_POST["user"])) {

			$dbhost = $_ENV["localhost"];
			$dbname = $_ENV["member"];
			$dbuser = $_ENV["root"];
			$dbpass = $_ENV["1234"];

			# Connectem a MySQL (host,usuari,contrassenya)
			$pdo = new PDO("mysql:host=$dbhost;dbname=$dbname",$dbuser,$dbpass);
	 
			$username = $_POST["user"];
			$pass = $_POST["password"];
			# (2.1) creem el string de la consulta (query)
			$qstr = "SELECT * FROM users WHERE name='$username' AND password=SHA2('$pass',512);";
			$consulta = $pdo->prepare($qstr);

			# mostrem la SQL query per veure el què s'executarà (a mode debug)
			echo "<br>$qstr<br>";

			# Enviem la query al SGBD per obtenir el resultat
			$consulta->execute();
	 
			# Gestió d'errors
			if( $consulta->errorInfo()[1] ) {
				echo "<p>ERROR: ".$consulta->errorInfo()[2]."</p>\n";
				die;
			}

			if( $consulta->rowCount() >= 1 )
				# hi ha 1 resultat o més d'usuaris amb nom i password
				foreach( $consulta as $user ) {
					echo "<div class='user'>Hola ".$user["name"]." (".$user["role"].").</div>";
				}
			else
				echo "<div class='user'>No hi ha cap usuari amb aquest nom o contrasenya.</div>";
		}
 	?>
 	
 	<fieldset>
 	<legend>Login form</legend>
  	<form method="post">
		User: <input type="text" name="user" /><br>
		Pass: <input type="text" name="password" /><br>
		<input type="submit" /><br>
 	</form>
  	</fieldset>
	
 </body>
 
 </html>