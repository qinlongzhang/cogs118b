import mysql.connector
from py4j.java_gateway import JavaGateway
gateway = JavaGateway()
mydb = mysql.connector.connect(
	host = "127.0.0.1",
	user = "root",
	passwd = "Zql970502",
	auth_plugin='mysql_native_password',
	database = "mydb"
	)
mycursor = mydb.cursor()
INSERT = "INSERT INTO testdb0 (epochIdx, batchIdx, accuracy) VALUES (%s, %s, %s)"
mycursor.execute(INSERT,(0,2,0.02))
mydb.commit()
print(mydb)
print(mycursor)