import keras.callbacks
import mysql.connector
class saveToDB(keras.callbacks.Callback):
	def __init__(self,dbInfo):
		self.epochIdx = 0
		if len(dbInfo)!= 0 and dbInfo:
			if dbInfo.has_key("database"):
				self.mydb = mysql.connector.connect(
								host = dbInfo["host"],
								user = dbInfo["user"],
								passwd = dbInfo["passwd"],
								auth_plugin='mysql_native_password',
								database = dbInfo["database"])
			else:
				self.mydb = mysql.connector.connect(
								host = dbInfo["host"],
								user = dbInfo["user"],
								passwd = dbInfo["passwd"],
								auth_plugin='mysql_native_password')
				mycursor = self.mydb.cursor()
				mycursor.execute("CREATE DATABASE modelDataBase")
				self.mydb = mysql.connector.connect(
								host = dbInfo["host"],
								user = dbInfo["user"],
								passwd = dbInfo["passwd"],
								auth_plugin='mysql_native_password',
								database = "modelDataBase")

			


			mycursor = self.mydb.cursor()
			mycursor.execute("CREATE TABLE modelBatch (id INT AUTO_INCREMENT PRIMARY KEY, epochIdx VARCHAR(255), batchIdx VARCHAR(255), accuracy VARCHAR(255), loss VARCHAR(255)")
			mycursor.execute("CREATE TABLE modelEpoch (id INT AUTO_INCREMENT PRIMARY KEY, epochIdx VARCHAR(255), accuracy VARCHAR(255),loss VARCHAR(255))")

			def on_train_begin(self,logs={}):


			def on_batch_end(self,batch,logs={}):
				sql = "INSERT INTO modelBatch(epochIdx,batchIdx,accuracy,loss) VALUES(%s,%s,%s,%s)"
				val = (self.epochIdx+1,batch,logs.get('dice_coefficient'),logs.get('loss'))
				mycursor = self.mydb.cursor()
				mycursor.execute(sql,val)

			def on_epoch_end(self,epoch,logs={}):
				sql = "INSERT INTO modelEpoch(epochIdx,accuracy,loss) VALUES (%s,%s,%s)"
				val = (epoch,logs.get('val_dice_coefficient'),logs.get('val_loss'))
				mycursor = self.mydb.cursor()
				mycursor.execute(sql,val)






			def on_epoch_end(self,epoch,logs={}):
				self.epochIdx = epoch 








	def clearDBALL(epoch,batch):










