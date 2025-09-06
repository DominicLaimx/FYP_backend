import mysql.connector

conn = mysql.connector.connect(
    host="aiviewmysql.mysql.database.azure.com",
    user="aiview",
    password="#PRASAD SHUBHANGAM RAJESH#123",
    database="ai_interview",
    ssl_disabled=False
)

cursor = conn.cursor()
cursor.execute("SHOW TABLES;")
for table in cursor.fetchall():
    print("✅", table)

cursor.close()
conn.close()
