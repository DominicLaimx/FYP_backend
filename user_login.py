from flask import Flask, request, jsonify
import mysql.connector
from flask_cors import CORS
import jwt
import datetime
import bcrypt  # For secure password hashing

DB_CONFIG = {
    "host": "aiviewmysql.mysql.database.azure.com",
    "user": "aiview",
    "password": "#PRASAD SHUBHANGAM RAJESH#123",
    "database": "ai_interview",
    "ssl_disabled": False  # 👈 Required for Azure
}

def create_user(name, email, password):
    """Store new user in database with hashed password."""
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())  # Hash password
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)",
            (name, email, hashed_password)
        )
        conn.commit()
        return True
    except mysql.connector.IntegrityError:
        return False  # Email already exists
    finally:
        cursor.close()
        conn.close()
