from database import db
import datetime

class Admin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(50))

class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(50))
    is_flagged = db.Column(db.Boolean, default=False)
    warnings = db.Column(db.Integer, default=0)

class ExamLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    event = db.Column(db.String(200))
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    student_id = db.Column(db.Integer)

class Question(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question_text = db.Column(db.Text, nullable=False)
    question_type = db.Column(db.String(20), nullable=False) # 'mcq' or 'brief'
    option_a = db.Column(db.String(200), nullable=True)
    option_b = db.Column(db.String(200), nullable=True)
    option_c = db.Column(db.String(200), nullable=True)
    option_d = db.Column(db.String(200), nullable=True)
    correct_option = db.Column(db.String(10), nullable=True)
    marks = db.Column(db.Integer, default=1)

class SystemSettings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    max_no_face = db.Column(db.Integer, default=30) # Frames/ticks before flag
    max_multi_face = db.Column(db.Integer, default=30)
    max_phone = db.Column(db.Integer, default=1)
    passing_marks = db.Column(db.Integer, default=0)

class Answer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer)
    question_id = db.Column(db.Integer)
    answer_text = db.Column(db.Text)
    awarded_marks = db.Column(db.Integer, default=None)