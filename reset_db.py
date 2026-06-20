from app import app, db
from models import Student, Admin, ExamLog, Answer, Question, SystemSettings
import os

def clear_db(full_reset=True):
    with app.app_context():
        if full_reset:
            print("Performing Full Reset...")
            db.drop_all()
            db.create_all()
            # Re-initialize system settings
            if not SystemSettings.query.first():
                db.session.add(SystemSettings())
                db.session.commit()
            print("Database completely reset. All users, logs, answers, and questions have been removed.")
        else:
            print("Performing Partial Reset (keeping Questions and Settings)...")
            db.session.query(Student).delete()
            db.session.query(Admin).delete()
            db.session.query(ExamLog).delete()
            db.session.query(Answer).delete()
            db.session.commit()
            print("User-related data cleared. Questions and settings were preserved.")

if __name__ == "__main__":
    # By default, we do a full reset as requested to "clear the db"
    clear_db(full_reset=True)
