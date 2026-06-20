from flask import Flask, render_template, Response, request, redirect, session, jsonify
from database import db
from models import Admin, Student, ExamLog, Question, SystemSettings, Answer
import cv2
import time
import mediapipe as mp
from ultralytics import YOLO
import os
import threading

app = Flask(__name__, static_folder='statics', static_url_path='/static')
app.secret_key = "supersecretkey"

# ===== DATABASE =====
basedir = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(basedir, 'instance', 'database.db')

app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

# ===== GLOBALS (LAZY LOAD) =====
camera = None
face_mesh = None
model = None

# ===== CAMERA THREAD =====
class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.video.set(3, 320)
        self.video.set(4, 240)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True

        t = threading.Thread(target=self.update, daemon=True)
        t.start()

    def update(self):
        while self.running:
            success, frame = self.video.read()
            if success:
                with self.lock:
                    self.frame = frame
        self.video.release()

    def stop(self):
        self.running = False

    def get_frame(self):
        with self.lock:
            return self.frame

# ===== LOGGING =====
def log_event(message, student_id=None):
    try:
        with app.app_context():
            new_log = ExamLog(event=message, student_id=student_id)
            db.session.add(new_log)
            # Increment student warnings
            student = Student.query.get(student_id)
            if student:
                student.warnings += 1
                if student.warnings >= 3:
                    student.is_flagged = True
            db.session.commit()
    except Exception as e:
        print("DB ERROR:", e)

# ===== FRAME GENERATOR =====
def generate_frames(student_id):
    global camera, face_mesh, model

    no_face_count = 0
    multi_face_count = 0
    
    with app.app_context():
        settings = SystemSettings.query.first()
        max_no_face_thresh = settings.max_no_face if settings else 30
        max_multi_face_thresh = settings.max_multi_face if settings else 30
        max_phone_thresh = settings.max_phone if settings else 1

    last_log_time = 0
    log_cooldown = 3

    frame_count = 0
    last_phone_box = None
    last_phone_time = 0

    while True:
        if camera is None or not camera.running:
            break
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        frame = frame.copy()
        frame_count += 1
        face_count = 0

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        # ===== FACE DETECTION =====
        if results.multi_face_landmarks:
            face_count = len(results.multi_face_landmarks)

            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape

                x_coords = [int(lm.x * w) for lm in face_landmarks.landmark]
                y_coords = [int(lm.y * h) for lm in face_landmarks.landmark]

                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # ===== NOSE TRACKING =====
                nose = face_landmarks.landmark[1]
                nose_x = int(nose.x * w)
                nose_y = int(nose.y * h)

                cv2.circle(frame, (nose_x, nose_y), 5, (255, 0, 0), -1)

                # ===== HEAD DIRECTION =====
                if nose_x < w * 0.3:
                    cv2.putText(frame, "LOOKING LEFT", (30, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                elif nose_x > w * 0.7:
                    cv2.putText(frame, "LOOKING RIGHT", (30, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                elif nose_y > h * 0.7:
                    cv2.putText(frame, "LOOKING DOWN", (30, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # ===== FACE COUNT LOGIC =====
        if face_count == 0:
            no_face_count += 1
            multi_face_count = 0
        elif face_count > 1:
            multi_face_count += 1
            no_face_count = 0
        else:
            no_face_count = 0
            multi_face_count = 0

        current_time = time.time()

        # ===== ALERTS =====
        if no_face_count > max_no_face_thresh:
            if current_time - last_log_time > log_cooldown:
                log_event("NO FACE DETECTED", student_id)
                last_log_time = current_time

            cv2.putText(frame, "NO FACE DETECTED", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        elif multi_face_count > max_multi_face_thresh:
            if current_time - last_log_time > log_cooldown:
                log_event("MULTIPLE FACES DETECTED", student_id)
                last_log_time = current_time

            cv2.putText(frame, "MULTIPLE FACES", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # ===== YOLO (PHONE DETECTION) =====
        if model and frame_count % 15 == 0:
            results_yolo = model(frame)

            for r in results_yolo:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    label = model.names[cls]

                    if label == "cell phone":
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        last_phone_box = (x1, y1, x2, y2)
                        last_phone_time = time.time()

                        if current_time - last_log_time > log_cooldown:
                            log_event("PHONE DETECTED", student_id)
                            last_log_time = current_time

        # ===== DRAW PHONE BOX =====
        if last_phone_box and time.time() - last_phone_time < 2:
            x1, y1, x2, y2 = last_phone_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "PHONE DETECTED", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # ===== STREAM =====
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        time.sleep(0.01)

# ===== ROUTES =====
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    global camera, face_mesh, model

    student_id = session.get('student')

    if camera is None:
        camera = VideoCamera()

    if face_mesh is None:
        face_mesh = mp.solutions.face_mesh.FaceMesh()

    if model is None:
        model = YOLO("yolov8n.pt")

    return Response(generate_frames(student_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ===== AUTH =====
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        admin = Admin.query.filter_by(
            username=request.form['username'],
            password=request.form['password']
        ).first()

        if admin:
            session['admin'] = admin.id
            return redirect('/admin/dashboard')

    return render_template('admin_login.html')

@app.route('/admin/register', methods=['GET', 'POST'])
def admin_register():
    if request.method == 'POST':
        username = request.form['username']
        existing_admin = Admin.query.filter_by(username=username).first()
        if existing_admin:
            return render_template('admin_register.html', error="Username already exists. Please choose a different one.")
            
        new_admin = Admin(
            username=username,
            password=request.form['password']
        )
        db.session.add(new_admin)
        db.session.commit()
        return redirect('/admin/login')
    return render_template('admin_register.html')


@app.route('/start_exam')
def start_exam():
    if 'student' not in session:
        return redirect('/student/login')

    questions = Question.query.all()
    return render_template('exam.html', questions=questions)

@app.route('/api/exam/warning', methods=['POST'])
def exam_warning():
    if 'student' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    student_id = session['student']
    data = request.json
    reason = data.get('reason', 'Unknown Warning')
    
    log_event(reason, student_id)
    
    student = Student.query.get(student_id)
    return jsonify({
        "warnings": student.warnings,
        "auto_submit": student.warnings >= 3
    })

@app.route('/api/exam/status', methods=['GET'])
def exam_status():
    if 'student' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    student_id = session['student']
    student = Student.query.get(student_id)
    if not student:
        return jsonify({"error": "Not found"}), 404
        
    return jsonify({
        "warnings": student.warnings,
        "auto_submit": student.warnings >= 3
    })

@app.route('/api/exam/misconduct', methods=['POST'])
def exam_misconduct():
    if 'student' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    student_id = session['student']
    data = request.json
    reason = data.get('reason', 'MISCONDUCT DETECTED')
    
    # This automatically flags the user
    try:
        with app.app_context():
            new_log = ExamLog(event=reason, student_id=student_id)
            db.session.add(new_log)
            student = Student.query.get(student_id)
            if student:
                student.warnings = 3
                student.is_flagged = True
            db.session.commit()
    except Exception as e:
        print("DB ERROR in Misconduct:", e)
        
    return jsonify({"success": True})

@app.route('/submit_exam', methods=['POST'])
def submit_exam():
    if 'student' not in session:
        return redirect('/student/login')
    
    student_id = session['student']
    questions = Question.query.all()
    
    for q in questions:
        answer_val = request.form.get(f'q_{q.id}')
        if answer_val:
            new_ans = Answer(
                student_id=student_id,
                question_id=q.id,
                answer_text=answer_val
            )
            
            # Auto MCQ Evaluator
            if q.question_type == 'mcq':
                correct_opt = q.correct_option.strip().upper() if q.correct_option else ""
                student_opt = answer_val.strip().upper()
                if student_opt == correct_opt:
                    new_ans.awarded_marks = q.marks
                else:
                    new_ans.awarded_marks = 0

            db.session.add(new_ans)
    
    db.session.commit()
    
    global camera
    if camera is not None:
        camera.stop()
        camera = None

    return redirect('/exam_submitted')

@app.route('/exam_submitted')
def exam_submitted():
    if 'student' not in session:
        return redirect('/student/login')

    return render_template('exam_submitted.html')

@app.route('/admin/dashboard')
def admin_dashboard():
    if 'admin' not in session:
        return redirect('/admin/login')

    logs = ExamLog.query.order_by(ExamLog.timestamp.desc()).all()
    students = Student.query.all()
    questions = Question.query.all()
    settings = SystemSettings.query.first()
    if not settings:
        settings = SystemSettings()
        db.session.add(settings)
        db.session.commit()

    # --- Analytics Logic ---
    total_awarded = 0
    graded_answers_count = 0
    student_scores = {}

    all_answers = Answer.query.filter(Answer.awarded_marks != None).all()
    for ans in all_answers:
        total_awarded += ans.awarded_marks
        graded_answers_count += 1
        if ans.student_id not in student_scores:
            student_scores[ans.student_id] = 0
        student_scores[ans.student_id] += ans.awarded_marks

    avg_score = round(total_awarded / (len(student_scores) if len(student_scores) > 0 else 1), 1)

    # Sort students by highest score
    toppers = []
    for sid, points in sorted(student_scores.items(), key=lambda item: item[1], reverse=True)[:3]:
        stu = Student.query.get(sid)
        if stu:
            toppers.append({'student': stu, 'score': points})

    return render_template('admin_dashboard.html', logs=logs, students=students, questions=questions, settings=settings, avg_score=avg_score, toppers=toppers)

@app.route('/admin/student/<int:student_id>/exam')
def admin_review_exam(student_id):
    if 'admin' not in session:
        return redirect('/admin/login')
    
    student = Student.query.get_or_404(student_id)
    questions = Question.query.all()
    answers = Answer.query.filter_by(student_id=student_id).all()
    
    # Map answers by question_id for easy rendering
    answer_map = {ans.question_id: ans for ans in answers}
    
    return render_template('admin_review_exam.html', student=student, questions=questions, answer_map=answer_map)

@app.route('/admin/student/<int:student_id>/grade_exam', methods=['POST'])
def grade_exam(student_id):
    if 'admin' not in session:
        return redirect('/admin/login')

    answers = Answer.query.filter_by(student_id=student_id).all()
    
    for ans in answers:
        score_val = request.form.get(f'grade_{ans.id}')
        if score_val and score_val.strip() != "":
            try:
                ans.awarded_marks = int(score_val)
            except ValueError:
                pass

    db.session.commit()
    return redirect(f'/admin/student/{student_id}/exam')

@app.route('/admin/settings/update', methods=['POST'])
def update_settings():
    if 'admin' not in session:
        return redirect('/admin/login')
    
    settings = SystemSettings.query.first()
    if settings:
        settings.max_no_face = int(request.form.get('max_no_face', 30))
        settings.max_multi_face = int(request.form.get('max_multi_face', 30))
        settings.passing_marks = int(request.form.get('passing_marks', 0))
        db.session.commit()
    return redirect('/admin/dashboard')

@app.route('/admin/questions/add', methods=['POST'])
def add_question():
    if 'admin' not in session:
        return redirect('/admin/login')
        
    q_type = request.form.get('question_type')
    new_q = Question(
        question_text=request.form.get('question_text'),
        question_type=q_type,
        marks=int(request.form.get('marks', 1))
    )
    if q_type == 'mcq':
        new_q.option_a = request.form.get('option_a')
        new_q.option_b = request.form.get('option_b')
        new_q.option_c = request.form.get('option_c')
        new_q.option_d = request.form.get('option_d')
        new_q.correct_option = request.form.get('correct_option')
        
    db.session.add(new_q)
    db.session.commit()
    return redirect('/admin/dashboard')


@app.route('/student/login', methods=['GET', 'POST'])
def student_login():
    if request.method == 'POST':
        student = Student.query.filter_by(
            username=request.form['username'],
            password=request.form['password']
        ).first()

        if student:
            session['student'] = student.id
            return redirect('/student/dashboard')

        if student:
            print("LOGIN SUCCESS", student)
            session['student'] = student.id
            return redirect('/student/dashboard')
        else:
            print("LOGIN FAILED")

    return render_template('student_login.html')


@app.route('/student/dashboard')
def student_dashboard():
    if 'student' not in session:
        return redirect('/student/login')

    student_id = session['student']
    student = Student.query.get(student_id)
    settings = SystemSettings.query.first()
    passing_mark = settings.passing_marks if settings else 0

    answers = Answer.query.filter_by(student_id=student_id).all()
    has_taken_exam = len(answers) > 0
    
    total_score = sum(ans.awarded_marks for ans in answers if ans.awarded_marks is not None)
    is_graded = any(ans.awarded_marks is not None for ans in answers)

    return render_template('student_dashboard.html', student=student, total_score=total_score, passing_mark=passing_mark, is_graded=is_graded, has_taken_exam=has_taken_exam)

@app.route('/student/register', methods=['GET', 'POST'])
def student_register():
    if request.method == 'POST':
        username = request.form['username']
        existing_student = Student.query.filter_by(username=username).first()
        if existing_student:
            return render_template('student_register.html', error="Username already exists. Please choose a different one.")
            
        new_student = Student(
            username=username,
            password=request.form['password']
        )
        db.session.add(new_student)
        db.session.commit()
        return redirect('/student/login')

    return render_template('student_register.html')

# ===== INIT =====
with app.app_context():
    db.create_all()
    
    # Initialize settings if not exist
    if not SystemSettings.query.first():
        db.session.add(SystemSettings())
        db.session.commit()

# ===== RUN =====
if __name__ == "__main__":
    app.run(debug=True)