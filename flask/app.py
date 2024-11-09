from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import sqlite3
import os
from dotenv import load_dotenv
import getpass

from model import allam


    
# إنشاء التطبيق
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# إعداد قاعدة البيانات
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL UNIQUE, password TEXT NOT NULL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, feedback TEXT, helpful INTEGER, FOREIGN KEY(user_id) REFERENCES users(id))''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def home():
    if 'user_id' in session:
        return render_template('index.html')
    return redirect(url_for('login'))


@app.route('/generate', methods=['POST'])
def generate():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
        
    user_message = request.json['input']
    chat_history = request.json.get('history', [])
    
    try:
        messages = [
            {"role": "system", "content": """You are a helpful assistant. Please provide complete, 
             detailed responses. Always finish your thoughts and never cut off mid-sentence. 
             If you're making a list or explaining something, make sure to complete all points."""}
        ]
        
        # Limit chat history to last 5 exchanges to prevent context overflow
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]
            
        messages.extend(chat_history)
        messages.append({"role": "user", "content": user_message})
        
        response = allam.invoke(messages)
        
        if not response or not response.content:
            return jsonify({"error": "Empty response from model"}), 500
            
        # Verify response isn't truncated
        response_content = response.content.strip()
        if not response_content.endswith(('.', '!', '?', '"', '؟', '.')):
            response_content += '.'
            
        return jsonify({
            'output': response_content,
            'history': messages + [{"role": "assistant", "content": response_content}]
        })
        
    except Exception as e:
        print(f"Error in generate route: {str(e)}")
        return jsonify({"error": str(e)}), 500
    


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return 'اسم المستخدم موجود بالفعل', 400
        finally:
            conn.close()

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
        user = c.fetchone()
        conn.close()

        if user:
            session['user_id'] = user[0]
            return redirect(url_for('home'))
        else:
            return 'اسم المستخدم أو كلمة المرور غير صحيحة', 400

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

@app.route('/feedback', methods=['POST'])
def feedback():
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "الرجاء تسجيل الدخول"}), 403

    feedback_data = request.json
    feedback_text = feedback_data['feedback']
    is_helpful = feedback_data['helpful']

    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("INSERT INTO feedback (user_id, feedback, helpful) VALUES (?, ?, ?)", 
              (session['user_id'], feedback_text, int(is_helpful)))
    conn.commit()
    conn.close()

    return jsonify({"success": True, "message": "تم استلام التغذية الراجعة"})

if __name__ == '__main__':
    app.run(debug=True)
