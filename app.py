# app.py

import os
from flask import Flask, render_template, request, redirect, url_for, session, flash
from dotenv import load_dotenv
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for servers
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from models import db, User # Assuming 'models.py' exists and defines 'db' and 'User'

import matplotlib.pyplot as plt
# CORRECT: Import the actual run_analysis from your utils/trading_bot.py
from utils.trading_bot import run_analysis 

# REMOVED: The dummy run_analysis function that was here.
# It should now ONLY be defined in utils/trading_bot.py


# Load environment variables
load_dotenv(".env")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "defaultsecret")
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'  # Use SQLite database
db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'  # Redirect unauthenticated users to the login page

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

ADMIN_USER = os.getenv("ADMIN_USER")
ADMIN_PWD = os.getenv("ADMIN_PWD")

# Global variables
profit_log = []
latest_insight = "" # This will now be populated by the real run_analysis

def generate_dummy_profit():
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    profit = round(50 + (len(profit_log) * 12.5), 2)
    profit_log.append((now, profit))
    return profit

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        admin_user = os.getenv("ADMIN_USER")
        admin_pwd = os.getenv("ADMIN_PWD")

        if username == admin_user and password == admin_pwd:
            # Check if admin user exists in the database
            user = User.query.filter_by(username=admin_user).first()
            if not user:
                # Create admin user if it doesn't exist
                user = User(username=admin_user, password=admin_pwd, email="admin@example.com")
                db.session.add(user)
                db.session.commit()
            login_user(user)
            return redirect(url_for("dashboard"))

        user = User.query.filter_by(username=username).first()
        if user and user.password == password: # In a real app, you'd hash passwords
            login_user(user)
            return redirect(url_for("dashboard"))

        return render_template("login.html", error="❌ Invalid credentials.")
    return render_template("login.html")

@app.route("/dashboard")
@login_required
def dashboard():
    global latest_insight
    generate_dummy_profit()

    # Create profit chart
    if profit_log:
        times, profits = zip(*profit_log)
        plt.figure(figsize=(10, 4))
        plt.plot(times, profits, marker="o", color="lime")
        plt.title("📈 Profit Over Time")
        plt.xlabel("Time")
        plt.ylabel("Profit ($)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        os.makedirs("static", exist_ok=True)
        chart_path = os.path.join("static", "profit_chart.png")
        plt.savefig(chart_path)
        plt.close()
    else:
        chart_path = None

    admin_user = os.getenv("ADMIN_USER")
    users = User.query.all() if current_user.username == admin_user else None
    return render_template("dashboard.html", chart_path=chart_path, ai_insight=latest_insight, admin_user=admin_user, users=users)

@app.route("/admin/add_user", methods=["POST"])
@login_required
def add_user():
    # In a real app, you'd also check if the current_user is an admin based on a role in the DB
    if current_user.username != os.getenv("ADMIN_USER"):
        flash("You are not authorized to perform this action.", "error")
        return redirect(url_for("dashboard"))

    username = request.form.get("username")
    password = request.form.get("password") # In a real app, hash this password!
    email = request.form.get("email")

    existing_user = User.query.filter_by(username=username).first()
    if existing_user:
        flash("Username already exists. Please choose a different one.", "error")
        return redirect(url_for("dashboard"))

    new_user = User(username=username, password=password, email=email)
    db.session.add(new_user)
    db.session.commit()
    flash("User added successfully.", "success")
    return redirect(url_for("dashboard"))

@app.route("/admin/remove_user", methods=["POST"])
@login_required
def remove_user():
    # In a real app, you'd also check if the current_user is an admin based on a role in the DB
    if current_user.username != os.getenv("ADMIN_USER"):
        flash("You are not authorized to perform this action.", "error")
        return redirect(url_for("dashboard"))

    username = request.form.get("username")
    user = User.query.filter_by(username=username).first()
    if user:
        db.session.delete(user)
        db.session.commit()
        flash("User removed successfully.", "success")
    else:
        flash("User not found.", "error")

    return redirect(url_for("dashboard"))

@app.route("/start", methods=["POST"])
@login_required
def start_bot():
    global latest_insight
    try:
        # This will now call the run_analysis from utils/trading_bot.py
        insight = run_analysis() 
        latest_insight = "🧠 AI Trading Insight:\n\n" + insight
    except Exception as e:
        latest_insight = f"❌ Bot failed: {str(e)}"
        print(f"Error running bot: {e}")  # Log the error for further investigation

    return redirect(url_for("dashboard"))

@app.route("/refresh_data", methods=["GET"])
@login_required
def refresh_data():
    # This route can be used to trigger a refresh of the dashboard data
    # without running the bot, useful for the "Refresh Data" button
    generate_dummy_profit() # Re-generate profit data
    # No need to re-run AI analysis here unless specifically desired
    return redirect(url_for("dashboard"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password") # In a real app, hash this password!
        email = request.form.get("email")

        # Prevent admin from being registered via this public route
        if username == os.getenv("ADMIN_USER"):
            flash("Cannot register as admin via this route.", "error")
            return render_template("register.html", error="Cannot register as admin via this route.")

        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return render_template("register.html", error="Username already exists. Please choose a different one.")

        new_user = User(username=username, password=password, email=email)
        db.session.add(new_user)
        db.session.commit()
        flash("Registration successful! Please log in.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))

with app.app_context():
    db.create_all() # This creates tables based on your models.py

if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)