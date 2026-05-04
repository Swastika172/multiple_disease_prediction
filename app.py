import os, json, hashlib, datetime
import numpy as np
import pickle
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Health Assistant", layout="wide", page_icon="🧑‍⚕️")

# ── PATHS ──────────────────────────────────────────────────────────────────────
WD        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(WD, 'data')
USERS_F   = os.path.join(DATA_DIR, 'users.json')
os.makedirs(DATA_DIR, exist_ok=True)

# ── LOAD MODELS ────────────────────────────────────────────────────────────────
diabetes_model      = pickle.load(open(f'{WD}/saved_models/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open(f'{WD}/saved_models/heart_disease_model.sav', 'rb'))
parkinsons_model    = pickle.load(open(f'{WD}/saved_models/parkinsons_model.sav', 'rb'))

# ══════════════════════════════════════════════════════════════════════════════
#  DATA HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _default_data():
    return {"users": {"admin": {
        "name": "Admin", "role": "admin",
        "password": hashlib.sha256("admin123".encode()).hexdigest(),
        "created_at": str(datetime.date.today()), "predictions": []
    }}}

def load_data():
    if os.path.exists(USERS_F):
        with open(USERS_F) as f:
            return json.load(f)
    d = _default_data(); save_data(d); return d

def save_data(d):
    with open(USERS_F, 'w') as f:
        json.dump(d, f, indent=2)

def hash_pw(p):
    return hashlib.sha256(p.encode()).hexdigest()

# ══════════════════════════════════════════════════════════════════════════════
#  AUTH
# ══════════════════════════════════════════════════════════════════════════════
def do_login(user, pw):
    d = load_data(); u = d["users"].get(user)
    return u if (u and u["password"] == hash_pw(pw)) else None

def do_register(user, name, pw):
    d = load_data()
    if user in d["users"]:
        return False, "Username already taken."
    d["users"][user] = {
        "name": name, "role": "user",
        "password": hash_pw(pw),
        "created_at": str(datetime.date.today()),
        "predictions": []
    }
    save_data(d); return True, "Account created! Please log in."

def record_prediction(username, disease, result, prob):
    d = load_data()
    if username in d["users"]:
        d["users"][username]["predictions"].append({
            "ts": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "disease": disease, "result": result, "probability": prob
        })
        save_data(d)

# ══════════════════════════════════════════════════════════════════════════════
#  ML HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def get_probability(model, x):
    try:
        return round(float(model.predict_proba([x])[0][1]) * 100, 1)
    except Exception:
        score = model.decision_function([x])[0]
        return round(float(1 / (1 + np.exp(-score))) * 100, 1)

RISK_LEVELS = [
    (25,  "Low Risk",       "#10b981", "🟢"),
    (50,  "Moderate Risk",  "#f59e0b", "🟡"),
    (75,  "High Risk",      "#f97316", "🟠"),
    (101, "Very High Risk", "#ef4444", "🔴"),
]

def get_risk(p):
    for thr, lbl, col, ico in RISK_LEVELS:
        if p < thr:
            return lbl, col, ico
    return RISK_LEVELS[-1][1], RISK_LEVELS[-1][2], RISK_LEVELS[-1][3]

# ══════════════════════════════════════════════════════════════════════════════
#  SUGGESTIONS DATABASE
# ══════════════════════════════════════════════════════════════════════════════
SUBS = {
    "Diabetes": {
        "low": {
            "🥗 Diet":      ["Balanced diet: vegetables, lean proteins, whole grains",
                             "Limit sugary drinks and processed snacks",
                             "Avoid skipping meals — eat 3 regular meals"],
            "🏃 Exercise":  ["30 mins moderate activity 3×/week",
                             "Walking, cycling, or swimming are ideal",
                             "Take stairs instead of elevator daily"],
            "💊 Medical":   ["Annual fasting blood sugar checkup",
                             "Know your family history of diabetes",
                             "Monitor BMI — stay in healthy range (18.5–24.9)"],
            "🧘 Lifestyle": ["Maintain healthy weight",
                             "Stay hydrated — 8 glasses of water/day",
                             "Avoid smoking and limit alcohol"],
        },
        "moderate": {
            "🥗 Diet":      ["Follow a low-glycemic index (GI) diet",
                             "Cut refined carbs: white bread, pasta, sugary cereals",
                             "Increase dietary fiber — oats, beans, lentils",
                             "Eat smaller, more frequent meals (5–6/day)"],
            "🏃 Exercise":  ["Daily 45-min brisk walk minimum",
                             "Resistance/strength training 2×/week",
                             "Target 150 mins/week total physical activity"],
            "💊 Medical":   ["Blood sugar check every 3–6 months",
                             "Monitor HbA1c — keep below 5.7%",
                             "Discuss prediabetes prevention with your doctor",
                             "Consider metformin if recommended"],
            "🧘 Lifestyle": ["Practice yoga or meditation for stress control",
                             "Get 7–8 hrs quality sleep every night",
                             "Quit smoking — raises diabetes risk significantly"],
        },
        "high": {
            "🥗 Diet":      ["Strict diabetic diet — count every carbohydrate",
                             "Avoid all processed, fried, and fast foods",
                             "Consult a registered dietician for a meal plan",
                             "Strict portion control at every meal"],
            "🏃 Exercise":  ["Supervised structured daily exercise program",
                             "150+ mins/week combining cardio and strength",
                             "Post-meal walks (15–20 mins) to lower glucose spikes"],
            "💊 Medical":   ["Consult an endocrinologist urgently",
                             "Oral glucose tolerance test (OGTT) recommended",
                             "Medication (metformin) very likely to be prescribed",
                             "Monitor fasting glucose and HbA1c every 3 months"],
            "🧘 Lifestyle": ["Quit smoking and eliminate alcohol completely",
                             "Weekly weight monitoring — log results",
                             "Reduce stress aggressively — it raises blood sugar",
                             "Join a diabetes education program"],
        },
        "very_high": {
            "🥗 Diet":      ["Medical nutrition therapy with a certified dietician",
                             "Very low-carb or ketogenic diet may be required",
                             "Track every meal — use a food diary or app",
                             "Zero sugary foods, drinks, or desserts"],
            "🏃 Exercise":  ["Medically supervised exercise program only",
                             "Start very slowly — even 10-min walks after meals help",
                             "Physical therapy may be needed if complications exist"],
            "💊 Medical":   ["IMMEDIATE comprehensive diabetes evaluation",
                             "Multiple daily blood glucose checks (before/after meals)",
                             "Insulin therapy may be required right away",
                             "Screen for complications: eyes, kidneys, nerves, feet"],
            "🧘 Lifestyle": ["Daily medical visits or telehealth monitoring",
                             "Emergency plan ready — know hypoglycemia symptoms",
                             "Join a diabetes support group",
                             "Track all symptoms and report changes immediately"],
        },
    },
    "Heart": {
        "low": {
            "🥗 Diet":      ["Mediterranean diet: fruits, vegetables, fish, olive oil",
                             "Reduce saturated fats (red meat, full-fat dairy)",
                             "Eliminate trans fats (packaged/fried foods)"],
            "🏃 Exercise":  ["150 mins/week moderate-intensity cardio",
                             "Walking, cycling, or swimming — pick what you enjoy",
                             "Include flexibility and balance exercises weekly"],
            "💊 Medical":   ["Annual cardiovascular checkup",
                             "Know your numbers: BP, LDL, HDL, triglycerides",
                             "Blood pressure: aim for under 120/80 mmHg"],
            "🧘 Lifestyle": ["Avoid smoking and secondhand smoke",
                             "Limit alcohol to 1 drink/day (women), 2 (men)",
                             "Manage stress — it directly impacts heart health"],
        },
        "moderate": {
            "🥗 Diet":      ["DASH diet to lower blood pressure",
                             "Reduce sodium intake to under 2,300 mg/day",
                             "Limit red meat — max 2 servings/week",
                             "Add potassium-rich foods: bananas, sweet potato, spinach"],
            "🏃 Exercise":  ["Daily 30-min moderate cardio",
                             "Always warm up 5–10 mins before exercising",
                             "Add yoga or stretching to reduce cardiac stress"],
            "💊 Medical":   ["Cholesterol panel every 6 months",
                             "Daily home blood pressure monitoring",
                             "Annual resting ECG",
                             "Discuss low-dose aspirin therapy with your doctor"],
            "🧘 Lifestyle": ["Sleep 7–8 hours — poor sleep raises heart risk",
                             "Stress management is critical — try journaling",
                             "Build a strong social support system"],
        },
        "high": {
            "🥗 Diet":      ["Strict heart-healthy diet under cardiologist guidance",
                             "Completely eliminate trans fats and processed meats",
                             "No added salt — season with herbs/spices instead",
                             "Very limited cholesterol-rich foods"],
            "🏃 Exercise":  ["Medically supervised cardiac rehabilitation program",
                             "Only physician-approved physical activities",
                             "Monitor heart rate during all exercise sessions"],
            "💊 Medical":   ["URGENT cardiologist consultation — do not delay",
                             "Stress/treadmill test likely to be recommended",
                             "Statin therapy almost certainly needed",
                             "Regular ECG and echocardiogram monitoring"],
            "🧘 Lifestyle": ["Complete smoking cessation — non-negotiable",
                             "Strict alcohol restriction or elimination",
                             "Log blood pressure twice daily (morning & evening)",
                             "Cardiac rehab program for structured recovery"],
        },
        "very_high": {
            "🥗 Diet":      ["Strict cardiac diet under direct medical supervision",
                             "Very low sodium — under 1,500 mg/day",
                             "Cardiac-specific meal plan from a clinical nutritionist",
                             "No caffeine, alcohol, or stimulants"],
            "🏃 Exercise":  ["Absolutely no strenuous activity until fully evaluated",
                             "Gentle walking only with explicit medical clearance",
                             "Cardiac rehab is mandatory after evaluation"],
            "💊 Medical":   ["IMMEDIATE cardiologist evaluation — call today",
                             "Cardiac imaging: echocardiogram and/or angiogram",
                             "Multiple medications almost certainly required",
                             "Know heart attack signs: chest pain, arm pain, sweating"],
            "🧘 Lifestyle": ["Emergency action plan clearly documented",
                             "Family members must know CPR",
                             "Always carry prescribed medications",
                             "ICU-level monitoring may be recommended"],
        },
    },
    "Parkinsons": {
        "low": {
            "🥗 Diet":      ["Antioxidant-rich foods: berries, leafy greens, nuts",
                             "Adequate protein (lean meats, legumes) for muscle health",
                             "Omega-3 fatty acids: fish, flaxseed, walnuts"],
            "🏃 Exercise":  ["Regular aerobic exercise 3–4×/week",
                             "Yoga or tai chi for balance and coordination",
                             "Dance or rhythmic movement — proven brain benefits"],
            "💊 Medical":   ["Neurological checkup if family history of Parkinson's",
                             "Monitor for early signs: tremors, slow movement, rigidity",
                             "Annual general health checkup sufficient at this stage"],
            "🧘 Lifestyle": ["Mental stimulation: puzzles, reading, learning new skills",
                             "Strong social engagement — fight isolation",
                             "7–8 hrs quality sleep — critical for brain health"],
        },
        "moderate": {
            "🥗 Diet":      ["Anti-inflammatory diet — limit processed foods",
                             "Avoid excessive protein at medication times (levodopa users)",
                             "Adequate hydration — 8 glasses/day minimum",
                             "CoQ10 and Vitamin D supplements (consult doctor first)"],
            "🏃 Exercise":  ["Balance training: tai chi, yoga, water aerobics",
                             "Strength training 2×/week to fight muscle loss",
                             "Daily stretching routine for flexibility"],
            "💊 Medical":   ["Consult neurologist for comprehensive evaluation",
                             "UPDRS (motor assessment) scale evaluation",
                             "MRI or DaTscan may be recommended",
                             "Track any symptom changes in a diary"],
            "🧘 Lifestyle": ["Daily voice exercises to maintain speech clarity",
                             "Fine motor skill practice: writing, crafts, piano",
                             "Occupational therapy evaluation recommended"],
        },
        "high": {
            "🥗 Diet":      ["Time protein intake away from levodopa medication",
                             "High-fiber diet to prevent constipation (very common)",
                             "Full registered dietician consultation",
                             "Soft foods if chewing/swallowing becomes difficult"],
            "🏃 Exercise":  ["Parkinson's-specific programs: Rock Steady Boxing, LSVT BIG",
                             "Daily supervised physical therapy sessions",
                             "Fall prevention exercises — balance boards, stepping",
                             "Water-based therapy if land exercise is difficult"],
            "💊 Medical":   ["Regular neurologist follow-up every 3 months",
                             "Brain imaging (DaTscan or MRI) for staging",
                             "Dopaminergic therapy (levodopa) evaluation",
                             "Speech-language therapy assessment urgently"],
            "🧘 Lifestyle": ["Home safety assessment — remove trip hazards",
                             "Grab bars, non-slip mats, raised toilet seats",
                             "Build a caregiver support network now",
                             "Join a Parkinson's support group"],
        },
        "very_high": {
            "🥗 Diet":      ["Dysphagia-modified diet if swallowing is compromised",
                             "Clinical dietician to manage full nutritional plan",
                             "High-calorie foods to prevent weight loss",
                             "Thickened liquids may be needed — speech therapist to advise"],
            "🏃 Exercise":  ["Supervised therapy only — no independent strenuous activity",
                             "Comprehensive fall prevention program is mandatory",
                             "Assistive devices: walker, cane, wheelchair as needed",
                             "Pool therapy for safe low-impact movement"],
            "💊 Medical":   ["Immediate specialist neurological evaluation",
                             "Deep Brain Stimulation (DBS) consultation if appropriate",
                             "Multidisciplinary team: neurologist, physio, OT, speech",
                             "Palliative care team involvement for quality of life"],
            "🧘 Lifestyle": ["Full-time caregiver or structured care support",
                             "Advance care planning — document wishes legally",
                             "Full home accessibility adaptations",
                             "Emotional and psychological support for patient and family"],
        },
    },
}

def get_subs(disease, prob):
    key = "low" if prob < 25 else "moderate" if prob < 50 else "high" if prob < 75 else "very_high"
    return SUBS[disease][key]

# ══════════════════════════════════════════════════════════════════════════════
#  RESULT UI  (gauge + suggestions)
# ══════════════════════════════════════════════════════════════════════════════
def show_result(disease, result, prob):
    risk_lbl, risk_col, risk_ico = get_risk(prob)
    positive = result == "Positive"
    res_col  = "#ef4444" if positive else "#10b981"
    res_ico  = "⚠️" if positive else "✅"
    res_txt  = f"{res_ico} {disease} {'Detected' if positive else 'Not Detected'}"

    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.03);border:1px solid {res_col}55;
         border-radius:18px;padding:22px 20px 10px;text-align:center;margin:16px 0;">
        <div style="font-size:24px;font-weight:800;color:{res_col};">{res_txt}</div>
        <div style="color:rgba(200,220,240,0.55);font-size:13px;margin-top:5px;">
            AI Confidence Score: <strong style="color:{risk_col};">{prob}%</strong>
            &nbsp;|&nbsp; {risk_ico} <strong style="color:{risk_col};">{risk_lbl}</strong>
        </div>
    </div>""", unsafe_allow_html=True)

    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        number={"suffix": "%", "font": {"size": 52, "color": risk_col}},
        title={"text": f"<b>{risk_ico} {risk_lbl}</b>", "font": {"size": 17, "color": "#e2e8f0"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#555"},
            "bar":  {"color": risk_col, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
            "steps": [
                {"range": [0,  25], "color": "rgba(16,185,129,0.12)"},
                {"range": [25, 50], "color": "rgba(245,158,11,0.12)"},
                {"range": [50, 75], "color": "rgba(249,115,22,0.12)"},
                {"range": [75,100], "color": "rgba(239,68,68,0.12)"},
            ],
            "threshold": {"line": {"color": risk_col, "width": 4}, "value": prob},
        }
    ))
    fig.update_layout(height=270, margin=dict(l=20,r=20,t=60,b=10),
                      paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
    st.plotly_chart(fig, use_container_width=True)

    # Suggestions
    subs = get_subs(disease, prob)
    st.markdown(f"### 💡 Personalised Recommendations")
    c1, c2 = st.columns(2)
    for i, (cat, tips) in enumerate(subs.items()):
        col = c1 if i % 2 == 0 else c2
        with col:
            lis = "".join(f"<li style='margin-bottom:7px;color:rgba(210,225,240,0.78);font-size:13px;'>{t}</li>" for t in tips)
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.08);
                 border-radius:14px;padding:18px 16px;margin-bottom:14px;">
                <div style="font-size:15px;font-weight:700;color:#e2e8f0;margin-bottom:10px;">{cat}</div>
                <ul style="margin:0;padding-left:18px;">{lis}</ul>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════
for k, v in [("logged_in",False),("username",None),("user_name",None),("role",None),("page","Home")]:
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
#  AUTH PAGE  (shown when not logged in)
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    st.markdown("""
<style>
@keyframes fadeInUp{from{opacity:0;transform:translateY(30px)}to{opacity:1;transform:translateY(0)}}
@keyframes shimmerText{0%{background-position:0% center}100%{background-position:200% center}}
.auth-outer{max-width:460px;margin:60px auto 0;animation:fadeInUp .7s ease both;}
.auth-logo{text-align:center;font-size:52px;margin-bottom:10px;}
.auth-title{
    text-align:center;font-size:32px;font-weight:900;
    background:linear-gradient(90deg,#fff 0%,#00d4ff 40%,#a78bfa 70%,#fff 100%);
    background-size:200% auto;-webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text;animation:shimmerText 4s linear infinite;margin-bottom:4px;
}
.auth-sub{text-align:center;color:rgba(200,220,240,0.5);font-size:13px;margin-bottom:28px;}
.stTabs [data-baseweb="tab-list"]{justify-content:center;gap:10px;}
.stTabs [data-baseweb="tab"]{font-weight:700;font-size:14px;}
div[data-testid="stForm"] .stButton>button,
.auth-outer .stButton>button{
    background:linear-gradient(135deg,#00d4ff,#7c3aed)!important;
    color:white!important;border:none!important;border-radius:12px!important;
    font-weight:700!important;font-size:15px!important;padding:14px!important;
    width:100%!important;margin-top:6px!important;
}
</style>
<div class="auth-outer">
    <div class="auth-logo">🏥</div>
    <div class="auth-title">Health Assistant</div>
    <p class="auth-sub">AI-Powered Multiple Disease Prediction System</p>
</div>
""", unsafe_allow_html=True)

    with st.container():
        tab_login, tab_reg = st.tabs(["🔑  Login", "📝  Create Account"])

        with tab_login:
            st.markdown("<br>", unsafe_allow_html=True)
            l_user = st.text_input("Username", placeholder="Enter your username", key="l_u")
            l_pass = st.text_input("Password", type="password", placeholder="Enter your password", key="l_p")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Login →", use_container_width=True, key="btn_login"):
                user = do_login(l_user, l_pass)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.username  = l_user
                    st.session_state.user_name = user["name"]
                    st.session_state.role      = user["role"]
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
            st.markdown("<br><div style='text-align:center;color:rgba(200,220,240,0.4);font-size:12px;'>Default admin → username: <b>admin</b> &nbsp;|&nbsp; password: <b>admin123</b></div>", unsafe_allow_html=True)

        with tab_reg:
            st.markdown("<br>", unsafe_allow_html=True)
            r_name = st.text_input("Full Name",        placeholder="John Doe",       key="r_n")
            r_user = st.text_input("Username",         placeholder="johndoe",        key="r_u")
            r_pass = st.text_input("Password",         type="password", placeholder="Min 6 characters", key="r_p")
            r_conf = st.text_input("Confirm Password", type="password", placeholder="Repeat password",  key="r_c")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Create Account →", use_container_width=True, key="btn_reg"):
                if not r_name or not r_user:
                    st.error("Name and username are required.")
                elif r_pass != r_conf:
                    st.error("Passwords do not match.")
                elif len(r_pass) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    ok, msg = do_register(r_user, r_name, r_pass)
                    st.success(msg) if ok else st.error(msg)

    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP  (logged-in users only)
# ══════════════════════════════════════════════════════════════════════════════
pages = ["Home", "Diabetes Prediction", "Heart Disease Prediction",
         "Parkinsons Prediction", "My History"]
icons = ["house-fill", "activity", "heart", "person", "clock-history"]
if st.session_state.role == "admin":
    pages.append("Dashboard")
    icons.append("bar-chart-fill")

with st.sidebar:
    st.markdown(f"<div style='padding:10px 0 6px;font-size:14px;color:rgba(200,220,240,0.7);'>👋 Welcome, <b>{st.session_state.user_name}</b></div>", unsafe_allow_html=True)
    selected = option_menu(
        "Multiple Disease Prediction",
        pages, menu_icon="hospital-fill", icons=icons,
        default_index=pages.index(st.session_state.page) if st.session_state.page in pages else 0
    )
    st.session_state.page = selected
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚪  Logout", use_container_width=True):
        for k in ["logged_in","username","user_name","role"]:
            st.session_state[k] = False if k == "logged_in" else None
        st.session_state.page = "Home"
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════
if selected == "Home":
    st.markdown("""
<style>
@keyframes gradientShift{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
@keyframes fadeInUp{from{opacity:0;transform:translateY(40px)}to{opacity:1;transform:translateY(0)}}
@keyframes shimmerText{0%{background-position:0% center}100%{background-position:200% center}}
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-12px)}}
@keyframes heartbeat{0%,100%{transform:scale(1)}15%{transform:scale(1.35)}30%{transform:scale(1)}45%{transform:scale(1.2)}60%{transform:scale(1)}}
@keyframes scanline{0%{transform:translateX(-100%)}100%{transform:translateX(500%)}}
@keyframes drawECG{to{stroke-dashoffset:0}}
@keyframes blinkArr{0%,100%{opacity:1}50%{opacity:0.3}}
.hero-wrap{background:linear-gradient(135deg,#04080f 0%,#0a1628 45%,#0d0a2e 100%);background-size:300% 300%;animation:gradientShift 12s ease infinite;border-radius:24px;padding:72px 40px 52px;text-align:center;position:relative;overflow:hidden;border:1px solid rgba(0,212,255,0.12);margin-bottom:24px;}
.hero-wrap::before{content:'';position:absolute;width:500px;height:500px;background:radial-gradient(circle,rgba(0,212,255,0.05) 0%,transparent 65%);top:-160px;left:-120px;border-radius:50%;pointer-events:none;}
.hero-badge{display:inline-block;background:rgba(0,212,255,0.07);border:1px solid rgba(0,212,255,0.35);color:#00d4ff;padding:7px 24px;border-radius:50px;font-size:11px;font-weight:700;letter-spacing:3px;text-transform:uppercase;margin-bottom:28px;animation:fadeInUp .7s ease both;}
.hero-title{font-size:clamp(32px,5.5vw,64px);font-weight:900;background:linear-gradient(90deg,#fff 0%,#00d4ff 35%,#a78bfa 65%,#fff 100%);background-size:200% auto;-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;animation:shimmerText 4s linear infinite,fadeInUp .7s ease .15s both;line-height:1.15;margin-bottom:20px;}
.hero-sub{font-size:clamp(13px,1.8vw,17px);color:rgba(200,220,240,0.62);max-width:560px;margin:0 auto 40px;line-height:1.78;animation:fadeInUp .7s ease .3s both;}
.ecg-wrap{max-width:600px;margin:0 auto 8px;animation:fadeInUp .7s ease .45s both;}
.ecg-path{fill:none;stroke:url(#ecgGrad);stroke-width:2.5;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:1300;stroke-dashoffset:1300;animation:drawECG 2.8s ease forwards 1s;filter:drop-shadow(0 0 5px rgba(0,212,255,0.75));}
.stats-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin-bottom:36px;}
.stat-box{background:rgba(255,255,255,0.022);border:1px solid rgba(255,255,255,0.07);border-radius:18px;padding:30px 14px;text-align:center;position:relative;overflow:hidden;transition:transform .3s,border-color .3s,box-shadow .3s;animation:fadeInUp .7s ease both;}
.stat-box:nth-child(1){animation-delay:.1s}.stat-box:nth-child(2){animation-delay:.2s}.stat-box:nth-child(3){animation-delay:.3s}
.stat-box::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,rgba(0,212,255,0.7),transparent);animation:scanline 3.5s linear infinite;}
.stat-box:hover{transform:translateY(-7px);border-color:rgba(0,212,255,0.28);box-shadow:0 22px 44px rgba(0,0,0,0.35),0 0 36px rgba(0,212,255,0.08);}
.stat-icon{font-size:32px;margin-bottom:10px;}
.stat-val{font-size:40px;font-weight:900;background:linear-gradient(135deg,#00d4ff,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1;margin-bottom:7px;}
.stat-lbl{font-size:11px;letter-spacing:2px;text-transform:uppercase;color:rgba(200,220,240,0.42);}
.sec-title{text-align:center;font-size:27px;font-weight:800;color:#e2e8f0;margin:50px 0 6px;}
.sec-sub{text-align:center;color:rgba(200,220,240,0.42);font-size:14px;margin-bottom:28px;}
.cards-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:22px;margin-bottom:50px;}
.d-card{border-radius:22px;padding:38px 22px 30px;text-align:center;position:relative;overflow:hidden;transition:transform .4s cubic-bezier(.175,.885,.32,1.275),box-shadow .4s,border-color .4s;animation:fadeInUp .7s ease both;}
.d-card:nth-child(1){animation-delay:.1s}.d-card:nth-child(2){animation-delay:.22s}.d-card:nth-child(3){animation-delay:.34s}
.d-card:hover{transform:translateY(-13px) scale(1.025);}
.d-card-diab{background:linear-gradient(145deg,rgba(251,146,60,.09) 0%,rgba(8,16,32,.95) 100%);border:1px solid rgba(251,146,60,.2);}
.d-card-diab:hover{border-color:rgba(251,146,60,.55);box-shadow:0 32px 64px rgba(0,0,0,.45),0 0 60px rgba(251,146,60,.14);}
.d-card-heart{background:linear-gradient(145deg,rgba(239,68,68,.09) 0%,rgba(8,16,32,.95) 100%);border:1px solid rgba(239,68,68,.2);}
.d-card-heart:hover{border-color:rgba(239,68,68,.55);box-shadow:0 32px 64px rgba(0,0,0,.45),0 0 60px rgba(239,68,68,.14);}
.d-card-park{background:linear-gradient(145deg,rgba(124,58,237,.1) 0%,rgba(8,16,32,.95) 100%);border:1px solid rgba(124,58,237,.22);}
.d-card-park:hover{border-color:rgba(124,58,237,.55);box-shadow:0 32px 64px rgba(0,0,0,.45),0 0 60px rgba(124,58,237,.14);}
.d-icon{font-size:58px;display:block;margin-bottom:14px;animation:float 3.5s ease-in-out infinite;}
.d-card-diab .d-icon{animation-delay:0s}.d-card-heart .d-icon{animation:heartbeat 1.6s ease-in-out infinite}.d-card-park .d-icon{animation-delay:1.2s}
.d-name{font-size:19px;font-weight:700;color:#e2e8f0;margin-bottom:10px;}
.d-desc{font-size:12.5px;color:rgba(200,220,240,0.48);line-height:1.68;margin-bottom:18px;}
.d-tag{display:inline-block;padding:3px 11px;border-radius:50px;font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin:2px;}
.tag-o{background:rgba(251,146,60,.12);color:#fb923c;border:1px solid rgba(251,146,60,.3);}
.tag-r{background:rgba(239,68,68,.12);color:#f87171;border:1px solid rgba(239,68,68,.3);}
.tag-p{background:rgba(167,139,250,.12);color:#a78bfa;border:1px solid rgba(167,139,250,.3);}
.tag-g{background:rgba(16,185,129,.12);color:#34d399;border:1px solid rgba(16,185,129,.3);}
.steps-grid{display:grid;grid-template-columns:1fr 44px 1fr 44px 1fr;align-items:center;margin-bottom:44px;}
.step-box{background:rgba(255,255,255,0.022);border:1px solid rgba(255,255,255,0.07);border-radius:18px;padding:30px 18px;text-align:center;animation:fadeInUp .7s ease both;}
.step-box:nth-child(1){animation-delay:.1s}.step-box:nth-child(3){animation-delay:.3s}.step-box:nth-child(5){animation-delay:.5s}
.step-arr{text-align:center;font-size:24px;color:rgba(0,212,255,.55);animation:blinkArr 2s ease-in-out infinite;}
.step-num{width:48px;height:48px;background:linear-gradient(135deg,#00d4ff,#7c3aed);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:19px;font-weight:900;color:white;margin:0 auto 14px;}
.step-ttl{font-size:15px;font-weight:700;color:#e2e8f0;margin-bottom:8px;}
.step-dsc{font-size:12px;color:rgba(200,220,240,0.44);line-height:1.65;}
.disclaimer{background:rgba(251,191,36,.055);border:1px solid rgba(251,191,36,.25);border-radius:14px;padding:14px 22px;text-align:center;color:rgba(251,191,36,.82);font-size:12.5px;margin-bottom:8px;}
div[data-testid="column"] .stButton>button{background:linear-gradient(135deg,#00d4ff 0%,#7c3aed 100%)!important;color:white!important;border:none!important;border-radius:14px!important;font-weight:700!important;font-size:14px!important;padding:16px!important;letter-spacing:.4px!important;transition:transform .25s ease,box-shadow .25s ease!important;}
div[data-testid="column"] .stButton>button:hover{transform:translateY(-4px)!important;box-shadow:0 12px 35px rgba(0,212,255,.35)!important;}
</style>""", unsafe_allow_html=True)

    st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">🧬 &nbsp; AI &nbsp;·&nbsp; Machine Learning &nbsp;·&nbsp; Healthcare</div>
    <div class="hero-title">Predict. Detect. Protect.</div>
    <p class="hero-sub">Harness the power of Machine Learning to detect critical health conditions early.
    Fast, accurate, and intelligent disease prediction right at your fingertips.</p>
    <div class="ecg-wrap">
        <svg viewBox="0 0 600 62" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:62px;">
            <defs>
                <linearGradient id="ecgGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%"   style="stop-color:#00d4ff;stop-opacity:0.2"/>
                    <stop offset="35%"  style="stop-color:#00d4ff;stop-opacity:1"/>
                    <stop offset="65%"  style="stop-color:#a78bfa;stop-opacity:1"/>
                    <stop offset="100%" style="stop-color:#a78bfa;stop-opacity:0.2"/>
                </linearGradient>
            </defs>
            <path class="ecg-path" d="M0,31 L52,31 L62,25 L72,31 L97,31 L107,37 L117,2 L127,60 L137,31 L147,22 L162,31 L217,31 L227,25 L237,31 L262,31 L272,37 L282,2 L292,60 L302,31 L312,22 L327,31 L382,31 L392,25 L402,31 L427,31 L437,37 L447,2 L457,60 L467,31 L477,22 L492,31 L600,31"/>
        </svg>
    </div>
</div>
<div class="stats-grid">
    <div class="stat-box"><div class="stat-icon">🧬</div><div class="stat-val">3</div><div class="stat-lbl">Diseases Covered</div></div>
    <div class="stat-box"><div class="stat-icon">⚡</div><div class="stat-val">SVM</div><div class="stat-lbl">ML Algorithm</div></div>
    <div class="stat-box"><div class="stat-icon">🎯</div><div class="stat-val">~78%</div><div class="stat-lbl">Model Accuracy</div></div>
</div>
<div class="sec-title">Disease Prediction Modules</div>
<p class="sec-sub">Three AI-powered tools — select from the sidebar or click below</p>
<div class="cards-grid">
    <div class="d-card d-card-diab"><span class="d-icon">🩺</span><div class="d-name">Diabetes Prediction</div><p class="d-desc">Analyses 8 key health indicators including glucose levels, BMI, and insulin resistance to assess diabetes risk.</p><span class="d-tag tag-o">Glucose</span><span class="d-tag tag-o">BMI</span><span class="d-tag tag-g">Insulin</span></div>
    <div class="d-card d-card-heart"><span class="d-icon">❤️</span><div class="d-name">Heart Disease Prediction</div><p class="d-desc">Evaluates 13 cardiovascular parameters — cholesterol, ECG results, chest pain type, and more.</p><span class="d-tag tag-r">Cholesterol</span><span class="d-tag tag-r">ECG</span><span class="d-tag tag-g">Blood Pressure</span></div>
    <div class="d-card d-card-park"><span class="d-icon">🧠</span><div class="d-name">Parkinson's Prediction</div><p class="d-desc">Processes 22 vocal biomarkers and frequency measurements to identify early Parkinson's patterns.</p><span class="d-tag tag-p">Voice Analysis</span><span class="d-tag tag-p">Jitter</span><span class="d-tag tag-g">Shimmer</span></div>
</div>
<div class="sec-title">How It Works</div>
<p class="sec-sub">Three simple steps to your health prediction</p>
<div class="steps-grid">
    <div class="step-box"><div class="step-num">1</div><div class="step-ttl">Enter Your Data</div><p class="step-dsc">Input your health metrics and medical parameters into the form fields</p></div>
    <div class="step-arr">→</div>
    <div class="step-box"><div class="step-num">2</div><div class="step-ttl">AI Analysis</div><p class="step-dsc">Our trained SVM model processes your inputs using patterns learned from real medical data</p></div>
    <div class="step-arr">→</div>
    <div class="step-box"><div class="step-num">3</div><div class="step-ttl">Instant Result</div><p class="step-dsc">Get a clear probability score, risk level, and personalised health recommendations</p></div>
</div>
<div class="disclaimer">⚠️ <strong>Medical Disclaimer:</strong> This tool is for educational purposes only and is <strong>not</strong> a substitute for professional medical advice. Always consult a qualified healthcare provider.</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🩺  Predict Diabetes", use_container_width=True):
            st.session_state.page = "Diabetes Prediction"; st.rerun()
    with c2:
        if st.button("❤️  Predict Heart Disease", use_container_width=True):
            st.session_state.page = "Heart Disease Prediction"; st.rerun()
    with c3:
        if st.button("🧠  Predict Parkinson's", use_container_width=True):
            st.session_state.page = "Parkinsons Prediction"; st.rerun()

    # ── CREATED BY ──
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);
         border-radius:20px;padding:32px 36px;text-align:center;max-width:560px;margin:0 auto;">
        <div style="font-size:11px;letter-spacing:3px;text-transform:uppercase;
             color:rgba(200,220,240,0.4);margin-bottom:18px;">Created By</div>
    </div>
    """, unsafe_allow_html=True)

    cb1, cb2, cb3 = st.columns([1, 1.6, 1])
    with cb2:
        img_path = os.path.join(WD, "1777884568779.png")
        if os.path.exists(img_path):
            st.image(img_path, width=120)
        st.markdown("""
        <div style="text-align:center;margin-top:14px;">
            <div style="font-size:22px;font-weight:800;
                 background:linear-gradient(135deg,#00d4ff,#a78bfa);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                 background-clip:text;">Swastika Srivastava</div>
            <div style="color:rgba(200,220,240,0.55);font-size:13px;
                 line-height:1.7;margin-top:8px;max-width:360px;margin-left:auto;margin-right:auto;">
                A college project harnessing Machine Learning to predict Diabetes,
                Heart Disease, and Parkinson's — making early health detection
                smarter, faster, and accessible to everyone.
            </div>
            <div style="margin-top:12px;">
                <span style="background:rgba(0,212,255,0.08);border:1px solid rgba(0,212,255,0.25);
                      color:#00d4ff;padding:4px 14px;border-radius:50px;font-size:11px;
                      font-weight:700;letter-spacing:1px;">🎓 College Project</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DIABETES PAGE
# ══════════════════════════════════════════════════════════════════════════════
if selected == "Diabetes Prediction":
    st.title("Diabetes Prediction using ML")
    col1, col2, col3 = st.columns(3)
    with col1: Pregnancies            = st.text_input("Number of Pregnancies")
    with col2: Glucose                = st.text_input("Glucose Level")
    with col3: BloodPressure          = st.text_input("Blood Pressure value")
    with col1: SkinThickness          = st.text_input("Skin Thickness value")
    with col2: Insulin                = st.text_input("Insulin Level")
    with col3: BMI                    = st.text_input("BMI value")
    with col1: DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function value")
    with col2: Age                    = st.text_input("Age of the Person")

    if st.button("Diabetes Test Result"):
        try:
            x = [float(v) for v in [Pregnancies, Glucose, BloodPressure, SkinThickness,
                                     Insulin, BMI, DiabetesPedigreeFunction, Age]]
            pred = diabetes_model.predict([x])[0]
            prob = get_probability(diabetes_model, x)
            result = "Positive" if pred == 1 else "Negative"
            record_prediction(st.session_state.username, "Diabetes", result, prob)
            show_result("Diabetes", result, prob)
        except ValueError:
            st.error("Please fill in all fields with valid numbers.")


# ══════════════════════════════════════════════════════════════════════════════
#  HEART DISEASE PAGE
# ══════════════════════════════════════════════════════════════════════════════
if selected == "Heart Disease Prediction":
    st.title("Heart Disease Prediction using ML")
    col1, col2, col3 = st.columns(3)
    with col1: age      = st.text_input("Age")
    with col2: sex      = st.text_input("Sex")
    with col3: cp       = st.text_input("Chest Pain types")
    with col1: trestbps = st.text_input("Resting Blood Pressure")
    with col2: chol     = st.text_input("Serum Cholestoral in mg/dl")
    with col3: fbs      = st.text_input("Fasting Blood Sugar > 120 mg/dl")
    with col1: restecg  = st.text_input("Resting Electrocardiographic results")
    with col2: thalach  = st.text_input("Maximum Heart Rate achieved")
    with col3: exang    = st.text_input("Exercise Induced Angina")
    with col1: oldpeak  = st.text_input("ST depression induced by exercise")
    with col2: slope    = st.text_input("Slope of the peak exercise ST segment")
    with col3: ca       = st.text_input("Major vessels colored by flourosopy")
    with col1: thal     = st.text_input("thal: 0 = normal; 1 = fixed defect; 2 = reversable defect")

    if st.button("Heart Disease Test Result"):
        try:
            x = [float(v) for v in [age, sex, cp, trestbps, chol, fbs, restecg,
                                     thalach, exang, oldpeak, slope, ca, thal]]
            pred = heart_disease_model.predict([x])[0]
            prob = get_probability(heart_disease_model, x)
            result = "Positive" if pred == 1 else "Negative"
            record_prediction(st.session_state.username, "Heart Disease", result, prob)
            show_result("Heart", result, prob)
        except ValueError:
            st.error("Please fill in all fields with valid numbers.")


# ══════════════════════════════════════════════════════════════════════════════
#  PARKINSONS PAGE
# ══════════════════════════════════════════════════════════════════════════════
if selected == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction using ML")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: fo             = st.text_input("MDVP:Fo(Hz)")
    with col2: fhi            = st.text_input("MDVP:Fhi(Hz)")
    with col3: flo            = st.text_input("MDVP:Flo(Hz)")
    with col4: Jitter_percent = st.text_input("MDVP:Jitter(%)")
    with col5: Jitter_Abs     = st.text_input("MDVP:Jitter(Abs)")
    with col1: RAP            = st.text_input("MDVP:RAP")
    with col2: PPQ            = st.text_input("MDVP:PPQ")
    with col3: DDP            = st.text_input("Jitter:DDP")
    with col4: Shimmer        = st.text_input("MDVP:Shimmer")
    with col5: Shimmer_dB     = st.text_input("MDVP:Shimmer(dB)")
    with col1: APQ3           = st.text_input("Shimmer:APQ3")
    with col2: APQ5           = st.text_input("Shimmer:APQ5")
    with col3: APQ            = st.text_input("MDVP:APQ")
    with col4: DDA            = st.text_input("Shimmer:DDA")
    with col5: NHR            = st.text_input("NHR")
    with col1: HNR            = st.text_input("HNR")
    with col2: RPDE           = st.text_input("RPDE")
    with col3: DFA            = st.text_input("DFA")
    with col4: spread1        = st.text_input("spread1")
    with col5: spread2        = st.text_input("spread2")
    with col1: D2             = st.text_input("D2")
    with col2: PPE            = st.text_input("PPE")

    if st.button("Parkinson's Test Result"):
        try:
            x = [float(v) for v in [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP,
                                     Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR,
                                     RPDE, DFA, spread1, spread2, D2, PPE]]
            pred = parkinsons_model.predict([x])[0]
            prob = get_probability(parkinsons_model, x)
            result = "Positive" if pred == 1 else "Negative"
            record_prediction(st.session_state.username, "Parkinsons", result, prob)
            show_result("Parkinsons", result, prob)
        except ValueError:
            st.error("Please fill in all fields with valid numbers.")


# ══════════════════════════════════════════════════════════════════════════════
#  MY HISTORY PAGE
# ══════════════════════════════════════════════════════════════════════════════
if selected == "My History":
    st.title("📋 My Prediction History")
    data = load_data()
    preds = data["users"].get(st.session_state.username, {}).get("predictions", [])

    if not preds:
        st.info("You haven't made any predictions yet. Try a disease prediction from the sidebar!")
    else:
        st.markdown(f"**Total predictions made: {len(preds)}**")
        for p in reversed(preds):
            risk_lbl, risk_col, risk_ico = get_risk(p["probability"])
            res_col = "#ef4444" if p["result"] == "Positive" else "#10b981"
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.08);
                 border-radius:14px;padding:16px 20px;margin-bottom:12px;
                 display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;">
                <div>
                    <div style="font-size:16px;font-weight:700;color:#e2e8f0;">🏥 {p['disease']}</div>
                    <div style="font-size:12px;color:rgba(200,220,240,0.45);margin-top:3px;">🕐 {p['ts']}</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:14px;font-weight:700;color:{res_col};">{p['result']}</div>
                    <div style="font-size:13px;color:{risk_col};">{risk_ico} {risk_lbl} &nbsp;·&nbsp; {p['probability']}%</div>
                </div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ADMIN DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if selected == "Dashboard" and st.session_state.role == "admin":
    st.title("📊 Admin Dashboard")
    data = load_data()
    all_users = data["users"]

    # ── Aggregate stats
    total_users = len(all_users)
    all_preds   = [p for u in all_users.values() for p in u.get("predictions", [])]
    total_preds = len(all_preds)

    disease_counts = {}
    risk_counts    = {"Low Risk": 0, "Moderate Risk": 0, "High Risk": 0, "Very High Risk": 0}
    pos_counts     = {"Positive": 0, "Negative": 0}

    for p in all_preds:
        disease_counts[p["disease"]] = disease_counts.get(p["disease"], 0) + 1
        lbl, _, _ = get_risk(p["probability"])
        risk_counts[lbl] = risk_counts.get(lbl, 0) + 1
        pos_counts[p["result"]] = pos_counts.get(p["result"], 0) + 1

    # ── KPI Cards
    k1, k2, k3, k4 = st.columns(4)
    for col, icon, val, lbl in [
        (k1, "👥", total_users,  "Total Users"),
        (k2, "🔬", total_preds,  "Total Predictions"),
        (k3, "⚠️", pos_counts.get("Positive", 0), "Positive Results"),
        (k4, "✅", pos_counts.get("Negative", 0), "Negative Results"),
    ]:
        with col:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.08);
                 border-radius:16px;padding:24px 16px;text-align:center;">
                <div style="font-size:30px;">{icon}</div>
                <div style="font-size:36px;font-weight:900;background:linear-gradient(135deg,#00d4ff,#a78bfa);
                     -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">{val}</div>
                <div style="font-size:11px;letter-spacing:2px;text-transform:uppercase;
                     color:rgba(200,220,240,0.45);margin-top:4px;">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown("#### 🥧 Predictions by Disease")
        if disease_counts:
            fig = px.pie(
                names=list(disease_counts.keys()),
                values=list(disease_counts.values()),
                color_discrete_sequence=["#00d4ff", "#ef4444", "#a78bfa"],
                hole=0.45
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
                              legend=dict(orientation="h", y=-0.15),
                              margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No predictions yet.")

    with ch2:
        st.markdown("#### 📊 Risk Level Distribution")
        if any(risk_counts.values()):
            colors = ["#10b981", "#f59e0b", "#f97316", "#ef4444"]
            fig = go.Figure(go.Bar(
                x=list(risk_counts.keys()),
                y=list(risk_counts.values()),
                marker_color=colors,
                marker_line_color="rgba(0,0,0,0)",
                text=list(risk_counts.values()),
                textposition="outside",
                textfont=dict(color="#e2e8f0")
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0", showlegend=False,
                xaxis=dict(showgrid=False), yaxis=dict(showgrid=False, showticklabels=False),
                margin=dict(l=10,r=10,t=10,b=10)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No predictions yet.")

    # ── Recent Predictions Table
    st.markdown("#### 🕐 Recent Predictions (All Users)")
    if all_preds:
        recent = sorted(all_preds, key=lambda x: x["ts"], reverse=True)[:15]
        # Find username for each pred
        pred_with_user = []
        for uname, udata in all_users.items():
            for p in udata.get("predictions", []):
                pred_with_user.append({**p, "user": udata["name"]})
        pred_with_user = sorted(pred_with_user, key=lambda x: x["ts"], reverse=True)[:15]

        for p in pred_with_user:
            risk_lbl, risk_col, risk_ico = get_risk(p["probability"])
            res_col = "#ef4444" if p["result"] == "Positive" else "#10b981"
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);
                 border-radius:12px;padding:12px 18px;margin-bottom:8px;
                 display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
                <div style="font-size:13px;color:#e2e8f0;"><b>{p['user']}</b> &nbsp;·&nbsp; {p['disease']}</div>
                <div style="font-size:12px;color:rgba(200,220,240,0.4);">{p['ts']}</div>
                <div style="font-size:13px;"><span style="color:{res_col};font-weight:700;">{p['result']}</span>
                    &nbsp; <span style="color:{risk_col};">{risk_ico} {p['probability']}%</span></div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("No predictions recorded yet.")

    # ── Registered Users
    st.markdown("#### 👥 Registered Users")
    for uname, udata in all_users.items():
        n_preds = len(udata.get("predictions", []))
        role_badge = "🔑 Admin" if udata["role"] == "admin" else "👤 User"
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);
             border-radius:12px;padding:12px 18px;margin-bottom:8px;
             display:flex;justify-content:space-between;align-items:center;">
            <div><b style="color:#e2e8f0;">{udata['name']}</b>
                <span style="color:rgba(200,220,240,0.4);font-size:12px;"> &nbsp;@{uname}</span></div>
            <div style="font-size:12px;color:rgba(200,220,240,0.5);">
                {role_badge} &nbsp;·&nbsp; {n_preds} predictions &nbsp;·&nbsp; Joined {udata.get('created_at','—')}
            </div>
        </div>""", unsafe_allow_html=True)
