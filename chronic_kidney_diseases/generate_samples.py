"""
Generates clearly distinct sample CSV datasets for 4 users.
Each user has a different risk profile so model accuracy differs significantly.
"""
import numpy as np
import pandas as pd
import os

np.random.seed(0)

def save(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {path}  ({len(df)} rows)")

# ─────────────────────────────────────────────
# STROKE
# cols: id,gender,age,hypertension,heart_disease,
#       ever_married,work_type,Residence_type,
#       avg_glucose_level,bmi,smoking_status,stroke
# ─────────────────────────────────────────────
def make_stroke(n, seed,
                age_mean, age_std,
                htn_rate, hd_rate,
                glucose_mean, glucose_std,
                bmi_mean, bmi_std,
                stroke_rate,
                smoker_rate):
    np.random.seed(seed)
    ids      = np.arange(seed*1000, seed*1000+n)
    gender   = np.random.choice(['Male','Female'], n)
    age      = np.clip(np.random.normal(age_mean, age_std, n).astype(int), 18, 95)
    htn      = np.random.binomial(1, htn_rate, n)
    hd       = np.random.binomial(1, hd_rate, n)
    married  = np.where(age > 30, np.random.choice(['Yes','No'], n, p=[0.8,0.2]),
                        np.random.choice(['Yes','No'], n, p=[0.2,0.8]))
    work     = np.random.choice(['Private','Govt_job','Self-employed','Never_worked'], n,
                                p=[0.55,0.25,0.15,0.05])
    res      = np.random.choice(['Urban','Rural'], n)
    glucose  = np.clip(np.random.normal(glucose_mean, glucose_std, n), 55, 300)
    bmi      = np.clip(np.random.normal(bmi_mean, bmi_std, n), 15, 55)
    smoke    = np.random.choice(['never smoked','formerly smoked','smokes'], n,
                                p=[1-smoker_rate-0.15, 0.15, smoker_rate])
    stroke   = np.random.binomial(1, stroke_rate, n)
    return pd.DataFrame({'id':ids,'gender':gender,'age':age,
                         'hypertension':htn,'heart_disease':hd,
                         'ever_married':married,'work_type':work,
                         'Residence_type':res,'avg_glucose_level':glucose.round(2),
                         'bmi':bmi.round(2),'smoking_status':smoke,'stroke':stroke})

# user1 — young, healthy, low risk  → high accuracy (easy to classify)
save(make_stroke(120, 1, age_mean=28, age_std=6,  htn_rate=0.05, hd_rate=0.03,
                 glucose_mean=78,  glucose_std=8,  bmi_mean=22, bmi_std=2.5,
                 stroke_rate=0.05, smoker_rate=0.05),
     'sample_data/user1/stroke.csv')

# user2 — middle-aged, mixed risk  → medium accuracy
save(make_stroke(120, 2, age_mean=52, age_std=10, htn_rate=0.35, hd_rate=0.20,
                 glucose_mean=130, glucose_std=30, bmi_mean=29, bmi_std=5,
                 stroke_rate=0.30, smoker_rate=0.30),
     'sample_data/user2/stroke.csv')

# user3 — elderly, high risk  → lower accuracy (noisy boundary)
save(make_stroke(120, 3, age_mean=68, age_std=8,  htn_rate=0.70, hd_rate=0.55,
                 glucose_mean=195, glucose_std=40, bmi_mean=36, bmi_std=6,
                 stroke_rate=0.65, smoker_rate=0.55),
     'sample_data/user3/stroke.csv')

# user4 — severely imbalanced (almost all stroke)  → very different accuracy
save(make_stroke(120, 4, age_mean=74, age_std=6,  htn_rate=0.90, hd_rate=0.80,
                 glucose_mean=230, glucose_std=25, bmi_mean=40, bmi_std=4,
                 stroke_rate=0.88, smoker_rate=0.70),
     'sample_data/user4/stroke.csv')


# ─────────────────────────────────────────────
# KIDNEY DISEASE
# cols: id,age,bp,sg,al,su,bgr,bu,sc,sod,pot,classification
# ─────────────────────────────────────────────
def make_kidney(n, seed, ckd_frac,
                age_ckd, age_healthy,
                bp_ckd, bp_healthy,
                bgr_ckd, bgr_healthy,
                sc_ckd, sc_healthy,
                noise=0.0):
    np.random.seed(seed)
    n_ckd = int(n * ckd_frac)
    n_ok  = n - n_ckd

    def block(size, is_ckd):
        age  = np.clip(np.random.normal(age_ckd if is_ckd else age_healthy, 10, size).astype(int), 5, 90)
        bp   = np.clip(np.random.normal(bp_ckd  if is_ckd else bp_healthy,  12, size).astype(int), 50, 140)
        sg   = np.random.choice([1.005,1.010,1.015,1.020,1.025], size,
                                p=[0.35,0.35,0.20,0.07,0.03] if is_ckd else [0.02,0.05,0.18,0.40,0.35])
        al   = np.random.choice([0,1,2,3,4,5], size,
                                p=[0.05,0.15,0.30,0.25,0.15,0.10] if is_ckd else [0.75,0.15,0.06,0.03,0.01,0.00])
        su   = np.random.choice([0,1,2,3,4,5], size,
                                p=[0.10,0.20,0.30,0.20,0.15,0.05] if is_ckd else [0.85,0.10,0.03,0.01,0.01,0.00])
        bgr  = np.clip(np.random.normal(bgr_ckd if is_ckd else bgr_healthy, 25, size).astype(int), 50, 300)
        bu   = np.clip(np.random.normal(55 if is_ckd else 20, 15, size).astype(int), 10, 120)
        sc   = np.clip(np.random.normal(sc_ckd if is_ckd else sc_healthy, 0.7, size).round(1), 0.4, 9.0)
        sod  = np.clip(np.random.normal(125 if is_ckd else 141, 8, size).astype(int), 100, 155)
        pot  = np.clip(np.random.normal(5.5 if is_ckd else 4.0, 0.8, size).round(1), 2.5, 8.0)
        label = ['ckd']*size if is_ckd else ['notckd']*size
        # flip some labels to add noise/overlap
        if noise > 0:
            flip = np.random.binomial(1, noise, size).astype(bool)
            label = np.array(label)
            label[flip] = np.where(label[flip]=='ckd', 'notckd', 'ckd')
            label = list(label)
        return pd.DataFrame({'age':age,'bp':bp,'sg':sg,'al':al,'su':su,
                             'bgr':bgr,'bu':bu,'sc':sc,'sod':sod,'pot':pot,
                             'classification':label})

    df = pd.concat([block(n_ckd, True), block(n_ok, False)]).sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, 'id', range(seed*1000, seed*1000+len(df)))
    return df

# user1 — 30% ckd, clear separation, no noise  → high accuracy ~90%
save(make_kidney(150, 1, ckd_frac=0.30,
                 age_ckd=68, age_healthy=28,
                 bp_ckd=100, bp_healthy=68,
                 bgr_ckd=175, bgr_healthy=88,
                 sc_ckd=3.5, sc_healthy=0.7, noise=0.0),
     'sample_data/user1/kidney_disease.csv')

# user2 — 50% ckd, moderate overlap, 15% noise  → medium ~75%
save(make_kidney(150, 2, ckd_frac=0.50,
                 age_ckd=55, age_healthy=45,
                 bp_ckd=88,  bp_healthy=78,
                 bgr_ckd=140, bgr_healthy=110,
                 sc_ckd=2.0, sc_healthy=1.2, noise=0.15),
     'sample_data/user2/kidney_disease.csv')

# user3 — 60% ckd, heavy overlap, 25% noise  → lower ~62%
save(make_kidney(150, 3, ckd_frac=0.60,
                 age_ckd=58, age_healthy=50,
                 bp_ckd=88,  bp_healthy=82,
                 bgr_ckd=138, bgr_healthy=118,
                 sc_ckd=1.8, sc_healthy=1.4, noise=0.25),
     'sample_data/user3/kidney_disease.csv')

# user4 — 80% ckd, very noisy  → ~50-55%
save(make_kidney(150, 4, ckd_frac=0.80,
                 age_ckd=60, age_healthy=52,
                 bp_ckd=90,  bp_healthy=84,
                 bgr_ckd=142, bgr_healthy=122,
                 sc_ckd=1.9, sc_healthy=1.5, noise=0.35),
     'sample_data/user4/kidney_disease.csv')


# ─────────────────────────────────────────────
# HYPERTENSION
# cols: male,age,currentSmoker,cigsPerDay,BPMeds,
#       diabetes,totChol,sysBP,diaBP,BMI,heartRate,glucose,Risk
# ─────────────────────────────────────────────
def make_hypertension(n, seed, risk_frac,
                      age_risk, age_ok,
                      sysBP_risk, sysBP_ok,
                      bmi_risk, bmi_ok,
                      glucose_risk, glucose_ok,
                      noise=0.0):
    np.random.seed(seed)
    n_risk = int(n * risk_frac)
    n_ok   = n - n_risk

    def block(size, is_risk):
        male    = np.random.binomial(1, 0.5, size)
        age     = np.clip(np.random.normal(age_risk if is_risk else age_ok, 9, size).astype(int), 20, 85)
        smoker  = np.random.binomial(1, 0.55 if is_risk else 0.10, size)
        cigs    = np.where(smoker, np.random.randint(5, 30, size), 0)
        bpmeds  = np.random.binomial(1, 0.60 if is_risk else 0.05, size)
        diab    = np.random.binomial(1, 0.45 if is_risk else 0.05, size)
        chol    = np.clip(np.random.normal(255 if is_risk else 195, 22, size).astype(int), 150, 320)
        sysBP   = np.clip(np.random.normal(sysBP_risk if is_risk else sysBP_ok, 12, size).round(1), 100, 200)
        diaBP   = np.clip(np.random.normal(95 if is_risk else 70, 9, size).round(1), 55, 120)
        bmi     = np.clip(np.random.normal(bmi_risk if is_risk else bmi_ok, 5, size).round(2), 16, 50)
        hr      = np.clip(np.random.normal(88 if is_risk else 70, 9, size).astype(int), 50, 120)
        glucose = np.clip(np.random.normal(glucose_risk if is_risk else glucose_ok, 18, size).astype(int), 50, 200)
        risk    = np.array([1]*size if is_risk else [0]*size)
        if noise > 0:
            flip = np.random.binomial(1, noise, size).astype(bool)
            risk[flip] = 1 - risk[flip]
        return pd.DataFrame({'male':male,'age':age,'currentSmoker':smoker,
                             'cigsPerDay':cigs,'BPMeds':bpmeds,'diabetes':diab,
                             'totChol':chol,'sysBP':sysBP,'diaBP':diaBP,
                             'BMI':bmi,'heartRate':hr,'glucose':glucose,'Risk':risk})

    return pd.concat([block(n_risk, True), block(n_ok, False)]).sample(frac=1, random_state=seed).reset_index(drop=True)

# user1 — 25% risk, very clear separation, no noise  → ~95%
save(make_hypertension(150, 1, risk_frac=0.25,
                       age_risk=72, age_ok=26,
                       sysBP_risk=168, sysBP_ok=102,
                       bmi_risk=38, bmi_ok=21,
                       glucose_risk=145, glucose_ok=75, noise=0.0),
     'sample_data/user1/hypertension.csv')

# user2 — 45% risk, moderate separation, 12% noise  → ~78%
save(make_hypertension(150, 2, risk_frac=0.45,
                       age_risk=58, age_ok=40,
                       sysBP_risk=148, sysBP_ok=118,
                       bmi_risk=32, bmi_ok=26,
                       glucose_risk=118, glucose_ok=88, noise=0.12),
     'sample_data/user2/hypertension.csv')

# user3 — 60% risk, overlapping features, 22% noise  → ~65%
save(make_hypertension(150, 3, risk_frac=0.60,
                       age_risk=55, age_ok=48,
                       sysBP_risk=140, sysBP_ok=128,
                       bmi_risk=30, bmi_ok=27,
                       glucose_risk=108, glucose_ok=95, noise=0.22),
     'sample_data/user3/hypertension.csv')

# user4 — 75% risk, very noisy  → ~52%
save(make_hypertension(150, 4, risk_frac=0.75,
                       age_risk=57, age_ok=50,
                       sysBP_risk=138, sysBP_ok=128,
                       bmi_risk=31, bmi_ok=28,
                       glucose_risk=112, glucose_ok=98, noise=0.32),
     'sample_data/user4/hypertension.csv')

print("\nAll done.")
