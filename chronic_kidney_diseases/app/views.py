from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.conf import settings
from functools import wraps
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import seaborn as sns
import io
import urllib, base64

plt.switch_backend('Agg')

from .models import User


def login_required(view_func):
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.session.get('username'):
            return redirect('login')
        return view_func(request, *args, **kwargs)
    return wrapper


def landing(request):
    return render(request, 'index.html')


def login(request):
    if request.session.get('username'):
        return redirect('home')
    response = render(request, 'loginform.html')
    response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response['Pragma'] = 'no-cache'
    response['Expires'] = '0'
    return response


def loginCheck(request):
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('email')
        try:
            user_object = User.objects.get(firstname=username, password=password)
        except User.DoesNotExist:
            user_object = None

        if user_object:
            request.session['useremail'] = user_object.email
            request.session['username'] = user_object.firstname
            return redirect('home')
        else:
            return render(request, 'loginform.html', {'error': 'Invalid username or password'})

    return redirect('login')


def logout(request):
    request.session.flush()
    response = redirect('login')
    response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response['Pragma'] = 'no-cache'
    response['Expires'] = '0'
    return response


def reg(request):
    return render(request, 'register.html')


def save(request):
    if request.method == 'POST':
        user = User(
            firstname=request.POST.get('username'),
            password=request.POST.get('password'),
            address=request.POST.get('address'),
            email=request.POST.get('email'),
            age=request.POST.get('age'),
            gender=request.POST.get('gender'),
            phone=request.POST.get('phone')
        )
        user.save()
        return render(request, 'loginform.html')
    return render(request, 'loginform.html')


@login_required
def home(request):
    response = render(request, 'home1.html')
    response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response['Pragma'] = 'no-cache'
    response['Expires'] = '0'
    return response


@login_required
def nvb(request):
    return render(request, 'pacweb1.html')


@login_required
def svm(request):
    return render(request, 'cvd.html')


@login_required
def dec(request):
    if request.method == 'POST':
        values = [float(request.POST.get(f'headline{i}') or 0) for i in range(1, 14)]
        csv_path = os.path.join(settings.BASE_DIR, 'data', 'cvd.csv')
        df = pd.read_csv(csv_path).fillna(0)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict([values])[0]
        acc = model.score(X_train, y_train)
        return render(request, 'result.html', {'predictedvalue': pred, 'accuracy': acc})
    return render(request, 'decisiontree.html')


@login_required
def mnb(request):
    return render(request, 'naivebayes.html')


@login_required
def graph(request):
    error = None
    if request.method == 'POST' and request.FILES.get('csv_file'):
        csv_file = request.FILES['csv_file']
        try:
            df = pd.read_csv(csv_file).dropna()
            if 'Risk' not in df.columns:
                error = "CSV must contain a 'Risk' column as the target."
            else:
                X = df.drop('Risk', axis=1)
                y = df['Risk']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                acc = round(model.score(X_test, y_test) * 100, 2)

                plt.figure(figsize=(6, 4))
                sns.countplot(x='Risk', data=df)
                plt.title('Hypertension Risk Distribution')
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                uri = urllib.parse.quote(base64.b64encode(buf.read()))
                plt.close()

                return render(request, 'graph.html', {'graph': uri, 'accuracy': acc})
        except Exception as e:
            error = f"Error processing file: {str(e)}"

    return render(request, 'graph.html', {'error': error})


@login_required
def pac(request):
    if request.method == 'POST':
        values = [float(request.POST.get(f'headline{i}') or 0) for i in range(1, 9)]
        csv_path = os.path.join(settings.BASE_DIR, 'data', 'diabetes.csv')
        df = pd.read_csv(csv_path).fillna(0)
        X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
        y = df['Outcome']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict([values])[0]
        acc = model.score(X_train, y_train)
        result = "Diabetes" if pred == 1 else "Not Diabetes"
        return render(request, 'result.html', {'predictedvalue': result, 'accuracy': acc})
    return render(request, 'pacweb1.html')


@login_required
def accuracy(request):
    error = None
    if request.method == 'POST' and request.FILES.get('csv_file'):
        csv_file = request.FILES['csv_file']
        try:
            df = pd.read_csv(csv_file)

            # encode categoricals before fillna so strings aren't replaced with 0
            le = LabelEncoder()
            for col in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
                if col in df.columns:
                    df[col] = le.fit_transform(df[col].astype(str).str.strip())

            df = df.fillna(0)

            required_cols = ['gender', 'age', 'hypertension', 'heart_disease',
                             'ever_married', 'work_type', 'Residence_type',
                             'avg_glucose_level', 'bmi', 'smoking_status', 'stroke']
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                error = f"Missing columns: {', '.join(missing)}"
            else:
                feature_cols = ['gender', 'age', 'hypertension', 'heart_disease',
                                'ever_married', 'work_type', 'Residence_type',
                                'avg_glucose_level', 'bmi', 'smoking_status']
                # include id if present
                if 'id' in df.columns:
                    feature_cols = ['id'] + feature_cols
                X = df[feature_cols]
                y = df['stroke']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = AdaBoostClassifier(
                    DecisionTreeClassifier(max_depth=5),
                    n_estimators=200, learning_rate=0.01, random_state=42
                )
                model.fit(X_train, y_train)
                acc = round(accuracy_score(y_test, model.predict(X_test)) * 100, 2)

                # generate stroke distribution graph
                plt.figure(figsize=(6, 4))
                sns.countplot(x='stroke', data=df)
                plt.title('Stroke Distribution')
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                uri = urllib.parse.quote(base64.b64encode(buf.read()))
                plt.close()

                return render(request, 'acc1.html', {'accuracy': acc, 'disease': 'Stroke', 'graph': uri})
        except Exception as e:
            error = f"Error processing file: {str(e)}"

    return render(request, 'acc1.html', {
        'error': error, 'disease': 'Stroke',
        'columns': 'id (optional), gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status, stroke'
    })


@login_required
def randomf(request):
    error = None
    if request.method == 'POST' and request.FILES.get('csv_file'):
        csv_file = request.FILES['csv_file']
        try:
            df = pd.read_csv(csv_file)

            # strip whitespace from all string columns
            for col in df.select_dtypes(include='object').columns:
                df[col] = df[col].astype(str).str.strip()

            required_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'classification']
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                error = f"Missing columns: {', '.join(missing)}"
            else:
                feature_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot']
                if 'id' in df.columns:
                    feature_cols = ['id'] + feature_cols

                # convert to numeric, coerce bad values to NaN then fill 0
                X = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
                y = df['classification']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = MultinomialNB()
                model.fit(X_train, y_train)
                acc = round(accuracy_score(y_test, model.predict(X_test)) * 100, 2)

                # generate classification distribution graph
                plt.figure(figsize=(6, 4))
                sns.countplot(x='classification', data=df)
                plt.title('Kidney Disease Classification Distribution')
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                uri = urllib.parse.quote(base64.b64encode(buf.read()))
                plt.close()

                return render(request, 'acc1.html', {'accuracy': acc, 'disease': 'Kidney Disease', 'graph': uri})
        except Exception as e:
            error = f"Error processing file: {str(e)}"

    return render(request, 'acc1.html', {
        'error': error, 'disease': 'Kidney Disease',
        'columns': 'id (optional), age, bp, sg, al, su, bgr, bu, sc, sod, pot, classification'
    })
