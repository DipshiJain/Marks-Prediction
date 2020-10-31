import tkinter as tk
import tkinter.font as font

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("TKAgg")

s_data = pd.read_csv("http://bit.ly/w-data")

def predicted_score(event):
    hour = float(hour_input.get())
    if (hour < 0.0) or (hour > 24.0):
        result_predicted_score.configure(text = "There are 24 hours in a day")
    else:
        X_Marks = np.array(s_data['Hours']).reshape(-1, 1)   
        y = np.array(s_data['Scores']).reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(X_Marks, y, test_size=0.2, random_state=0) 
        linear_regressor = LinearRegression()  
        linear_regressor.fit(X_train, y_train) 
        line_Marks = linear_regressor.coef_*X_Marks+ linear_regressor.intercept_
        y_pred = linear_regressor.predict(X_test)
        marks = linear_regressor.predict([[hour]])
        if (float(marks) > 100):
            marks = 99.0
        fig = plt.figure(figsize = (5,4),dpi = 100)
        graph = fig.add_subplot(111)
        graph.plot(X_Marks, line_Marks)  
        graph.title.set_text('Hours vs Percentage')  
        graph.set_xlabel('Hours Studied')  
        graph.set_ylabel('Percentage Score')  
        canvas = FigureCanvasTkAgg(fig, master = window_marks_predict)  # A tk.DrawingArea.
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)

        toolbar = NavigationToolbar2Tk(canvas, window_marks_predict)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)

        result_predicted_score.configure(text = "If you study for " + str(hour)+ " hours per day, you will score {0:.2f}%".format(float(marks)))

def marks_predict():
    global hour_input
    global result_predicted_score 
    global window_marks_predict
    window_marks_predict = tk.Toplevel(window)
    window_marks_predict.geometry("700x600")
    tk.Label(window_marks_predict, text="Marks Predictor", justify= tk.CENTER, font = font.Font(size = 32,weight='bold')).pack()
    frame_marks_predict = tk.LabelFrame(window_marks_predict, padx = 5, pady = 5)
    frame_marks_predict.pack(padx = 8, pady = 8)
    tk.Label(frame_marks_predict, text="Predict Percentage", justify= tk.LEFT, font = font.Font(size = 20,weight='bold')).pack()
    hour_input = tk.Entry(frame_marks_predict)
    hour_input.bind("<Return>",predicted_score)
    hour_input.pack()
    result_predicted_score = tk.Label(frame_marks_predict)
    result_predicted_score.pack()
    
def predicted_hour(event):
    percentage = int(percent_input.get())
    if (percentage < 0) or (percentage > 100):
        result_predicted_score.configure(text = "The percentage value is out of range")
    else:
        X_Hours = np.array(s_data['Scores']).reshape(-1, 1) 
        y = np.array(s_data['Hours']).reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(X_Hours, y, test_size=0.2, random_state=0) 
        linear_regressor = LinearRegression()  
        linear_regressor.fit(X_train, y_train) 
        line_hours = linear_regressor.coef_*X_Hours+ linear_regressor.intercept_
        y_pred = linear_regressor.predict(X_test)
        hour = linear_regressor.predict([[percentage]])
        
        fig = plt.figure(figsize = (5,4),dpi = 100)
        graph = fig.add_subplot(111)
        graph.plot(X_Hours, line_hours)  
        graph.set_title('Hours vs Percentage')  
        graph.set_ylabel('Hours Studied')  
        graph.set_xlabel('Percentage Score')  
        canvas = FigureCanvasTkAgg(fig, master = window_hours_predict)  # A tk.DrawingArea.
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(canvas, window_hours_predict)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)
        
        result_predicted_hour.configure(text = "If you want to score " + str(percentage) + "% you should study for {0:.2f} Hours per day".format(float(hour)))

def hours_predict():
    global percent_input
    global result_predicted_hour 
    global window_hours_predict
    window_hours_predict = tk.Toplevel(window)
    window_hours_predict.geometry("700x600")
    tk.Label(window_hours_predict, text="Marks Predictor", justify= tk.CENTER, font = font.Font(size = 32,weight='bold')).pack()
    frame_hours_predict = tk.LabelFrame(window_hours_predict, padx = 5, pady = 5)
    frame_hours_predict.pack(padx = 8, pady = 8)
    tk.Label(frame_hours_predict, text="Predict Hour", justify= tk.CENTER, font = font.Font(size = 20,weight='bold')).pack()
    percent_input = tk.Entry(frame_hours_predict)
    percent_input.bind("<Return>",predicted_hour)
    percent_input.pack()
    result_predicted_hour = tk.Label(frame_hours_predict)
    result_predicted_hour.pack()

window = tk.Tk()
window.geometry("400x150")
tk.Label(window, text="Marks Predictor", justify= tk.CENTER, font = font.Font(size = 32,weight='bold')).pack()
button1 = tk.Button(window,text = "Predict Percentage", width = 15, command = marks_predict)
button1.pack(padx = 10, pady = 10)
button2 = tk.Button(window,text = "Predict Hour", width = 15, command = hours_predict)
button2.pack(padx = 10, pady = 10)
window.mainloop()
