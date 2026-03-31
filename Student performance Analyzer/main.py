from model import train_models

reg_model, clf_model = train_models()

print("Enter Student Details:")
hours = float(input("Study Hours: "))
attendance = float(input("Attendance: "))
previous_marks = float(input("Previous Marks: "))

data = [[hours, attendance, previous_marks]]

# Prediction
predicted_marks = reg_model.predict(data)
result = clf_model.predict(data)

print("\n Predicted Marks:", predicted_marks[0])

if result[0] == 1:
    print(" Result: PASS")
else:
    print(" Result: FAIL")