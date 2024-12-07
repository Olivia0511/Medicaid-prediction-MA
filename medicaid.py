import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the updated Medicaid data from the uploaded file
file_path = 'Medicaid.xlsx'  # Replace with your file path
medicaid_data = pd.ExcelFile(file_path)
data = medicaid_data.parse('Sheet1')

# Extract and clean data
expenditure = data.iloc[0, 1:].dropna().values.astype(float)  # Expenditure data (2014 onward)
enrollment = data.iloc[3, 1:].dropna().values.astype(float)  # Enrollment data (2017 onward)

# Years for expenditure and enrollment
expenditure_years = np.arange(2014, 2014 + len(expenditure)).reshape(-1, 1)
enrollment_years = np.arange(2017, 2017 + len(enrollment)).reshape(-1, 1)

# Future years (2024-2033) for predictions
future_years = np.arange(2024, 2034).reshape(-1, 1)

# Build regression models
expenditure_model = LinearRegression().fit(expenditure_years, expenditure)
enrollment_model = LinearRegression().fit(enrollment_years, enrollment)

# Predict next 10 years (2024-2033) for both enrollment and expenditure
predicted_expenditure = expenditure_model.predict(future_years)
predicted_enrollment = enrollment_model.predict(future_years)

# Combine historical and forecast data
all_expenditure_years = np.append(expenditure_years.flatten(), future_years.flatten())
all_enrollment_years = np.append(enrollment_years.flatten(), future_years.flatten())
all_expenditure = np.append(expenditure, predicted_expenditure)
all_enrollment = np.append(enrollment, predicted_enrollment)

# Plotting the Enrollment Graph
plt.figure(figsize=(10, 6))
plt.plot(all_enrollment_years[:len(enrollment)], enrollment, label="Historical Enrollment (2017-2023)", color="blue", linestyle="-")
plt.plot(all_enrollment_years[len(enrollment):], predicted_enrollment, label="Predicted Enrollment (2024-2033)", color="blue", linestyle="--")
plt.xticks(np.arange(2013, 2034, 2))  # Ensure year labels are integers
plt.title("Medicaid Enrollment Forecast (2013-2033)", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Enrollment (in millions)", fontsize=12)
plt.legend()
plt.grid(True)
plt.show()  # Ensure this plot is displayed before moving to the next

# Plotting the Expenditure Graph
plt.figure(figsize=(10, 6))
plt.plot(all_expenditure_years[:len(expenditure)], expenditure, label="Historical Expenditure (2014-2023)", color="green", linestyle="-")
plt.plot(all_expenditure_years[len(expenditure):], predicted_expenditure, label="Predicted Expenditure (2024-2033)", color="green", linestyle="--")
plt.xticks(np.arange(2013, 2034, 2))  # Ensure year labels are integers
plt.title("Medicaid Expenditure Forecast (2013-2033)", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Expenditure (in billions)", fontsize=12)
plt.legend()
plt.grid(True)
plt.show()  # Ensure this plot is displayed
