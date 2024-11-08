import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np

# Load the model
model = joblib.load('SVM_label.pkl')

# GUI Application Class
class DiscHerniationPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Post-Surgery Sensory Improvement Predictor")

        # Feature names
        self.entries = {}
        self.feature_names = [
            'PfirrmannGrade',
            'ResNet50-sagittal-DTL1864',
            'ResNet50-sagittal-DTL1013',
            'ResNet50-sagittal-DTL1176',
            'ResNet50-sagittal-DTL355',
            'ResNet50-transverse-DTL1058',
            'Osteoporosis',
            'ResNet50-transverse-DTL1701',
            'ResNet50-transverse-DTL1360',
            'ResNet50-transverse-DTL472',
            'ResNet50-transverse-DTL1869',
            'ResNet50-transverse-DTL114',
            'ResNet50-sagittal-DTL1087',
            'ResNet50-sagittal-DTL1243',
            'ResNet50-transverse-DTL503',
            'ResNet50-transverse-DTL970',
            'ResNet50-transverse-DTL834',
            'ResNet50-sagittal-DTL1397',
            'Hypertension',
            'ResNet50-sagittal-DTL1236'
        ]

        # Create input fields and dropdown menus
        input_frame = tk.Frame(root)
        input_frame.grid(row=0, column=0, padx=10, pady=10)

        # Osteoporosis: Dropdown selection
        self.create_label_and_optionmenu(input_frame, 'Osteoporosis', 'Osteoporosis (0=No, 1=Yes):', {0: 'No (0)', 1: 'Yes (1)'}, 0)

        # Hypertension: Dropdown selection
        self.create_label_and_optionmenu(input_frame, 'Hypertension', 'Hypertension (0=No, 1=Yes):', {0: 'No (0)', 1: 'Yes (1)'}, 1)

        # Pfirrmann Grade: Dropdown selection
        cp_options = {
            1: 'I',
            2: 'II',
            3: 'III',
            4: 'IV',
            5: 'V',
        }
        self.create_label_and_optionmenu(input_frame, 'PfirrmannGrade', 'Pfirrmann Grade:', cp_options, 2)

        # Numeric input fields for the remaining features
        row_index = 3
        for feature_name in self.feature_names[1:]:  # Start from 1 to skip the first dropdown option
            if feature_name not in ["Osteoporosis", "Hypertension", "PfirrmannGrade"]:  # Avoid duplicates
                self.create_label_and_entry(input_frame, feature_name, f"{feature_name}:", row_index)
                row_index += 1

        # Predict button
        predict_button = tk.Button(root, text="Predict", command=self.predict)
        predict_button.grid(row=1, column=0, pady=10)

        # Text box for displaying results
        self.result_text = tk.Text(root, height=10, width=50)
        self.result_text.grid(row=2, column=0, padx=10, pady=10)

    def create_label_and_entry(self, frame, feature_name, label_text, row):
        """Create a label and entry widget."""
        label = tk.Label(frame, text=label_text)
        label.grid(row=row, column=0, padx=10, pady=5)
        entry = tk.Entry(frame)
        entry.grid(row=row, column=1, padx=10, pady=5)
        self.entries[feature_name] = entry

    def create_label_and_optionmenu(self, frame, feature_name, label_text, options, row):
        """Create a label and option menu widget."""
        label = tk.Label(frame, text=label_text)
        label.grid(row=row, column=0, padx=10, pady=5)
        var = tk.StringVar(frame)
        var.set(list(options.keys())[0])  # Set default value
        optionmenu = tk.OptionMenu(frame, var, *options.keys())
        optionmenu.grid(row=row, column=1, padx=10, pady=5)
        self.entries[feature_name] = var

    def predict(self):
        try:
            # Get user input features
            feature_values = []
            for feature in self.feature_names:
                value = self.entries[feature].get()
                feature_values.append(float(value))
            features = np.array([feature_values])

            # Make prediction
            predicted_class = model.predict(features)[0]
            predicted_proba = model.predict_proba(features)[0]

            # Generate prediction results and advice
            result = f"Predicted Class: {predicted_class}\n"
            result += f"Prediction Probabilities: {predicted_proba}\n"
            advice = self.generate_advice(predicted_class, predicted_proba)
            result += advice

            # Display prediction results and advice
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, result)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
    
    def generate_advice(self, predicted_class, predicted_proba):
        """Generate advice based on the prediction result."""
        probability = predicted_proba[predicted_class] * 100

        if predicted_class == 1:
            advice = (
                f"\nBased on our model's prediction, your numbness improvement after the surgery is significant. "
                f"The model predicts a {probability:.1f}% likelihood of significant postoperative improvement in numbness."
            )
        else:
            advice = (
                f"\nBased on our model's prediction, your numbness improvement after the surgery is limited. "
                f"The model predicts a {probability:.1f}% likelihood of limited improvement."
            )

        return advice

# Create the main application window
root = tk.Tk()
app = DiscHerniationPredictorApp(root)

# Run the application
root.mainloop()
