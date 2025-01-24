import panel as pn
import io
from dataprocessing_of_gui import file_processing,rfc,gbm,svm,xgboost,lr
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

custom_style = {
    'padding': '14px',
    'font-style': 'italic',
    'border-radius': '12px',
    'width': '180px'
}

style_to_display= {
    'padding': '14px',
    'font-style': 'italic',
    'border-radius': '12px',
    'font-size': '13pt',
    'width': 'auto',
    'font-weight': 'bold'
}

logopath=r"//Users//shivakumarbiru//Desktop//individual_project//Images//university_logo.png"
logo = pn.pane.Image(logopath, width=150, height=150)
global_data = {"normalized_data": None, "test_labels_df": None,
                   "classifier_accuracies": {}}

def run_classifier(event):
    if global_data["normalized_data"] is not None and global_data["test_labels_df"] is not None:
        selected_classifier = classifier_select.value
        if selected_classifier == "Random Forest":
            normalized_data = global_data["normalized_data"]
            test_labels_df = global_data["test_labels_df"]
            accuracy_direct, precision_direct, recall_direct, f1_direct, CM_rfc = rfc(normalized_data, test_labels_df)
            row1.clear()
            row1.append(CM_rfc)
            metrics_placeholder.object = f"""
            <div style="font-size:16px; font-weight:bold; color:#333;">
            <h3>Metrics for RFC</h3>
            <ul>
                <li>Accuracy: <span style="color:blue;">{accuracy_direct:.4f}</span></li>
                <li>Precision: <span style="color:green;">{precision_direct:.4f}</span></li>
                <li>Recall: <span style="color:orange;">{recall_direct:.4f}</span></li>
                <li>F1-score: <span style="color:red;">{f1_direct:.4f}</span></li>
            </ul>
            </div>
            """
        elif selected_classifier == "SVM":
            normalized_data = global_data["normalized_data"]
            test_labels_df = global_data["test_labels_df"]
            accuracy_direct, precision_direct, recall_direct, f1_direct, cm_svm = svm(normalized_data, test_labels_df)
            row1.clear()
            row1.append(cm_svm)
            metrics_placeholder.object = f"""
            <div style="font-size:16px; font-weight:bold; color:#333;">
            <h3>Metrics for SVM</h3>
            <ul>
                <li>Accuracy: <span style="color:blue;">{accuracy_direct:.4f}</span></li>
                <li>Precision: <span style="color:green;">{precision_direct:.4f}</span></li>
                <li>Recall: <span style="color:orange;">{recall_direct:.4f}</span></li>
                <li>F1-score: <span style="color:red;">{f1_direct:.4f}</span></li>
            </ul>
            </div>
            """
        elif selected_classifier == "Logistic Regression":
            normalized_data = global_data["normalized_data"]
            test_labels_df = global_data["test_labels_df"]
            accuracy_direct, precision_direct, recall_direct, f1_direct, cm_lr = lr(normalized_data, test_labels_df)
            row1.clear()
            row1.append(cm_lr)
            metrics_placeholder.object = f"""
            <div style="font-size:16px; font-weight:bold; color:#333;">
            <h3>Metrics for Logistic Regression</h3>
            <ul>
                <li>Accuracy: <span style="color:blue;">{accuracy_direct:.4f}</span></li>
                <li>Precision: <span style="color:green;">{precision_direct:.4f}</span></li>
                <li>Recall: <span style="color:orange;">{recall_direct:.4f}</span></li>
                <li>F1-score: <span style="color:red;">{f1_direct:.4f}</span></li>
            </ul>
            </div>
            """
        elif selected_classifier == "XG Boost":
            normalized_data = global_data["normalized_data"]
            test_labels_df = global_data["test_labels_df"]
            accuracy_direct, precision_direct, recall_direct, f1_direct, cm_xg = xgboost(normalized_data, test_labels_df)
            row1.clear()
            row1.append(cm_xg)
            metrics_placeholder.object = f"""
            <div style="font-size:16px; font-weight:bold; color:#333;">
            <h3>Metrics for XG Boost</h3>
            <ul>
                <li>Accuracy: <span style="color:blue;">{accuracy_direct:.4f}</span></li>
                <li>Precision: <span style="color:green;">{precision_direct:.4f}</span></li>
                <li>Recall: <span style="color:orange;">{recall_direct:.4f}</span></li>
                <li>F1-score: <span style="color:red;">{f1_direct:.4f}</span></li>
            </ul>
            </div>
            """
        elif selected_classifier == "Gradient Boosting":
            normalized_data = global_data["normalized_data"]
            test_labels_df = global_data["test_labels_df"]
            accuracy_direct, precision_direct, recall_direct, f1_direct, cm_gbm = gbm(normalized_data, test_labels_df)
            row1.clear()
            row1.append(cm_gbm)
            metrics_placeholder.object = f"""
            <div style="font-size:16px; font-weight:bold; color:#333;">
            <h3>Metrics for Gradient Boosting</h3>
            <ul>
                <li>Accuracy: <span style="color:blue;">{accuracy_direct:.4f}</span></li>
                <li>Precision: <span style="color:green;">{precision_direct:.4f}</span></li>
                <li>Recall: <span style="color:orange;">{recall_direct:.4f}</span></li>
                <li>F1-score: <span style="color:red;">{f1_direct:.4f}</span></li>
            </ul>
            </div>
          """

        else:
            status_message.object = "No classifier selected."
        global_data["classifier_accuracies"][selected_classifier] = accuracy_direct
    else:
        status_message.object = "Please upload a file and process it before running the classifier."

file_path_field=pn.widgets.FileInput(name="file", accept=".csv",)

file_upload_button=pn.widgets.Button(name="Upload", button_type='primary',styles=custom_style)

loading_spinner = pn.indicators.LoadingSpinner(value=False, width=50, height=50)

status_message = pn.pane.Markdown("")

def upload_button_click(event):
    if file_path_field.value is not None:
        compare_accuracies_column.clear()
        loading_spinner.value = True
        status_message.object = ""
        try:
            print("Reading file content...")
            out = io.BytesIO()
            file_path_field.save(out)
            out.seek(0)
            content = out.getvalue()
            print("File content preview:")
            print(content[:500])                                                       
            if len(content) == 0:
                raise ValueError("The uploaded file is empty.")
            try:
                df = pd.read_csv(io.StringIO(content.decode('utf-8')))
                print("CSV file preview:")
                print(df.head())
            except Exception as e:
                raise ValueError(f"Could not parse the file as a CSV: {e}")
            normalized_data, test_labels_df = file_processing(io.BytesIO(content))
            global_data["normalized_data"] = normalized_data
            global_data["test_labels_df"] = test_labels_df
            print("File content read successfully. Processing file...")
            loading_spinner.value = False
            status_message.object = "Data processing is done."
            print("Data processing completed.")
        except Exception as e:
            loading_spinner.value = False
            status_message.object = f"Error: {e}"
            print(f"Error during file processing: {e}")
    else:
        status_message.object = "No file uploaded."
        print("No file uploaded.")

# Attach the callback to the upload button

file_upload_button.on_click(upload_button_click)

classifier_select = pn.widgets.Select(name="Select classifier", options=["Random Forest", "Gradient Boosting", "SVM", "Logistic Regression", "XG Boost"], width=300,)
run_button = pn.widgets.Button(name="Run",button_type='primary',styles=custom_style)
compare_button = pn.widgets.Button(name="Compare Accuracies", button_type='success', styles=custom_style)


def compare_accuracies(event):
    if global_data["classifier_accuracies"]:
        classifiers = list(global_data["classifier_accuracies"].keys())
        accuracies = list(global_data["classifier_accuracies"].values())
        
        # Convert accuracies to percentage
        accuracies_percentage = [acc * 100 for acc in accuracies]
        
        plt.figure(figsize=(10, 6))  # Adjust figure size as needed
        sns.barplot(x=classifiers, y=accuracies_percentage, palette="viridis")
        
        plt.xlabel("Classifiers")
        plt.ylabel("Accuracy (%)")
        plt.title("Comparison of Classifier Accuracies")
        plt.ylim(0, 100)
        
        # Display accuracy percentage on top of each bar
        for i, acc in enumerate(accuracies_percentage):
            plt.text(i, acc + 1, f'{acc:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')
        
        # Display the plot in a Panel pane
        accuracy_comparison_pane.object = plt.gcf()
        accuracy_comparison_pane.width = 500  # Set the width of the pane
        accuracy_comparison_pane.height = 550  # Set the height of the pane
        compare_accuracies_column.clear()
        compare_accuracies_column.append(accuracy_comparison_pane)
        plt.close()
    else:
        status_message.object = "No classifiers have been run yet."

compare_button.on_click(compare_accuracies)



# Attach the classifier runner to the Run button
run_button.on_click(run_classifier)

conf_matrix_placeholder = pn.pane.Markdown("<div style='font-size:18px;'><b>Confusion Matrix will be displayed here</b></div>")
metrics_placeholder = pn.pane.Markdown("<div style='font-size:18px;'><b>Metrics will be displayed here</b></div>")
accuracy_comparison_pane = pn.pane.Matplotlib()
# Main content placeholders
row1 = pn.Row(conf_matrix_placeholder,width=509)
row2 = pn.Row(metrics_placeholder, width=300)

# Main column for content
main_column = pn.Column(row1, row2)
compare_accuracies_column = pn.Column(
    pn.pane.Markdown("<div style='font-size:18px;'><b>Here we display the comparison of accuracies</b></div>"),
 # Make the column stretch to fit its container
    width=500  # Set the width to 500 pixels
)
logo_column = pn.Column(logo, sizing_mode='fixed', width=200, height=100)

main = pn.Row(
        main_column,
        compare_accuracies_column,  # Fix the width of this column
        pn.layout.HSpacer(),  # Adds space between columns and logo
        pn.Column(
            logo_column,
            sizing_mode='fixed',
            width=120,
            height=120,
            styles={'position': 'fixed', 'top': '30px', 'right': '40px'}
        ),
        sizing_mode='stretch_width'  # Ensures the row stretches to fit the container
    )

template = pn.template.BootstrapTemplate(
    title="GUI for Optimization of classifiers for person detection",
    sidebar=pn.Column(
        pn.pane.Markdown("## Enter file path", styles=style_to_display),
        file_path_field,
        file_upload_button,
        loading_spinner,
        status_message,
        pn.pane.Markdown("## Select any classifier", styles=style_to_display),
        classifier_select,
        run_button,
        compare_button,
        styles={"width": "100%", "padding": "15px"}
    ),
    main =[main],
    header_background='#6424db',
    site="Frankfurt University of Applied Sciences",
    sidebar_width=350,
    busy_indicator=None
)
template.servable()


