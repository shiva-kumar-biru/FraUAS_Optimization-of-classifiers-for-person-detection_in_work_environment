import io

import boto3
import matplotlib.pyplot as plt

import panel as pn
import seaborn as sns
from PIL import Image

from dataprocessing_of_gui import (
    file_processing,
    gbm,
    lr,
    rfc,
    svm,
    xgboost,
)


custom_style = {
    "padding": "14px",
    "font-style": "italic",
    "border-radius": "12px",
    "width": "180px",
}

style_to_display = {
    "padding": "14px",
    "font-style": "italic",
    "border-radius": "12px",
    "font-size": "13pt",
    "width": "auto",
    "font-weight": "bold",
}


# Initialize S3 client
s3 = boto3.client("s3")

bucket_name = "individualprojectdataset"
s3_key = "Images/university_logo.png"

response = s3.get_object(Bucket=bucket_name, Key=s3_key)
image_data = response["Body"].read()

image = Image.open(io.BytesIO(image_data))
logo = pn.pane.Image(io.BytesIO(image_data), width=150, height=150)

global_data = {
    "normalized_data": None,
    "test_labels_df": None,
    "classifier_accuracies": {},
}


def run_classifier(event):
    if (
        global_data["normalized_data"] is None
        or global_data["test_labels_df"] is None
    ):
        status_message.object = (
            "Please upload a file and process it before running the classifier."
        )
        return

    selected_classifier = classifier_select.value
    normalized_data = global_data["normalized_data"]
    test_labels_df = global_data["test_labels_df"]

    if selected_classifier == "Random Forest":
        accuracy, precision, recall, f1, cm = rfc(
            normalized_data, test_labels_df
        )
        title = "RFC"
    elif selected_classifier == "SVM":
        accuracy, precision, recall, f1, cm = svm(
            normalized_data, test_labels_df
        )
        title = "SVM"
    elif selected_classifier == "Logistic Regression":
        accuracy, precision, recall, f1, cm = lr(
            normalized_data, test_labels_df
        )
        title = "Logistic Regression"
    elif selected_classifier == "XG Boost":
        accuracy, precision, recall, f1, cm = xgboost(
            normalized_data, test_labels_df
        )
        title = "XG Boost"
    elif selected_classifier == "Gradient Boosting":
        accuracy, precision, recall, f1, cm = gbm(
            normalized_data, test_labels_df
        )
        title = "Gradient Boosting"
    else:
        status_message.object = "No classifier selected."
        return

    row1.clear()
    row1.append(cm)

    metrics_placeholder.object = f"""
    <div style="font-size:16px; font-weight:bold; color:#333;">
        <h3>Metrics for {title}</h3>
        <ul>
            <li>Accuracy: <span style="color:blue;">{accuracy:.4f}</span></li>
            <li>Precision: <span style="color:green;">{precision:.4f}</span></li>
            <li>Recall: <span style="color:orange;">{recall:.4f}</span></li>
            <li>F1-score: <span style="color:red;">{f1:.4f}</span></li>
        </ul>
    </div>
    """

    global_data["classifier_accuracies"][selected_classifier] = accuracy


file_path_field = pn.widgets.FileInput(name="file", accept=".csv")

file_upload_button = pn.widgets.Button(
    name="Upload", button_type="primary", styles=custom_style
)

loading_spinner = pn.indicators.LoadingSpinner(
    value=False, width=50, height=50
)

status_message = pn.pane.Markdown("")


def upload_button_click(event):
    if file_path_field.value is None:
        status_message.object = "No file uploaded."
        return

    compare_accuracies_column.clear()
    loading_spinner.value = True
    status_message.object = ""

    try:
        out = io.BytesIO()
        file_path_field.save(out)
        out.seek(0)
        content = out.getvalue()

        if not content:
            raise ValueError("The uploaded file is empty.")

        normalized_data, test_labels_df = file_processing(
            io.BytesIO(content)
        )

        global_data["normalized_data"] = normalized_data
        global_data["test_labels_df"] = test_labels_df

        status_message.object = "Data processing is done."
    except Exception as exc:
        status_message.object = f"Error: {exc}"
    finally:
        loading_spinner.value = False


file_upload_button.on_click(upload_button_click)

classifier_select = pn.widgets.Select(
    name="Select classifier",
    options=[
        "Random Forest",
        "Gradient Boosting",
        "SVM",
        "Logistic Regression",
        "XG Boost",
    ],
    width=300,
)

run_button = pn.widgets.Button(
    name="Run", button_type="primary", styles=custom_style
)

compare_button = pn.widgets.Button(
    name="Compare Accuracies",
    button_type="success",
    styles=custom_style,
)


def compare_accuracies(event):
    if not global_data["classifier_accuracies"]:
        status_message.object = "No classifiers have been run yet."
        return

    classifiers = list(global_data["classifier_accuracies"].keys())
    accuracies = list(global_data["classifier_accuracies"].values())
    accuracies_percentage = [acc * 100 for acc in accuracies]

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=classifiers,
        y=accuracies_percentage,
        palette="viridis",
    )

    plt.xlabel("Classifiers")
    plt.ylabel("Accuracy (%)")
    plt.title("Comparison of Classifier Accuracies")
    plt.ylim(0, 100)

    for i, acc in enumerate(accuracies_percentage):
        plt.text(
            i,
            acc + 1,
            f"{acc:.2f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    accuracy_comparison_pane.object = plt.gcf()
    accuracy_comparison_pane.width = 500
    accuracy_comparison_pane.height = 550

    compare_accuracies_column.clear()
    compare_accuracies_column.append(accuracy_comparison_pane)

    plt.close()


compare_button.on_click(compare_accuracies)
run_button.on_click(run_classifier)

conf_matrix_placeholder = pn.pane.Markdown(
    "<div style='font-size:18px;'><b>Confusion Matrix will be displayed here</b></div>"
)

metrics_placeholder = pn.pane.Markdown(
    "<div style='font-size:18px;'><b>Metrics will be displayed here</b></div>"
)

accuracy_comparison_pane = pn.pane.Matplotlib()

row1 = pn.Row(conf_matrix_placeholder, width=509)
row2 = pn.Row(metrics_placeholder, width=300)

main_column = pn.Column(row1, row2)

compare_accuracies_column = pn.Column(
    pn.pane.Markdown(
        "<div style='font-size:18px;'><b>Here we display the comparison of accuracies</b></div>"
    ),
    width=500,
)

logo_column = pn.Column(
    logo, sizing_mode="fixed", width=200, height=100
)

main = pn.Row(
    main_column,
    compare_accuracies_column,
    pn.layout.HSpacer(),
    pn.Column(
        logo_column,
        sizing_mode="fixed",
        width=120,
        height=120,
        styles={"position": "fixed", "top": "30px", "right": "40px"},
    ),
    sizing_mode="stretch_width",
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
        styles={"width": "100%", "padding": "15px"},
    ),
    main=[main],
    header_background="#6424db",
    site="Frankfurt university of applied sciences",
    sidebar_width=350,
    busy_indicator=None,
)

template.servable()
