import matplotlib.pyplot as plt

# Data
classifiers = ['Random Forest', 'SVM', 'Logistic Regression', 'XG Boost', 'Gradient Boosting']
accuracies = [97.4, 98.1, 96.4, 98, 97.6]

# Create bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(classifiers, accuracies, color='skyblue')
plt.ylim(90,100)  # Adjust y-axis for better visibility
plt.xlabel('Classifiers')
plt.ylabel('Accuracy (%)')
plt.title('Classifier Accuracies')

# Add percentage above each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}%', ha='center', va='bottom')

plt.show()