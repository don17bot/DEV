# Import libraries
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, plot_model

# -------------------------------
# 1️⃣ Load the Iris dataset
# -------------------------------
iris = load_iris()
X = iris.data                      # Features (sepal & petal measurements)
y = to_categorical(iris.target)    # One-hot encoded labels (3 classes)

# -------------------------------
# 2️⃣ Split the dataset
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 3️⃣ Standardize the data
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# 4️⃣ Define the Base Model
# -------------------------------
model = Sequential([
    Dense(10, input_shape=(4,), activation='relu'),  # Hidden layer 1
    Dense(10, activation='relu'),                    # Hidden layer 2
    Dense(3, activation='softmax')                   # Output layer (3 classes)
])

# -------------------------------
# 5️⃣ Compile the Model
# -------------------------------
model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# 6️⃣ Train the Model
# -------------------------------
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# -------------------------------
# 7️⃣ Experiment: Different Batch Sizes
# -------------------------------
for batch_size in [16, 32, 64]:
    print(f"\nTraining with batch size: {batch_size}")
    model.fit(X_train, y_train, epochs=50, batch_size=batch_size, verbose=1)

# -------------------------------
# 8️⃣ Experiment: Vary Hidden Layers
# -------------------------------
for layers in [1, 2, 3]:
    print(f"\nTraining with {layers} hidden layers")
    model = Sequential()
    model.add(Dense(10, input_shape=(4,), activation='relu'))
    for _ in range(layers - 1):
        model.add(Dense(10, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.01),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# -------------------------------
# 9️⃣ Experiment: Vary Learning Rates
# -------------------------------
for lr in [0.001, 0.01, 0.1]:
    print(f"\nTraining with learning rate: {lr}")
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# -------------------------------
# 🔟 Evaluate Model
# -------------------------------
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# -------------------------------
# 1️⃣1️⃣ Visualize Model Architecture
# -------------------------------
plot_model(model, show_shapes=True,
           show_layer_names=True, to_file='model.png')

# -------------------------------
# 1️⃣2️⃣ Print number of neurons in each layer
# -------------------------------
print("\nNeurons in each layer of the model:")
for layer in model.layers:
    if hasattr(layer, 'units'):
        print(f"Layer {layer.name}: {layer.units} neurons")
