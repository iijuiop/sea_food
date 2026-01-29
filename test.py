import sys
sys.stdout.reconfigure(encoding='utf-8')
import tensorflow as tf
import numpy as np
import json, os, cv2
import matplotlib.pyplot as plt

# ===== Load model and labels =====
model = tf.keras.models.load_model("seafood_model.keras")

with open("labels.json", "r", encoding="utf-8") as f:
    class_names = json.load(f)

print("Loaded class names:", class_names)  # <-- debug: kiểm tra tên lớp

# --- NEW: chuẩn bị confusion matrix ---
num_classes = len(class_names)
label_to_index = {name: i for i, name in enumerate(class_names)}
confusion = np.zeros((num_classes, num_classes), dtype=int)
# ------------------------------------------------

# ===== Config =====
IMG_SIZE = 224
VAL_DIR = "dataset/val"
CONFIDENCE_THRESHOLD = 0.5

# ===== Helper: predict single image =====
def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize và tiền xử lý
    img_array = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_array = np.expand_dims(img_array, axis=0)
    # Vẫn giữ tiền xử lý này, ĐÃ KIỂM TRA LÀ KHỚP VỚI train.py 
    # (vì train.py đã được sửa để áp dụng preprocess_input qua map)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    preds = model.predict(img_array, verbose=0)
    class_idx = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0])) # Chuyển sang float để hiển thị dễ hơn
    predicted_label = class_names[class_idx] # Nhãn dự đoán cao nhất (dùng để tính Accuracy)
    
    display_label = predicted_label if confidence >= CONFIDENCE_THRESHOLD else "Unknown"
    
    # Trả về cả index dự đoán để cập nhật confusion matrix
    return class_idx, predicted_label, display_label, confidence

# ===== Iterate over validation images =====
total = 0
correct = 0

for folder in os.listdir(VAL_DIR):
    folder_path = os.path.join(VAL_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    print(f"\nTesting folder: {folder}")
    print("-" * 40)

    for img_file in os.listdir(folder_path):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        img_path = os.path.join(folder_path, img_file)
        pred_idx, predicted_label, display_label, conf = predict_image(img_path)
        
        total += 1
        if predicted_label == folder:
            correct += 1
        if folder in label_to_index:
            true_idx = label_to_index[folder]
            confusion[true_idx, pred_idx] += 1
        else:
            print(f"Warning: true label '{folder}' not found in class_names, skipping confusion update.")
            
        print(f"Image: {img_file:<30} → Pred: {display_label:<15} ({conf*100:.1f}%)  (argmax: {predicted_label})")

print("\n" + "="*50)
print(f"Final Accuracy: {(correct/total)*100:.2f}% ({correct}/{total})")

# --- Hiển thị Confusion Matrix ---
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(confusion, interpolation='nearest', cmap='Blues')
ax.figure.colorbar(im, ax=ax)
ax.set(
    xticks=np.arange(num_classes),
    yticks=np.arange(num_classes),
    xticklabels=class_names,
    yticklabels=class_names,
    ylabel='True label',
    xlabel='Predicted label',
    title='Confusion Matrix'
)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

# Annotate counts
thresh = confusion.max() / 2 if confusion.max() > 0 else 0
for i in range(num_classes):
    for j in range(num_classes):
        ax.text(j, i, format(confusion[i, j], 'd'),
                ha="center", va="center",
                color="white" if confusion[i, j] > thresh else "black")

plt.tight_layout()
plt.show()

fig.savefig("confusion_matrix.png", dpi=150)
print("Saved confusion_matrix.png")
plt.close()