# ⚡ High-Precision Vision AI (CLIP Streamlit App)

An interactive, high-precision computer vision web application built with **Streamlit**, **PyTorch**, and Hugging Face's **CLIP Model** (`openai/clip-vit-large-patch14`). 

The application performs hierarchical image classification across 10 dataset classes with detailed sub-type and breed identification.

---

## 🚀 Features

- **Hierarchical Classification**: First detects the primary dataset category, then runs ensembled classification for fine-grained sub-types/breeds.
- **Dynamic Themes & UI**: Includes animated cyberpunk themes, interactive charts, and glassmorphism styling.
- **GPU & VRAM Optimized**: Built using PyTorch `float16` precision to prevent CUDA Out-Of-Memory (OOM) crashes.
- **Real-time Visualization**: Interactive probability distribution charts powered by Plotly.

---

## 🏷️ Supported Classes & Sub-Types

1. **Airplane** (Commercial Airliner, Fighter Jet, Helicopter, etc.)
2. **Automobile** (Sedan, SUV, Sports Car, Vintage Car, etc.)
3. **Bird** (Parrot, Eagle, Owl, Peacock, etc.)
4. **Cat** (Persian, Siamese, Maine Coon, Bengal, etc.)
5. **Deer** (White-tailed Deer, Reindeer, Elk, Moose, etc.)
6. **Dog** (Labrador, German Shepherd, Beagle, Poodle, etc.)
7. **Frog** (Tree Frog, Bullfrog, Poison Dart Frog, etc.)
8. **Horse** (Arabian, Thoroughbred, Mustang, Pony, etc.)
9. **Ship** (Cruise Ship, Cargo Ship, Yacht, Sailboat, etc.)
10. **Truck** (Pickup, Semi Truck, Monster Truck, Dump Truck, etc.)

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/ayan-baig-123/image-detector.git](https://github.com/ayan-baig-123/image-detector.git)
cd image-detector