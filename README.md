# MonasteryAI ☸️🏛️

> **AI-Powered 3D Monastic Cultural Heritage Preservation & Interactive WebGL Walkthrough Engine**

MonasteryAI is a cutting-edge platform for digitizing, reconstructing, and experiencing sacred Buddhist monasteries and monuments in high-fidelity 3D WebGL. It pairs state-of-the-art **Photogrammetric Vision Pipelines (Multi-View Stereo & 3D Gaussian Splatting)** with an **offline-capable on-device Monastic AI Guide** and **spatial audio beacons**.

---

## 🌟 Key Features

1. **Photogrammetric 3D Reconstruction Pipeline:**
   * Automated keyframe extraction and crowd filtering from drone footage.
   * Multi-View Stereo (MVS) & Structure-from-Motion (SfM) point cloud / mesh synthesis.
   * Direct export to binary glTF (`.glb`) and 3D Gaussian Splats (`.splat`).

2. **Interactive 3D WebGL Client:**
   * **First-Person Walkthrough:** Realistic ground physics, footstep head-bobbing, sprinting, and collision detection.
   * **Aerial Orbit Mode:** Smooth panoramic bird's-eye camera rotation around 3D sacred monuments.
   * **Mini-Radar HUD & Compass:** Real-time spatial tracking with cardinal direction indicators and player coordinates.
   * **Proximity Beacons & Spatial Audio:** Interactive historical beacons with automatic audio narration upon approach.

3. **Cloud CDN & Vector Database:**
   * High-speed global asset streaming via **Supabase Storage CDN**.
   * Monastic cultural heritage metadata and embeddings repository.

---

## 🏛️ Reconstructed Heritage Sites

* **Namdroling Golden Temple (Main Shrine, Bylakuppe):** Grand central staircase, red colonnade, painted Tibetan murals, and gilded roof pinnacles.
* **The Great Emei Mountain (Golden Summit, Sichuan):** 48-meter Golden Bodhisattva Samantabhadra statue seated on four white elephants.
* **Pemayangtse Monastery (Pelling, West Sikkim):** Historic 1705 Nyingma monastery and Sangtokpalri celestial wooden masterpiece.
* **Rumtek Monastery (Dharma Chakra Centre, East Sikkim):** Traditional Tibetan Dukhang shrine hall and central sacred courtyard pillar.

---

## 🚀 Quick Start

### 1. Run the WebGL Experience Locally
```bash
# Start a simple HTTP server
python -m http.server 8080
```
Open **`http://localhost:8080/index.html`** in your browser.

### 2. Configure Environment
Copy `.env.example` to `.env` and fill in your credentials:
```bash
cp .env.example .env
```

### 3. Reconstruct 3D Models
```bash
# Structural Architectural 3D Reconstruction
python reconstruct_namdroling_temple_3d.py

# Multi-View Stereo Photogrammetry
python reconstruct_curated_mvs.py

# Rumtek Shrine Reconstruction
python reconstruct_rumtek_shrine_3d.py
```

---

## 🛠️ Tech Stack

* **Frontend:** Three.js (WebGL), Vanilla HTML5/CSS3 (Glassmorphism & Gold Theme), Web Speech API.
* **Computer Vision & 3D:** OpenCV, NumPy, Multi-View Stereo, Structure-from-Motion, Trimesh.
* **Backend & Cloud:** Supabase (Storage CDN & Database), Kaggle GPU Cloud Kernels.

---

## 📜 License
MIT License. Built for cultural heritage preservation and immersive education.
