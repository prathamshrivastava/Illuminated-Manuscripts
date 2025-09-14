# 📜 Manuscript Image Captioning: Multimodal Deep Learning Approach

This project explores how **textual content in medieval manuscripts** can be computationally analyzed to generate **meaningful captions for associated images**. By integrating **Computer Vision (CV)** and **Natural Language Processing (NLP)**, the framework learns from manuscript images and their corresponding texts, capturing both visual and thematic information.

The motivation behind this work is that **manuscript illustrations cannot be fully understood by visual cues alone**; combining textual context ensures captions retain historical and thematic significance.

---

## 🔍 Research Goal

The goal of this project is to investigate the **relationship between manuscript text and illustrations**. Specifically, it addresses:

- How chapter texts in manuscripts can be summarized and analyzed computationally.  
- How extracted keywords and thematic summaries can inform automated caption generation.  
- How multimodal deep learning can enhance metadata creation for manuscript images.  

This approach contributes to **digital manuscript studies**, enabling semi-automated annotation and better understanding of illustrated historical texts.

---

## 🧠 Approach

### 1. Image Understanding
- Images are processed using **DenseNet201**, a pre-trained Convolutional Neural Network (CNN).  
- The CNN extracts high-level **visual features** capturing spatial and semantic information.

### 2. Text Understanding
- Manuscript texts are cleaned, tokenized, and converted to sequences.  
- Sequences are embedded and fed into an **LSTM network** to capture **contextual and thematic meaning**.

### 3. Multimodal Fusion
- Image and text features are **combined** into a joint representation.  
- The fused representation enables generation of captions that reflect both **visual content and textual themes**.

### 4. Training Optimization
The model is trained using best practices to ensure robust learning:  
- **Adam optimizer** for adaptive learning rates  
- **EarlyStopping** to prevent overfitting  
- **ModelCheckpoint** to save the best model  
- **ReduceLROnPlateau** to adjust learning rate dynamically  

### 5. Visualization & Analysis
- Training loss and validation loss trends are visualized with **Matplotlib** and **Seaborn**.  
- Results are analyzed to understand learning dynamics and improve model performance.

---



