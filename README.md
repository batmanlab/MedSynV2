# <div align="center">MedSynV2: Flexible Multimodal Controllable Generation of 3D Medical Images</div>

<p align="center">
<b>Weicheng Dai</b><sup>1</sup>,
Chenyu Wang<sup>1</sup>,
Shantanu Ghosh<sup>1</sup>,
Kayhan Batmanghelich<sup>1</sup><sup>*</sup>  
<br/>
<sup>1</sup>Department of Electrical and Computer Engineering, Boston University  
<br/>
<sup>*</sup>Corresponding author
</p>

<p align="center">
<i>ECCV 2026</i>
</p>

<p align="center">
<!-- <a href=""><img src="https://img.shields.io/badge/Paper-ECCV%202026-blue"/></a> -->
</p>

---

## 🧠 Overview

**MedSynV2** is a **flexible multimodal framework for controllable 3D medical image generation**, designed to overcome the limitations of existing text-only or segmentation-only conditioning methods.

Our model supports **optional and partial conditioning** from:
- 📝 **Radiology reports** (semantic, flexible)
- 🧩 **Segmentation prompts** (precise, spatial)

Crucially, MedSynV2 **does not require full-organ annotations**.  
Users may provide segmentation for a *specific anatomy or abnormality*, whose semantic meaning is specified through an accompanying text description.

This design enables **scalable, fine-grained control** over volumetric generation while maintaining high image fidelity.

---

## ✨ Key Features

- 🔀 **Multimodal conditioning**: text-only, segmentation-only, or both
- 🧩 **Partial segmentation support** (no full-organ masks required)
- 🧠 **Strong semantic grounding** via text-described segmentation prompts
- ⚡ **Memory-efficient diffusion transformer** for high-resolution 3D volumes
- 🔍 **Gated attention** for long radiology reports
- 🧬 Generates **anatomically consistent, high-resolution CT volumes**

---

## 🧩 Method Overview

<p align="center">
<img src="assets/medsynv2_overview.png" width="90%"/>
</p>

<p align="center">
<a href="assets/medsynv2_overview.pdf">📄 View full-resolution PDF</a>
</p>

MedSynV2 extends diffusion transformers to jointly process:
- Image tokens
- Segmentation tokens
- Text tokens from radiology reports

A gated attention mechanism enables effective conditioning on **long, unstructured clinical text**, while preserving spatial controllability through segmentation prompts.

---

## 📊 Results Summary

We evaluate MedSynV2 on **large-scale 3D CT datasets**, using:

- **Perceptual metrics** (FID, SSIM, PSNR)
- **Semantic consistency metrics**
- **Radiologist evaluation**

### Key findings:
- 🚀 **~24% relative improvement in mean FID**
- 🧠 Strong semantic alignment between generated and real CT volumes
- 🧬 High-resolution, anatomically coherent synthesis
- 📈 Improved **data efficiency** when used for data augmentation
- 🌱 Generalization to concepts beyond training distribution

---

## 📦 Code Status

The full release will include:
- Model architecture and diffusion transformer implementation
- Multimodal conditioning pipelines
- Inference scripts

---

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{dai2026medsynv2,
  title     = {Flexible Multimodal Controllable Generation of 3D Medical Images},
  author    = {Dai, Weicheng and Wang, Chenyu and Ghosh, Shantanu and Batmanghelich, Kayhan},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}