# PyTorch Showcase - Portfolio Integration Documentation

## Overview
This document explains the PyTorch for Deep Learning Professional Certificate showcase integration into yasarkaankaya.com portfolio.

## Project Structure

### 1. PyTorch Repository
**Location**: `C:\Users\kaank\Pytorch`
**GitHub**: https://github.com/YKaanKaya/deeplearning-ai-pytorch

```
deeplearning-ai-pytorch/
├── Course-1-PyTorch-Exploration/
│   ├── Module 1/ (Introduction to PyTorch)
│   ├── Module 2/ (Neural Networks Fundamentals)
│   ├── Module 3/ (Deep Learning Techniques)
│   └── Module 4/ (Image Classification Projects)
├── Course-2-Techniques-and-Ecosystem/ (In Progress)
├── Course-3-Advanced-Architectures/ (Upcoming)
├── README.md
└── .gitignore
```

### 2. Portfolio Showcase Page
**Location**: `C:\Users\kaank\portfolio-check\app\showcase\PyTorchDeepLearning`
**GitHub**: https://github.com/YKaanKaya/yasarkaankaya
**Live URL**: https://yasarkaankaya.com/showcase/PyTorchDeepLearning

```
PyTorchDeepLearning/
├── components/
│   ├── ShowcaseApp.tsx     # Main showcase component
│   ├── CourseContent.tsx   # Course progress and structure
│   ├── TechStack.tsx       # Technologies and tools
│   └── GitHubRepo.tsx      # Repository information
└── page.tsx                # Page entry point with metadata
```

## Key Files and Their Purpose

### page.tsx
- Main entry point for the showcase page
- Contains SEO metadata (title, description, OpenGraph, Twitter cards)
- Wraps ShowcaseApp component
- **Update when**: Changing page title, description, or meta tags

### ShowcaseApp.tsx
- Main showcase layout and UI
- Hero section with course overview
- Key benefits cards (4 cards highlighting course features)
- Tab navigation for content sections
- **Update when**: Changing hero text, adding new tabs, or modifying layout

### CourseContent.tsx
- Displays 3 courses with completion status:
  - Course 1: Completed ✅
  - Course 2: In Progress 🔄
  - Course 3: Upcoming 📅
- Lists modules and learning outcomes
- **Update when**: Completing courses, adding new modules

### TechStack.tsx
- Lists technologies organized by category:
  - Deep Learning Framework (PyTorch, TorchVision)
  - NLP & Transformers (Hugging Face)
  - MLOps & Deployment (ONNX, MLflow)
  - Optimization Techniques
  - Development Tools
- **Update when**: Using new technologies or tools

### GitHubRepo.tsx
- Links to GitHub repository
- Shows repository structure
- Lists what's included in each course
- **Update when**: Repository structure changes

## Portfolio Homepage Integration

### Project Card Location
**File**: `C:\Users\kaank\portfolio-check\components\portfolio.tsx`
**Section**: Projects section (around line 375-420)

The PyTorch project card appears in the projects grid:
```tsx
{
  title: 'PyTorch for Deep Learning',
  description: 'Professional certificate from DeepLearning.AI...',
  link: 'https://github.com/YKaanKaya/deeplearning-ai-pytorch',
  demoLink: '/showcase/PyTorchDeepLearning',
  tags: ['PyTorch', 'Deep Learning', 'Computer Vision', 'NLP', 'MLOps', 'ONNX']
}
```

## How to Update Progress

### When Completing Course 2

1. **Update CourseContent.tsx**:
```tsx
{
  id: 2,
  title: "Course 2: PyTorch: Techniques and Ecosystem Tools",
  status: "completed", // Change from "in-progress" to "completed"
  ...
}
```

2. **Update Course 3 status**:
```tsx
{
  id: 3,
  title: "Course 3: Advanced Architectures and Deployment",
  status: "in-progress", // Change from "upcoming" to "in-progress"
  ...
}
```

3. **Update README.md in PyTorch repo**:
- Change Course 2 from 🔄 to ✅
- Update course description if needed

### Adding New Projects/Assignments

1. Add to the appropriate course folder in the PyTorch repository
2. Update README.md to reflect new content
3. No changes needed to the showcase page (unless you want to highlight specific projects)

## Development Workflow

### Local Development

```bash
# Portfolio
cd C:\Users\kaank\portfolio-check
npm install
npm run dev
# Visit: http://localhost:3000/showcase/PyTorchDeepLearning

# PyTorch Repo (if making changes)
cd C:\Users\kaank\Pytorch
git add .
git commit -m "Your message"
git push origin main
```

### Deployment

The portfolio auto-deploys via Netlify/Vercel when you push to GitHub:

```bash
cd C:\Users\kaank\portfolio-check
git add .
git commit -m "Update PyTorch showcase"
git push origin main
```

## Important Notes

### .gitignore Patterns
Large dataset files are excluded from the PyTorch repository:
- `*.tgz`, `*.tar.gz`, `*.zip`
- `EMNIST_DATA/`, `cifar_*/`, `flower_data/`, `plants_dataset/`
- Model files: `*.pth`, `*.pt`, `*.ckpt`

### Known Issues

**Home Page Compilation Error**: There's a pre-existing syntax error in `components/portfolio.tsx` that affects the home page but NOT the PyTorch showcase page. The showcase works perfectly.

**Workaround if needed**:
```bash
rm -rf .next  # Clear Next.js cache
npm run dev   # Restart dev server
```

## File Locations Quick Reference

| Item | Location |
|------|----------|
| PyTorch Repo | `C:\Users\kaank\Pytorch` |
| Portfolio Repo | `C:\Users\kaank\portfolio-check` |
| Showcase Page | `app/showcase/PyTorchDeepLearning/` |
| Project Card | `components/portfolio.tsx` (line ~390) |
| Course Content | `PyTorchDeepLearning/components/CourseContent.tsx` |

## Future Enhancements

Ideas for future updates:
- Add interactive code demos
- Include project screenshots/videos
- Add certificate image when completed
- Create a "Projects" tab with specific assignments
- Add performance metrics or model results

## Links

- **PyTorch Repo**: https://github.com/YKaanKaya/deeplearning-ai-pytorch
- **Portfolio Repo**: https://github.com/YKaanKaya/yasarkaankaya
- **Live Showcase**: https://yasarkaankaya.com/showcase/PyTorchDeepLearning
- **Course Program**: https://learn.deeplearning.ai/specializations/pytorch-for-deep-learning-professional-certificate/overview

## Contact

For questions or issues, refer to:
- Portfolio issues: https://github.com/YKaanKaya/yasarkaankaya/issues
- PyTorch repo issues: https://github.com/YKaanKaya/deeplearning-ai-pytorch/issues

---

*Last Updated: November 9, 2025*
*Session: Initial PyTorch showcase creation*
