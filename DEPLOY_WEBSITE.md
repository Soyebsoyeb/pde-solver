# 🌐 Deploy Website to GitHub Pages

Your website is now ready to be deployed! Follow these steps:

## 📋 Quick Deploy (2 Minutes)

### Step 1: Push the docs folder

```bash
git add docs/
git commit -m "Add website for GitHub Pages deployment"
git push
```

### Step 2: Enable GitHub Pages

1. Go to your repository: **https://github.com/Soyebsoyeb/pde-solver**
2. Click **"Settings"** (top menu)
3. Click **"Pages"** (left sidebar)
4. Under "Source":
   - Select **"Deploy from a branch"**
   - Branch: **main**
   - Folder: **/docs**
5. Click **"Save"**

### Step 3: Wait & Access

- GitHub will build your site (takes 1-2 minutes)
- Your website will be live at:
  
  **https://soyebsoyeb.github.io/pde-solver/**

---

## ✅ What You'll Get

Your live website will feature:
- 🎨 Beautiful landing page with gradient design
- 📊 Interactive solver demo
- 📡 API documentation links
- 🔬 Features showcase
- 💻 Code examples
- 📈 Performance stats

---

## 🔧 Troubleshooting

### Website not showing up?

1. **Check build status:**
   - Go to "Actions" tab in your repository
   - Should see "pages build and deployment" workflow
   - Wait for green checkmark

2. **Force rebuild:**
   - Go to Settings → Pages
   - Change source to "None", save
   - Change back to "main" branch, "/docs" folder, save

3. **Check URL:**
   - Must be: `https://YOUR-USERNAME.github.io/REPO-NAME/`
   - Yours: `https://soyebsoyeb.github.io/pde-solver/`

### Need to make changes?

1. Edit `docs/index.html`
2. Commit and push
3. GitHub will automatically rebuild (takes 1-2 min)

---

## 🎨 Customize Your Website

The website file is at: `docs/index.html`

You can modify:
- Colors and styling (CSS in `<style>` section)
- Content and text
- Links and navigation
- Add more pages

After changes:
```bash
git add docs/index.html
git commit -m "Update website design"
git push
```

---

## 📱 Update README with Website Link

After deployment, add to your README.md:

```markdown
## 🌐 Live Demo

Visit the live website: **https://soyebsoyeb.github.io/pde-solver/**

- Interactive solver demo
- API documentation
- Feature showcase
- Quick start guide
```

---

## 🚀 Alternative: Deploy Full API

For the full FastAPI backend, you'll need a hosting service:

### Option 1: Railway.app (Easiest)
1. Sign up at railway.app
2. "New Project" → Deploy from GitHub
3. Select your repository
4. Automatically deployed!

### Option 2: Render.com
1. Sign up at render.com
2. "New Web Service"
3. Connect GitHub repo
4. Build command: `pip install -r requirements.txt`
5. Start command: `python start_server.py`

### Option 3: Heroku
1. Create `Procfile`:
   ```
   web: python start_server.py
   ```
2. Push to Heroku
3. Your API is live!

---

## ✨ What's the Difference?

### GitHub Pages (Static Site)
- ✅ FREE forever
- ✅ Fast and reliable
- ✅ Perfect for documentation/showcase
- ❌ No backend API calls
- ❌ Demo is simulated (no real solving)

### Full Deployment (Railway/Render)
- ✅ Full API functionality
- ✅ Real PDE solving
- ✅ Train and evaluate models
- 💰 May have costs after free tier
- ⚡ Requires server resources

---

## 🎯 Recommendation

**Start with GitHub Pages** for:
- Portfolio showcase
- Documentation
- Project demonstration
- No cost

**Later deploy full API** when:
- Need real computation
- Want production use
- Have users making requests

---

## 📞 Quick Reference

**Your Repository:** https://github.com/Soyebsoyeb/pde-solver  
**Your Website (after setup):** https://soyebsoyeb.github.io/pde-solver/  
**Documentation:** Check README.md for full guide  

---

## ✅ Checklist

- [ ] Run: `git add docs/`
- [ ] Run: `git commit -m "Add website for GitHub Pages"`
- [ ] Run: `git push`
- [ ] Go to GitHub → Settings → Pages
- [ ] Select main branch, /docs folder
- [ ] Click Save
- [ ] Wait 1-2 minutes
- [ ] Visit: https://soyebsoyeb.github.io/pde-solver/
- [ ] 🎉 Your website is live!

**You're just 5 clicks away from having a live website!** 🚀
